from pathlib import Path
import tempfile
import unittest

import torch

from dhga.checkpoint import load_dhga_checkpoint, load_training_checkpoint, save_dhga_checkpoint, save_training_checkpoint
from dhga.config import DHGAConfig
from dhga.experts import AppearanceExpert, SemanticExpert
from dhga.geometry.boundary_corruption import make_bidirectional_corruption, make_local_boundary_corruption
from dhga.geometry.ray_sampler import make_ray_offsets_mm, sample_along_normals
from dhga.geometry.transport_head import GeometryTransportHead
from dhga.geometry.boundary_points import extract_boundary_points, sparse_displacements_to_dense_narrowband
from dhga.geometry.sdf import mask_to_sdf, sdf_normals, update_sdf_with_displacement
from dhga.inference import finalize_mask
from dhga.losses import cross_supervision_loss
from dhga.routing import DisagreementRouter
from dhga.shared_voxtell import SharedEncoderOnce
from dhga.trainer import DHGASmokeModel, run_synthetic_smoke
from dhga.trainer import DHGAStageTrainer
from dhga.teacher import EMATeacher
from dhga.evaluation import connected_components_3d, spacing_from_reader_properties


class DHGAStaticTests(unittest.TestCase):
    def test_sdf_sign_and_displacement(self):
        mask = torch.zeros(1, 1, 9, 9, 9, dtype=torch.bool)
        mask[..., 3:6, 3:6, 3:6] = True
        sdf = mask_to_sdf(mask)
        self.assertLess(float(sdf[..., 4, 4, 4]), 0.0)
        self.assertGreater(float(sdf[..., 0, 0, 0]), 0.0)
        outward = update_sdf_with_displacement(sdf, torch.ones_like(sdf))
        self.assertLess(float(outward[..., 2, 4, 4]), float(sdf[..., 2, 4, 4]))

    def test_sdf_normals_point_outward(self):
        z = torch.arange(7).view(1, 1, 7, 1, 1).float()
        sdf = z - 3.0
        sdf = sdf.expand(1, 1, 7, 5, 5).contiguous()
        normals = sdf_normals(sdf)
        self.assertGreater(float(normals[..., 3, 2, 2, 0]), 0.99)

    def test_boundary_points_and_dense_displacement(self):
        mask = torch.zeros(1, 1, 7, 7, 7, dtype=torch.bool)
        mask[..., 2:5, 2:5, 2:5] = True
        sdf = mask_to_sdf(mask)
        points, normals, valid = extract_boundary_points(sdf, 1.5, 32)
        self.assertEqual(points.shape, (1, 32, 3))
        self.assertEqual(normals.shape, (1, 32, 3))
        dense = sparse_displacements_to_dense_narrowband(points, torch.ones(1, 32), valid, (7, 7, 7), spacing=(1.0, 1.0, 2.0), diffusion_mm=2.0)
        self.assertGreater(float(dense.abs().sum()), 0.0)

    def test_empty_or_full_mask_has_no_boundary(self):
        sdf = mask_to_sdf(torch.zeros(1, 1, 5, 5, 5, dtype=torch.bool))
        _, _, valid = extract_boundary_points(sdf, 1.0, 8)
        self.assertFalse(bool(valid.any()))
        sdf = mask_to_sdf(torch.ones(1, 1, 5, 5, 5, dtype=torch.bool))
        _, _, valid = extract_boundary_points(sdf, 1.0, 8)
        self.assertFalse(bool(valid.any()))

    def test_ray_sampler_coordinate_order_and_spacing(self):
        volume = torch.zeros(1, 1, 5, 6, 7)
        volume[..., 2, 3, 4] = 1.0
        points = torch.tensor([[[2.0, 3.0, 3.0]]])
        normals = torch.tensor([[[0.0, 0.0, 1.0]]])
        offsets = torch.tensor([0.0, 2.0])
        samples, valid = sample_along_normals(volume, points, normals, offsets, spacing=(1.0, 1.0, 2.0))
        self.assertTrue(bool(valid.all()))
        self.assertAlmostEqual(float(samples[0, 0, 1, 0]), 1.0, places=5)

    def test_ray_offsets_do_not_exceed_radius_and_include_zero(self):
        offsets = make_ray_offsets_mm(5.5, 2.0)
        self.assertTrue(bool((offsets.abs() <= 5.5 + 1e-6).all()))
        self.assertTrue(bool((offsets == 0).any()))

    def test_bidirectional_corruption_recovery_sign(self):
        sdf = torch.zeros(8, 1, 7, 7, 7)
        corrupted, target, _ = make_bidirectional_corruption(sdf, 2.0, ["outward", "inward"])
        perturb = sdf - corrupted
        nonzero = perturb.abs() > 0
        self.assertTrue(bool(nonzero.any()))
        self.assertTrue(torch.allclose(target[nonzero], -perturb[nonzero]))

    def test_local_corruption_has_positive_negative_zero_regions(self):
        sdf = torch.randn(2, 1, 12, 12, 12)
        corrupted, target, choices = make_local_boundary_corruption(sdf, 2.0, ["inward", "outward", "zero"])
        perturb = sdf - corrupted
        self.assertTrue(bool((perturb > 0).any()))
        self.assertTrue(bool((perturb < 0).any()))
        self.assertTrue(bool((perturb == 0).any()))
        self.assertTrue(torch.allclose(target, -perturb))
        self.assertEqual(choices.shape[-1], 3)

    def test_shared_encoder_called_once_for_two_experts(self):
        shared = SharedEncoderOnce(torch.nn.Conv3d(1, 2, 1))
        features = shared(torch.randn(1, 1, 4, 4, 4))
        sem = SemanticExpert()
        app = AppearanceExpert([2], [0])
        sem(features, torch.randn(1, 1, 4, 4, 4))
        app(features, torch.randn(1, 1, 4, 4, 4))
        self.assertEqual(shared.num_calls, 1)

    def test_expert_trainable_sets_are_distinct(self):
        config = DHGAConfig()
        model = DHGASmokeModel(config)
        sem_ids = {id(p) for p in model.semantic_decoder.parameters() if p.requires_grad}
        app_ids = {id(p) for p in model.appearance_expert.parameters() if p.requires_grad}
        self.assertTrue(sem_ids)
        self.assertTrue(app_ids)
        self.assertFalse(sem_ids & app_ids)

    def test_disagreement_not_cross_supervised(self):
        sem = torch.ones(1, 1, 4, 4, 4) * 0.95
        app = torch.ones_like(sem) * 0.05
        router = DisagreementRouter("none")(sem, app)
        loss = cross_supervision_loss(sem, app, router)
        self.assertLess(float(loss), 0.01)

    def test_router_region_probability_renormalizes_without_geo_suppression(self):
        sem = torch.ones(1, 1, 3, 3, 3) * 0.8
        app = torch.ones_like(sem) * 0.2
        router = DisagreementRouter("none")
        with torch.no_grad():
            router.spatial_head[-1].bias[:] = torch.tensor([0.0, 0.0, 5.0])
        out = router(sem, app)
        self.assertGreater(float(out.w_geo.detach().mean()), 0.9)
        self.assertAlmostEqual(float(out.fused_prob.detach().mean()), 0.5, places=4)

    def test_router_three_way_weights_sum_and_initial_geo_gate_small(self):
        sem = torch.rand(1, 1, 4, 4, 4)
        app = torch.rand_like(sem)
        out = DisagreementRouter("none")(sem, app)
        self.assertTrue(torch.allclose(out.w_sem + out.w_app + out.w_geo, torch.ones_like(out.w_sem), atol=1e-6))
        self.assertLess(float(out.w_geo.detach().mean()), 0.05)

    def test_router_accepts_visual_context(self):
        sem = torch.rand(1, 1, 3, 3, 3)
        app = torch.rand_like(sem)
        visual = torch.randn(1, 4, 2, 2, 2)
        out = DisagreementRouter("none")(sem, app, visual_context=visual)
        self.assertEqual(out.fused_prob.shape, sem.shape)

    def test_spacing_properties_are_not_reversed(self):
        self.assertEqual(spacing_from_reader_properties({"spacing": (0.7, 0.8, 2.5)}), (0.7, 0.8, 2.5))

    def test_identity_geometry_keeps_mask(self):
        config = DHGAConfig(dhga_geometry_enabled=True)
        prob = torch.rand(1, 1, 5, 5, 5)
        initial = prob >= config.pred_threshold
        final = finalize_mask(prob, torch.zeros_like(prob), config)
        self.assertTrue(torch.equal(initial, final))

    def test_checkpoint_strict_roundtrip(self):
        config = DHGAConfig()
        model = DHGASmokeModel(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dhga.pt"
            modules = {"model": model}
            save_dhga_checkpoint(path, modules, config, {"stage": "test"})
            payload = load_dhga_checkpoint(path, modules)
        self.assertEqual(payload["format"], "dhga_checkpoint_v1")

    def test_config_rejects_init_and_resume_together(self):
        with self.assertRaises(ValueError):
            DHGAConfig(init_checkpoint="a.pt", resume_checkpoint="b.pt")

    def test_resume_checkpoint_checks_stage(self):
        config = DHGAConfig(dhga_stage="B")
        model = DHGASmokeModel(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.pt"
            save_training_checkpoint(path, model, config, metadata={"stage": "B"})
            with self.assertRaises(RuntimeError):
                load_training_checkpoint(path, model, expected_stage="C")

    def test_training_checkpoint_restores_optimizer_ema_scaler_and_step(self):
        config = DHGAConfig()
        model = DHGASmokeModel(config)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.pt"
            save_training_checkpoint(path, model, config, optimizer=optim, ema=model, scaler=scaler, epoch=3, global_step=17)
            payload = load_training_checkpoint(path, model, optim, model, scaler)
        self.assertEqual(int(payload["epoch"]), 3)
        self.assertEqual(int(payload["global_step"]), 17)

    def test_ema_teacher_tracks_only_trainable_parameters(self):
        config = DHGAConfig()
        model = DHGASmokeModel(config)
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.router.parameters():
            param.requires_grad_(True)
        ema = EMATeacher(model)
        self.assertTrue(ema.names)
        self.assertTrue(all(name.startswith("router.") for name in ema.names))

    def test_stage_trainable_parameter_sets(self):
        class FakeLoRA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(1, 1)
                self.delta = torch.nn.Linear(1, 1)

        class FakeStageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.injected_lora = torch.nn.ModuleDict({"x": FakeLoRA()})
                self.appearance_expert = torch.nn.Linear(1, 1)
                self.router = torch.nn.Conv3d(1, 1, 1)
                self.geometry_head = torch.nn.Linear(1, 1)
                self.ray_tokens = torch.nn.Linear(1, 1)
                self.geometry_visual_proj = torch.nn.Conv3d(1, 1, 1)

        expected = {
            "A": set(),
            "B": {"injected_lora.x.delta.weight", "injected_lora.x.delta.bias", "appearance_expert.weight", "appearance_expert.bias", "router.weight", "router.bias"},
            "C": {"geometry_head.weight", "geometry_head.bias", "ray_tokens.weight", "ray_tokens.bias", "geometry_visual_proj.weight", "geometry_visual_proj.bias"},
            "D": {"router.weight", "router.bias", "geometry_head.weight", "geometry_head.bias", "ray_tokens.weight", "ray_tokens.bias", "geometry_visual_proj.weight", "geometry_visual_proj.bias"},
        }
        for stage, names in expected.items():
            trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
            trainer.config = DHGAConfig(dhga_stage=stage)
            trainer.model = FakeStageModel()
            trainer._set_stage_trainability()
            trainable = {name for name, param in trainer.model.named_parameters() if param.requires_grad}
            self.assertEqual(trainable, names)

    def test_invalid_ray_keeps_near_zero_displacement(self):
        offsets = make_ray_offsets_mm(2.0, 1.0)
        head = GeometryTransportHead(3, offsets)
        tokens = torch.randn(1, 4, offsets.numel(), 3)
        valid = torch.zeros(1, 4, offsets.numel(), dtype=torch.bool)
        out = head(tokens, valid)
        self.assertEqual(float(out["expected_displacement_mm"].detach().abs().max()), 0.0)
        self.assertFalse(bool(out["valid_points"].any()))

    def test_geometry_chunk_matches_full(self):
        offsets = make_ray_offsets_mm(2.0, 1.0)
        head = GeometryTransportHead(3, offsets)
        tokens = torch.randn(1, 6, offsets.numel(), 3)
        valid = torch.ones(1, 6, offsets.numel(), dtype=torch.bool)
        full = head(tokens, valid)["expected_displacement_mm"]
        parts = []
        for start in range(0, 6, 2):
            parts.append(head(tokens[:, start:start + 2], valid[:, start:start + 2])["expected_displacement_mm"])
        self.assertTrue(torch.allclose(full, torch.cat(parts, dim=1)))

    def test_connected_components_synthetic_volume(self):
        import numpy as np

        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[0, 0, 0] = True
        mask[4, 4, 4] = True
        self.assertEqual(connected_components_3d(mask), 2)

    def test_synthetic_smoke(self):
        result = run_synthetic_smoke(DHGAConfig(), "cpu")
        self.assertEqual(result.shared_encoder_calls, 1)
        self.assertGreaterEqual(result.loss, 0.0)
        self.assertIn("dhga_expert_corr", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
