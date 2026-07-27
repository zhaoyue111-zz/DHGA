from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest

import torch

from dhga.checkpoint import load_dhga_checkpoint, load_training_checkpoint, save_dhga_checkpoint, save_training_checkpoint
from dhga.config import DHGAConfig
from dhga.experts import AppearanceExpert, AppearanceFeatureAdapter, SemanticExpert
from dhga.geometry.boundary_corruption import make_bidirectional_corruption, make_local_boundary_corruption
from dhga.geometry.ray_sampler import make_ray_offsets_mm, sample_along_normals
from dhga.geometry.transport_head import GeometryTransportHead
from dhga.geometry.boundary_points import extract_boundary_points, sparse_displacements_to_dense_narrowband
from dhga.geometry.sdf import mask_to_sdf, sdf_normals, update_sdf_with_displacement
from dhga.inference import finalize_mask, finalize_probability
from dhga.losses import cross_supervision_loss
from dhga.routing import DisagreementRouter
from dhga.shared_voxtell import SharedEncoderOnce
from dhga.trainer import DHGASmokeModel, run_synthetic_smoke
from dhga.trainer import DHGAStageTrainer
from dhga.teacher import EMATeacher
from dhga.evaluation import compute_binary_case_metrics, connected_components_3d, spacing_from_reader_properties
from dhga.voxtell_model import PromptConditionedRayTokens


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

    def test_sparse_to_dense_uses_scatter_add_for_duplicate_indices(self):
        points = torch.tensor([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]])
        disp = torch.tensor([[1.0, 3.0]])
        valid = torch.tensor([[True, True]])
        dense, weight = sparse_displacements_to_dense_narrowband(
            points,
            disp,
            valid,
            (5, 5, 5),
            spacing=(1.0, 1.0, 1.0),
            diffusion_mm=0.1,
            return_weight=True,
        )
        self.assertAlmostEqual(float(dense[0, 0, 2, 2, 2]), 2.0, places=5)
        self.assertAlmostEqual(float(weight[0, 0, 2, 2, 2]), 2.0, places=5)

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

    def test_appearance_adapter_groupnorm_explicit_selected_index_and_single_dropout(self):
        adapter = AppearanceFeatureAdapter(4, hidden_ratio=0.5, dropout=0.5)
        self.assertFalse(any(isinstance(module, torch.nn.InstanceNorm3d) for module in adapter.modules()))
        self.assertFalse(any(isinstance(module, torch.nn.Dropout3d) for module in adapter.modules()))
        features = SharedEncoderOnce(torch.nn.Conv3d(1, 4, 1))(torch.randn(1, 1, 3, 3, 3))
        features = features.__class__(
            image=features.image,
            encoder_stages=[features.encoder_stages[0], features.encoder_stages[0] + 1.0],
            selected_feature=features.encoder_stages[0],
            metadata={"selected_feature_idx": 1},
        )
        expert = AppearanceExpert([4, 4], [1], dropout=0.0)
        adapted = expert.adapt_features(features, feature_dropout=0.0)
        self.assertIs(adapted.selected_feature, adapted.encoder_stages[1])

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

    def test_unified_finalize_restores_region_when_gate_or_displacement_zero(self):
        config = DHGAConfig(dhga_geometry_enabled=True)
        region = torch.rand(1, 1, 4, 4, 4)
        sdf = torch.randn_like(region)
        displaced = finalize_probability(region, sdf, torch.ones_like(region), torch.zeros_like(region), config)
        self.assertTrue(torch.equal(displaced, region))
        displaced = finalize_probability(region, sdf, torch.zeros_like(region), torch.ones_like(region), config)
        self.assertTrue(torch.equal(displaced, region))

    def test_config_json_explicit_cli_overrides(self):
        import run_3d_dhga

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(DHGAConfig(dhga_stage="B", epochs=1, data_dir="json_data").to_dict()))
            old_argv = sys.argv
            try:
                sys.argv = [
                    "run_3d_dhga.py",
                    "--config_json",
                    str(path),
                    "--dhga_stage",
                    "C",
                    "--epochs",
                    "7",
                    "--data_dir",
                    "cli_data",
                    "--no_amp",
                ]
                args = run_3d_dhga.parse_args()
                config = run_3d_dhga.config_from_args(args)
            finally:
                sys.argv = old_argv
        self.assertEqual(config.dhga_stage, "C")
        self.assertEqual(config.epochs, 7)
        self.assertEqual(config.data_dir, "cli_data")
        self.assertFalse(config.amp)

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

    def test_ema_teacher_hard_sync_after_init_checkpoint(self):
        model = DHGASmokeModel(DHGAConfig())
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.router.parameters():
            param.requires_grad_(True)
        ema = EMATeacher(model, decay=0.99)
        with torch.no_grad():
            for param in model.router.parameters():
                param.add_(10.0)
        ema.sync_from(model)
        params = dict(model.named_parameters())
        for name in ema.names:
            self.assertTrue(torch.allclose(ema.shadow[name], params[name]))

    def test_stage_c_loss_combination_applies_weights_once(self):
        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.config = DHGAConfig(dhga_boundary_recovery_weight=2.0, dhga_minimal_transport_weight=3.0)
        recovery = torch.tensor(5.0)
        minimal = torch.tensor(7.0)
        self.assertEqual(float(trainer._combine_stage_c_loss(recovery, minimal)), 31.0)

    def test_stage_d_router_target_loss_backpropagates_to_router(self):
        router = DisagreementRouter("none")
        sem = torch.rand(1, 1, 4, 4, 4)
        app = torch.rand_like(sem)
        student = SimpleNamespace(router=router(sem, app))
        teacher = SimpleNamespace(
            semantic_prob=sem.detach(),
            appearance_prob=app.detach(),
            anchor_prob=(0.5 * (sem + app)).detach(),
            router=router(sem.detach(), app.detach()),
        )
        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        loss = trainer._stage_d_router_target_loss(student, teacher)
        loss.backward()
        grads = [param.grad for param in router.parameters() if param.requires_grad]
        self.assertTrue(any(grad is not None and bool((grad.abs() > 0).any()) for grad in grads))

    def test_stage_a_uses_baseline_forward(self):
        class FakeBaseline(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.called = False

            def baseline_forward(self, patch, spacing):
                self.called = True
                prob = torch.full((1, 1, 2, 2, 2), 0.5)
                return SimpleNamespace(
                    semantic_prob=prob,
                    appearance_prob=prob,
                    anchor_prob=prob,
                    router=DisagreementRouter("none")(prob, prob),
                )

        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.model = FakeBaseline()
        loss, metrics = trainer._stage_a(torch.zeros(1, 1, 2, 2, 2), (1.0, 1.0, 1.0))
        self.assertTrue(trainer.model.called)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(metrics["dhga_stage_a_forced_baseline"], 1.0)

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
        self.assertEqual(float(out["entropy"].detach().abs().max()), 0.0)

    def test_truncated_ray_initialization_stays_near_zero(self):
        offsets = make_ray_offsets_mm(6.0, 2.0)
        head = GeometryTransportHead(3, offsets)
        tokens = torch.randn(1, 1, offsets.numel(), 3)
        valid = offsets.view(1, 1, -1) >= 0
        out = head(tokens, valid)
        self.assertLess(float(out["expected_displacement_mm"].detach().abs().max()), 1e-3)

    def test_smoke_geometry_token_dim_includes_geo_gate(self):
        maker = PromptConditionedRayTokens(prompt_dim=5, hidden_dim=16)
        scalar = torch.zeros(1, 2, 3, 1)
        visual = torch.zeros(1, 2, 3, 2)
        tokens = maker(
            scalar,
            scalar,
            scalar,
            scalar,
            scalar,
            scalar,
            scalar,
            visual,
            torch.tensor([-1.0, 0.0, 1.0]),
            torch.zeros(1, 1, 5),
        )
        self.assertEqual(tokens.shape[-1], 1 + 1 + 1 + 1 + 1 + 1 + 1 + 2 + 1 + 16)

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

    def test_synthetic_evaluation_metrics_complete(self):
        import numpy as np

        gt = np.zeros((3, 3, 3), dtype=bool)
        pred = np.zeros_like(gt)
        sem = np.zeros_like(gt)
        app = np.zeros_like(gt)
        gt[1, 1, 1] = True
        pred[1, 1, 1] = True
        pred[0, 0, 0] = True
        sem[1, 1, 1] = True
        app[0, 0, 0] = True
        metrics = compute_binary_case_metrics(pred, gt, sem, app)
        for key in (
            "dice",
            "iou",
            "precision",
            "recall",
            "fp_voxels",
            "fn_voxels",
            "pred_gt_volume_ratio",
            "connected_components",
            "oracle_union_dice",
            "oracle_intersection_iou",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["fp_voxels"], 1)
        self.assertEqual(metrics["fn_voxels"], 0)

    def test_synthetic_smoke(self):
        result = run_synthetic_smoke(DHGAConfig(), "cpu")
        self.assertEqual(result.shared_encoder_calls, 1)
        self.assertGreaterEqual(result.loss, 0.0)
        self.assertIn("dhga_expert_corr", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
