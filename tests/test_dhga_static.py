from pathlib import Path
from types import SimpleNamespace
import inspect
import json
import random
import sys
import tempfile
import unittest

import torch

from dhga.checkpoint import STAGE_C_FRESH_GEOMETRY_PREFIXES, load_dhga_checkpoint, load_training_checkpoint, save_dhga_checkpoint, save_training_checkpoint
from dhga.config import DHGAConfig
from dhga.experts import AppearanceExpert, AppearanceFeatureAdapter, SemanticExpert
from dhga.geometry.boundary_corruption import make_bidirectional_corruption, make_local_boundary_corruption
from dhga.geometry.ray_sampler import make_ray_offsets_mm, sample_along_normals
from dhga.geometry.transport_head import GeometryTransportHead
from dhga.geometry.boundary_points import extract_boundary_points, sparse_displacements_to_dense_narrowband
from dhga.geometry.sdf import mask_to_sdf, sdf_normals, update_sdf_with_displacement
from dhga.inference import finalize_mask, finalize_probability, geometry_effective_gate
from dhga.losses import cross_supervision_loss, weighted_bce_prob
from dhga.routing import DisagreementRouter
from dhga.shared_voxtell import SharedEncoderOnce
from dhga.trainer import DHGASmokeModel, run_synthetic_smoke, signed_displacement_metrics
from dhga.trainer import DHGAStageTrainer
from dhga.teacher import EMATeacher
from dhga.text_layer_ensemble import fuse_text_layer_ensemble, summarize_layers, text_layer_training_loss
from dhga.evaluation import compute_binary_case_metrics, compute_geometry_case_metrics, compute_raw_disagreement_metrics, connected_components_3d, spacing_from_reader_properties, surface_distance_metrics, write_float_volume_like_reader
from dhga.voxtell_model import DHGAVoxTellModel, PromptConditionedRayTokens


class FakeVoxTellEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = None
        self.stages = torch.nn.ModuleList([
            torch.nn.Conv3d(1, 4, 3, padding=1),
            torch.nn.Conv3d(4, 8, 3, stride=2, padding=1),
        ])


class FakeVoxTellDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(8, 1)
        self.multihead_attn = torch.nn.MultiheadAttention(8, 1)


class FakeVoxTellTransformerDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeVoxTellDecoderLayer()])

    def forward(self, tgt, memory, pos=None, memory_key_padding_mask=None):
        x = tgt
        memory_with_pos = memory if pos is None else memory + pos
        for layer in self.layers:
            x, _ = layer.multihead_attn(x, memory_with_pos, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)
        return x, None


class FakeVoxTellNativeDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deep_supervision = False
        self.low = torch.nn.Conv3d(8, 1, 1)
        self.high = torch.nn.Conv3d(4, 1, 1)

    def forward(self, skips, prompt_embeds):
        prompt = prompt_embeds[-1].mean(dim=(1, 2)).view(skips[-1].shape[0], 1, 1, 1, 1)
        low = self.low(skips[-1]) + prompt
        high = self.high(skips[0]) + torch.nn.functional.interpolate(low, size=skips[0].shape[-3:], mode="trilinear", align_corners=False)
        outputs = [high, low]
        return outputs if self.deep_supervision else outputs[:1]


class FakeVoxTellNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeVoxTellEncoder()
        self.selected_decoder_layer = 1
        self.project_bottleneck_embed = torch.nn.Linear(8, 8)
        self.project_text_embed = torch.nn.Linear(8, 8)
        self.transformer_decoder = FakeVoxTellTransformerDecoder()
        self.project_to_decoder_channels = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
        self.decoder = FakeVoxTellNativeDecoder()
        self.pos_embed = torch.nn.Parameter(torch.zeros(64, 1, 8), requires_grad=False)


def make_fake_text_layer_model(config: DHGAConfig | None = None) -> DHGAVoxTellModel:
    cfg = config or DHGAConfig(
        dhga_stage="B",
        dhga_stage_b_method="text_layer_ensemble",
        dhga_geometry_enabled=False,
        dhga_appearance_feature_layers=[0, 1],
    )
    text = torch.randn(1, 1, 8)
    return DHGAVoxTellModel(FakeVoxTellNetwork(), text, cfg, num_classes=1, num_templates=1)


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

    def test_geometry_transport_head_zero_prior_is_mild(self):
        offsets = torch.arange(-3.0, 4.0)
        head = GeometryTransportHead(1, offsets)
        self.assertAlmostEqual(float(head.center_bias.detach()), 2.0, places=5)
        self.assertTrue(torch.allclose(head.zero_displacement_prior, -offsets.abs()))
        out = head(torch.zeros(1, 1, offsets.numel(), 1))
        center_prob = float(out["prob"][0, 0, head.center_index].detach())
        nonzero_prob = float(out["prob"][0, 0][offsets != 0].sum().detach())
        self.assertGreater(nonzero_prob, 0.1)
        self.assertLess(center_prob, 0.9)

    def test_signed_displacement_metrics_use_only_valid_points(self):
        displacement = torch.tensor([[0.50, -0.40, 0.10, 100.0, float("nan")]])
        valid = torch.tensor([[True, True, True, False, False]])
        metrics = signed_displacement_metrics(displacement, valid, 0.25, "dhga_target")
        self.assertAlmostEqual(metrics["dhga_target_abs_displacement_mm"], (0.50 + 0.40 + 0.10) / 3.0, places=6)
        self.assertAlmostEqual(metrics["dhga_target_positive_displacement_ratio"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["dhga_target_negative_displacement_ratio"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["dhga_target_near_zero_displacement_ratio"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["dhga_target_nonzero_displacement_ratio"], 2.0 / 3.0, places=6)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_signed_displacement_metrics_zero_when_no_valid_points(self):
        displacement = torch.tensor([[float("nan"), 3.0, -3.0]])
        valid = torch.zeros_like(displacement, dtype=torch.bool)
        metrics = signed_displacement_metrics(displacement, valid, 0.25, "dhga_target")
        self.assertTrue(all(value == 0.0 for value in metrics.values()))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_signed_displacement_metrics_use_same_rules_for_prediction_and_target(self):
        values = torch.tensor([[-0.26, -0.25, 0.0, 0.25, 0.26]])
        valid = torch.ones_like(values, dtype=torch.bool)
        target = signed_displacement_metrics(values, valid, 0.25, "dhga_target")
        pred = signed_displacement_metrics(values, valid, 0.25, "dhga_pred")
        for suffix in (
            "abs_displacement_mm",
            "positive_displacement_ratio",
            "negative_displacement_ratio",
            "near_zero_displacement_ratio",
            "nonzero_displacement_ratio",
        ):
            self.assertAlmostEqual(target[f"dhga_target_{suffix}"], pred[f"dhga_pred_{suffix}"], places=6)
        self.assertAlmostEqual(target["dhga_target_positive_displacement_ratio"], 1.0 / 5.0, places=6)
        self.assertAlmostEqual(target["dhga_target_negative_displacement_ratio"], 1.0 / 5.0, places=6)
        self.assertAlmostEqual(target["dhga_target_near_zero_displacement_ratio"], 3.0 / 5.0, places=6)
        self.assertAlmostEqual(target["dhga_target_nonzero_displacement_ratio"], 2.0 / 5.0, places=6)

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

    def test_probability_bce_is_autocast_safe(self):
        prob = torch.full((1, 1, 2, 2, 2), 0.7, requires_grad=True)
        target = torch.ones_like(prob)
        weight = torch.ones_like(prob)
        with torch.autocast("cpu", enabled=True):
            loss = weighted_bce_prob(prob, target, weight)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prob.grad)

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

    def test_geometry_effective_gate_requires_disagreement_boundary_and_validity(self):
        config = DHGAConfig(dhga_geometry_boundary_band_mm=6.0, dhga_geometry_min_gate=0.0)
        w_geo = torch.ones(1, 1, 1, 1, 4)
        disagreement = torch.tensor([[[[[1.0, 0.0, 1.0, 1.0]]]]])
        sdf = torch.tensor([[[[[0.0, 0.0, 7.0, 0.0]]]]])
        valid = torch.tensor([[[[[1.0, 1.0, 1.0, 0.0]]]]])
        displacement = torch.ones_like(w_geo)
        gate = geometry_effective_gate(w_geo, disagreement, sdf, config, valid, displacement)
        self.assertEqual(gate.flatten().tolist(), [1.0, 0.0, 0.0, 0.0])

    def test_finalize_clamps_displacement_and_preserves_far_regions(self):
        config = DHGAConfig(dhga_geometry_enabled=True, dhga_geometry_max_displacement_mm=3.0, dhga_geometry_boundary_band_mm=6.0)
        region = torch.full((1, 1, 1, 1, 2), 0.5)
        sdf = torch.tensor([[[[[0.0, 10.0]]]]])
        disp = torch.full_like(region, 30.0)
        final = finalize_probability(region, sdf, disp, torch.ones_like(region), config, expert_disagreement=torch.ones_like(region))
        expected_active = torch.sigmoid(torch.tensor(3.0))
        self.assertAlmostEqual(float(final[..., 0]), float(expected_active), places=5)
        self.assertAlmostEqual(float(final[..., 1]), 0.5, places=5)

    def test_surface_and_geometry_case_metrics_report_before_after(self):
        gt = torch.zeros(5, 5, 5, dtype=torch.bool).numpy()
        gt[1:4, 1:4, 1:4] = True
        fused = gt.copy()
        fused[1, 1, 1] = False
        final = gt.copy()
        final[0, 0, 0] = True
        surface = surface_distance_metrics(final, gt, (1.0, 1.0, 1.0), 1.0)
        self.assertGreater(surface["surface_dice"], 0.0)
        metrics = compute_geometry_case_metrics(fused, final, gt, (1.0, 1.0, 1.0), 1.0)
        self.assertGreater(metrics["geometry_after_dice"], metrics["fused_before_geometry_dice"])
        self.assertEqual(metrics["geometry_tp_delta"], 1)
        self.assertEqual(metrics["geometry_fn_delta"], -1)
        self.assertEqual(metrics["geometry_tp_gained_voxels"], 1)
        self.assertEqual(metrics["geometry_tp_lost_voxels"], 0)
        self.assertEqual(metrics["geometry_fn_recovered_voxels"], 1)
        self.assertEqual(metrics["geometry_fp_added_voxels"], 1)

    def test_geometry_dense_scatter_does_not_apply_disagreement_weight_twice(self):
        source = inspect.getsource(DHGAVoxTellModel.run_geometry)
        self.assertIn("sparse_displacement,", source)
        self.assertNotIn("sparse_displacement * self._sample_point_weight", source)

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

    def test_stage_b_to_c_init_keeps_fresh_geometry_but_resume_restores_it(self):
        class TinyStageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.semantic_expert = torch.nn.Linear(1, 1)
                self.appearance_expert = torch.nn.Linear(1, 1)
                self.router = torch.nn.Linear(1, 1)
                self.geometry_head = torch.nn.Linear(1, 1)
                self.ray_tokens = torch.nn.Linear(1, 1)
                self.geometry_visual_proj = torch.nn.Conv3d(1, 1, 1)

        def fill_module(module: torch.nn.Module, value: float) -> None:
            with torch.no_grad():
                for param in module.parameters():
                    param.fill_(value)

        tmp = Path(".save") / "unit_checkpoint_tests"
        tmp.mkdir(parents=True, exist_ok=True)
        stage_b_path = tmp / "stage_b_to_c_init.pt"
        stage_c_path = tmp / "stage_c_resume.pt"
        try:
            source_b = TinyStageModel()
            fill_module(source_b.semantic_expert, 2.0)
            fill_module(source_b.appearance_expert, 3.0)
            fill_module(source_b.router, 4.0)
            fill_module(source_b.geometry_head, 90.0)
            fill_module(source_b.ray_tokens, 91.0)
            fill_module(source_b.geometry_visual_proj, 92.0)
            save_training_checkpoint(stage_b_path, source_b, DHGAConfig(dhga_stage="B"), metadata={"stage": "B"})

            stage_c_init = TinyStageModel()
            geometry_before = {
                name: tensor.detach().clone()
                for name, tensor in stage_c_init.state_dict().items()
                if name.startswith(STAGE_C_FRESH_GEOMETRY_PREFIXES)
            }
            init_payload = load_training_checkpoint(
                stage_b_path,
                stage_c_init,
                load_training_state=False,
                skip_model_prefixes=STAGE_C_FRESH_GEOMETRY_PREFIXES,
            )
            self.assertIn("geometry_head.weight", init_payload["skipped_model_keys"])
            self.assertTrue(torch.allclose(stage_c_init.semantic_expert.weight, torch.full_like(stage_c_init.semantic_expert.weight, 2.0)))
            self.assertTrue(torch.allclose(stage_c_init.appearance_expert.weight, torch.full_like(stage_c_init.appearance_expert.weight, 3.0)))
            self.assertTrue(torch.allclose(stage_c_init.router.weight, torch.full_like(stage_c_init.router.weight, 4.0)))
            for name, tensor in stage_c_init.state_dict().items():
                if name.startswith(STAGE_C_FRESH_GEOMETRY_PREFIXES):
                    self.assertTrue(torch.allclose(tensor, geometry_before[name]), msg=f"{name} should keep source-code initialization")

            source_c = TinyStageModel()
            fill_module(source_c.geometry_head, 7.0)
            fill_module(source_c.ray_tokens, 8.0)
            fill_module(source_c.geometry_visual_proj, 9.0)
            optim_c = torch.optim.AdamW(source_c.parameters(), lr=1e-3)
            loss = sum(param.sum() for param in source_c.parameters())
            loss.backward()
            optim_c.step()
            saved_geometry = {
                name: tensor.detach().clone()
                for name, tensor in source_c.state_dict().items()
                if name.startswith(STAGE_C_FRESH_GEOMETRY_PREFIXES)
            }
            save_training_checkpoint(stage_c_path, source_c, DHGAConfig(dhga_stage="C"), optimizer=optim_c, metadata={"stage": "C"})

            resumed_c = TinyStageModel()
            resumed_optim = torch.optim.AdamW(resumed_c.parameters(), lr=5e-4)
            resume_payload = load_training_checkpoint(stage_c_path, resumed_c, resumed_optim, expected_stage="C")
            self.assertEqual(resume_payload["skipped_model_keys"], [])
            for name, tensor in resumed_c.state_dict().items():
                if name.startswith(STAGE_C_FRESH_GEOMETRY_PREFIXES):
                    self.assertTrue(torch.allclose(tensor, saved_geometry[name]), msg=f"{name} should restore on Stage C resume")
            self.assertGreater(len(resumed_optim.state_dict()["state"]), 0)
        finally:
            stage_b_path.unlink(missing_ok=True)
            stage_c_path.unlink(missing_ok=True)

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

    def test_stage_b_router_target_loss_backpropagates_when_cross_weight_is_zero(self):
        router = DisagreementRouter("none")
        sem = torch.rand(1, 1, 4, 4, 4)
        app = 1.0 - sem
        student = SimpleNamespace(router=router(sem, app))
        teacher = SimpleNamespace(
            semantic_prob=sem.detach(),
            appearance_prob=app.detach(),
            anchor_prob=sem.detach(),
            router=router(sem.detach(), app.detach()),
        )
        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        loss = trainer._router_target_loss(student, teacher, min_weight=0.05)
        loss.backward()
        grads = [param.grad for param in router.parameters() if param.requires_grad]
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertTrue(any(grad is not None and bool((grad.abs() > 0).any()) for grad in grads))

    def test_router_target_weight_default_is_configurable(self):
        self.assertAlmostEqual(DHGAConfig().dhga_router_target_weight, 0.5)
        self.assertAlmostEqual(DHGAConfig(dhga_router_target_weight=0.25).dhga_router_target_weight, 0.25)
        with self.assertRaises(ValueError):
            DHGAConfig(dhga_router_target_weight=-0.1)

    def test_router_supervision_weight_prioritizes_foreground_and_disagreement(self):
        router = DisagreementRouter("none")
        sem = torch.zeros(1, 1, 1, 1, 4)
        app = torch.zeros_like(sem)
        anchor = torch.zeros_like(sem)
        sem[..., 1] = 0.8
        app[..., 1] = 0.8
        anchor[..., 1] = 0.8
        sem[..., 2] = 0.9
        app[..., 2] = 0.1
        anchor[..., 2] = 0.5
        teacher = SimpleNamespace(
            semantic_prob=sem,
            appearance_prob=app,
            anchor_prob=anchor,
            router=router(sem, app),
        )
        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        weight = trainer._router_supervision_weight(teacher, min_weight=0.0)
        self.assertGreater(float(weight[..., 1]), float(weight[..., 0]))
        self.assertGreater(float(weight[..., 2]), float(weight[..., 0]))

    def test_router_reliability_target_emphasizes_high_disagreement_geometry(self):
        router = DisagreementRouter("none")
        sem = torch.tensor([[[[[0.5, 0.9]]]]])
        app = torch.tensor([[[[[0.5, 0.1]]]]])
        anchor = torch.tensor([[[[[0.5, 0.5]]]]])
        teacher = SimpleNamespace(
            semantic_prob=sem,
            appearance_prob=app,
            anchor_prob=anchor,
            router=router(sem, app),
        )
        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        target = trainer._router_reliability_target(teacher)
        geo_target = target[:, 2:3]
        self.assertGreater(float(geo_target[..., 1]), float(geo_target[..., 0]))

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

    def test_stage_b_anchor_guided_patch_selection(self):
        class FakeAnchor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def baseline_forward(self, patch, spacing):
                self.calls += 1
                prob = patch[:, :1].clamp(0, 1)
                return SimpleNamespace(anchor_prob=prob)

        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.config = DHGAConfig(steps_per_volume=5, dhga_stage_b_anchor_candidate_patches=5, dhga_stage_b_include_background_patch=True)
        trainer.model = FakeAnchor()
        trainer.device = torch.device("cpu")
        trainer.stage_b_anchor_cache = {}
        volume = torch.zeros(1, 5, 2, 2)
        volume[:, 0] = 0.95
        volume[:, 1] = 0.90
        volume[:, 2] = 0.50
        volume[:, 3] = 0.05
        volume[:, 4] = 0.0
        slicers = [
            (slice(None), slice(0, 1), slice(None), slice(None)),
            (slice(None), slice(1, 2), slice(None), slice(None)),
            (slice(None), slice(2, 3), slice(None), slice(None)),
            (slice(None), slice(3, 4), slice(None), slice(None)),
            (slice(None), slice(4, 5), slice(None), slice(None)),
        ]
        random.seed(7)
        selected = trainer._stage_b_anchor_guided_slicers("case_a", volume, (1.0, 1.0, 1.0), slicers)
        self.assertEqual([kind for _, kind in selected].count("foreground"), 3)
        self.assertEqual([kind for _, kind in selected].count("boundary"), 1)
        self.assertEqual([kind for _, kind in selected].count("background"), 1)
        self.assertEqual(trainer.model.calls, 5)
        self.assertEqual(len(trainer.stage_b_anchor_cache["case_a"]), 5)
        random.seed(11)
        cached = trainer._stage_b_anchor_guided_slicers("case_a", volume * 0, (1.0, 1.0, 1.0), list(reversed(slicers)))
        self.assertEqual([kind for _, kind in cached].count("foreground"), 3)
        self.assertEqual([kind for _, kind in cached].count("boundary"), 1)
        self.assertEqual([kind for _, kind in cached].count("background"), 1)
        self.assertEqual(trainer.model.calls, 5)

    def test_stage_b_patch_kind_counts_follow_ratio_and_shortfall_priority(self):
        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.config = DHGAConfig(steps_per_volume=4, dhga_stage_b_include_background_patch=False)
        self.assertEqual(trainer._stage_b_patch_kind_counts(4), {"foreground": 3, "boundary": 1, "background": 0})
        self.assertEqual(trainer._stage_b_patch_kind_counts(3), {"foreground": 3, "boundary": 0, "background": 0})
        trainer.config = DHGAConfig(steps_per_volume=5, dhga_stage_b_include_background_patch=True)
        self.assertEqual(trainer._stage_b_patch_kind_counts(5), {"foreground": 3, "boundary": 1, "background": 1})
        self.assertEqual(trainer._stage_b_patch_kind_counts(4), {"foreground": 2, "boundary": 1, "background": 1})
        self.assertEqual(trainer._stage_b_patch_kind_counts(2), {"foreground": 1, "boundary": 1, "background": 0})

    def test_best_checkpoint_replaces_epoch_checkpoint_saves(self):
        class FakeWriter:
            def add_scalar(self, *args, **kwargs):
                pass

        class FakeEvaluator:
            def __init__(self, *args, **kwargs):
                pass

            def evaluate_split(self, split, max_cases):
                return {"mean_fused_dice": 0.7, "rows": []}

        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.config = DHGAConfig(dhga_stage="B", val_label_dir="/tmp")
        trainer.save_dir = Path(tempfile.mkdtemp())
        trainer.model = DHGASmokeModel(trainer.config)
        trainer.optimizer = None
        trainer.teacher = None
        trainer.scaler = None
        trainer.global_step = 3
        trainer.best_validation_score = None
        trainer.best_epoch = None
        trainer.writer = FakeWriter()
        trainer.prompts = ["liver"]
        trainer.predictor = SimpleNamespace()

        import dhga.evaluation as evaluation_module

        original = evaluation_module.DHGAEvaluator
        evaluation_module.DHGAEvaluator = FakeEvaluator
        try:
            trainer._run_periodic_test_evaluation(2, {"epoch_loss_mean": 1.0})
        finally:
            evaluation_module.DHGAEvaluator = original
        self.assertTrue((trainer.save_dir / "best_stage_b.pt").exists())
        self.assertTrue((trainer.save_dir / "last_stage_b.pt").exists())
        payload = load_training_checkpoint(trainer.save_dir / "best_stage_b.pt", trainer.model, load_training_state=False)
        self.assertEqual(payload["metadata"]["best_metric"], "mean_fused_dice")
        self.assertEqual(payload["metadata"]["best_epoch"], 2)
        self.assertFalse((trainer.save_dir / "checkpoint_epoch_0002.pt").exists())

    def test_epoch_metrics_include_distribution_summary(self):
        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.writer = None
        metrics = trainer._log_epoch_metrics(1, [{"loss": 1.0}, {"loss": 2.0}, {"loss": 10.0}])
        self.assertEqual(metrics["epoch_loss_mean"], 13.0 / 3.0)
        self.assertEqual(metrics["epoch_loss_median"], 2.0)
        self.assertEqual(metrics["epoch_loss_p90"], 2.0)
        self.assertEqual(metrics["epoch_loss_max"], 10.0)

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

    def test_stage_b_trainability_validation_requires_lora_appearance_and_router(self):
        class FakeStageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.injected_lora = torch.nn.ModuleDict()
                self.appearance_expert = torch.nn.Linear(1, 1)
                self.router = torch.nn.Conv3d(1, 1, 1)
                self.geometry_head = torch.nn.Linear(1, 1)
                self.ray_tokens = torch.nn.Linear(1, 1)
                self.geometry_visual_proj = torch.nn.Conv3d(1, 1, 1)

        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.config = DHGAConfig(dhga_stage="B")
        trainer.model = FakeStageModel()
        trainer._set_stage_trainability()
        with self.assertRaises(RuntimeError):
            trainer._validate_stage_trainability()

    def test_gradient_group_metrics_report_nonzero_stage_b_groups(self):
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

        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.config = DHGAConfig(dhga_stage="B")
        trainer.model = FakeStageModel()
        trainer._set_stage_trainability()
        trainer._validate_stage_trainability()
        for _, param in trainer.model.named_parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param)
        metrics = trainer._gradient_group_metrics()
        self.assertGreater(metrics["dhga_grad_semantic_lora_norm"], 0.0)
        self.assertGreater(metrics["dhga_grad_appearance_expert_norm"], 0.0)
        self.assertGreater(metrics["dhga_grad_router_norm"], 0.0)
        self.assertEqual(metrics["dhga_grad_semantic_lora_trainable_tensors"], 2.0)
        self.assertEqual(metrics["dhga_grad_router_trainable_tensors"], 2.0)

    def test_stage_b_gradient_validation_catches_disconnected_groups(self):
        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.config = DHGAConfig(dhga_stage="B")
        metrics = {
            "dhga_grad_semantic_lora_tensors": 0.0,
            "dhga_grad_semantic_lora_trainable_tensors": 48.0,
            "dhga_grad_appearance_expert_tensors": 6.0,
            "dhga_grad_router_tensors": 0.0,
            "dhga_grad_router_trainable_tensors": 4.0,
        }
        with self.assertRaises(RuntimeError):
            trainer._validate_required_gradient_flow(metrics)

    def test_appearance_disabled_does_not_change_requires_grad(self):
        class TinyDHGA(DHGAVoxTellModel):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.appearance_expert = torch.nn.Linear(1, 1)

        model = TinyDHGA()
        before = [param.requires_grad for param in model.appearance_expert.parameters()]
        with model.appearance_disabled():
            inside = [param.requires_grad for param in model.appearance_expert.parameters()]
        after = [param.requires_grad for param in model.appearance_expert.parameters()]
        self.assertEqual(inside, before)
        self.assertEqual(after, before)

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
            "fused_dice",
            "semantic_dice",
            "semantic_iou",
            "semantic_precision",
            "semantic_recall",
            "appearance_dice",
            "appearance_iou",
            "appearance_precision",
            "appearance_recall",
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

    def test_raw_disagreement_is_absolute_probability_difference(self):
        sem = torch.tensor([[[0.1, 0.9]]])
        app = torch.tensor([[[0.4, 0.2]]])
        raw = (sem.float() - app.float()).abs()
        self.assertTrue(torch.allclose(raw, torch.tensor([[[0.3, 0.7]]])))

    def test_raw_disagreement_metrics_cover_union_and_gt_boundary(self):
        import numpy as np

        raw = np.zeros((5, 5, 5), dtype=np.float32)
        raw[1:4, 1:4, 1:4] = 0.2
        raw[2, 2, 2] = 0.8
        sem = np.zeros_like(raw, dtype=bool)
        app = np.zeros_like(raw, dtype=bool)
        gt = np.zeros_like(raw, dtype=bool)
        sem[1:3, 1:3, 1:3] = True
        app[2:4, 2:4, 2:4] = True
        gt[1:4, 1:4, 1:4] = True
        metrics = compute_raw_disagreement_metrics(raw, sem, app, gt, spacing=(1.0, 1.0, 2.0), boundary_band_mm=1.5)
        for key in (
            "raw_disagreement_mean",
            "raw_disagreement_p50",
            "raw_disagreement_p75",
            "raw_disagreement_p90",
            "raw_disagreement_p95",
            "raw_disagreement_gt_0.1_rate",
            "raw_disagreement_gt_0.2_rate",
            "raw_disagreement_gt_0.3_rate",
            "raw_disagreement_union_mean",
            "raw_disagreement_union_p90",
            "raw_disagreement_union_voxel_rate",
            "raw_disagreement_gt_boundary_mean",
            "raw_disagreement_gt_boundary_p95",
            "raw_disagreement_gt_boundary_voxel_rate",
        ):
            self.assertIn(key, metrics)
        self.assertGreater(metrics["raw_disagreement_mean"], 0.0)
        self.assertGreater(metrics["raw_disagreement_union_mean"], metrics["raw_disagreement_mean"])
        self.assertGreater(metrics["raw_disagreement_gt_boundary_voxel_rate"], 0.0)
        self.assertLessEqual(metrics["raw_disagreement_gt_boundary_voxel_rate"], 1.0)

    def test_float_volume_writer_rejects_non_3d(self):
        import numpy as np

        with self.assertRaises(ValueError):
            write_float_volume_like_reader(np.zeros((1, 2, 3, 4), dtype=np.float32), "/tmp/unused.nii.gz", {})

    def test_text_layer_ensemble_aligns_native_decoder_layers(self):
        config = DHGAConfig(dhga_stage_b_method="text_layer_ensemble")
        low = torch.zeros(1, 1, 2, 3, 4)
        high = torch.zeros(1, 1, 4, 6, 8)
        summary = summarize_layers([low, high], config, target_shape=(4, 6, 8))
        self.assertEqual(tuple(summary["layer_probs"].shape), (2, 1, 1, 4, 6, 8))
        self.assertEqual(tuple(summary["p_mean"].shape), (1, 1, 4, 6, 8))

    def test_local_voxtell_decoder_contract_is_highest_resolution_first(self):
        voxtell_path = Path("/data/zy/VoxTell_from_disk/voxtell/model/voxtell_model.py")
        if not voxtell_path.exists():
            self.skipTest("local VoxTell source is not available")
        source = voxtell_path.read_text()
        self.assertIn("seg_outputs = seg_outputs[::-1]", source)
        self.assertIn("return seg_outputs[:1]", source)
        self.assertIn("returns predictions at all scales", source)
        self.assertIn("from highest to lowest resolution", source)

    def test_text_layer_candidate_rules_and_bounded_enhancement(self):
        config = DHGAConfig(
            dhga_stage_b_method="text_layer_ensemble",
            dhga_text_layer_foreground_support_threshold=0.5,
            dhga_text_layer_candidate_max_ratio=1.0,
            dhga_text_layer_candidate_alpha=0.5,
            dhga_text_layer_stability_threshold=0.05,
            dhga_text_layer_reliable_bg_threshold=0.3,
            dhga_text_layer_reliable_fg_threshold=0.7,
        )
        sem = {
            "p_mean": torch.tensor([[[[[0.50, 0.10]]]]]),
            "p_max": torch.tensor([[[[[0.95, 0.20]]]]]),
            "u_layer": torch.tensor([[[[[0.30, 0.02]]]]]),
        }
        app = {
            "p_mean": torch.tensor([[[[[0.45, 0.10]]]]]),
            "p_max": torch.tensor([[[[[0.90, 0.20]]]]]),
            "u_layer": torch.tensor([[[[[0.25, 0.02]]]]]),
        }
        fused = fuse_text_layer_ensemble(sem, app, config)
        self.assertTrue(bool(fused["candidate_fg"][0, 0, 0, 0, 0] > 0))
        self.assertFalse(bool(fused["candidate_fg"][0, 0, 0, 0, 1] > 0))
        self.assertLessEqual(float(fused["p_final"].max()), float(fused["p_max_all"].max()) + 1e-6)
        self.assertFalse(bool((fused["candidate_fg"].bool() & fused["reliable_bg"].bool()).any()))
        alpha_zero = DHGAConfig.from_mapping({**config.to_dict(), "dhga_text_layer_candidate_alpha": 0.0})
        no_enhance = fuse_text_layer_ensemble(sem, app, alpha_zero)
        self.assertTrue(torch.allclose(no_enhance["p_final"], no_enhance["p_base"]))

    def test_text_layer_all_background_not_candidate_and_loss_is_finite(self):
        config = DHGAConfig(dhga_stage_b_method="text_layer_ensemble")
        sem = {
            "p_mean": torch.full((1, 1, 1, 2, 2), 0.05),
            "p_max": torch.full((1, 1, 1, 2, 2), 0.10),
            "u_layer": torch.full((1, 1, 1, 2, 2), 0.50),
        }
        app = {
            "p_mean": torch.full((1, 1, 1, 2, 2), 0.05),
            "p_max": torch.full((1, 1, 1, 2, 2), 0.10),
            "u_layer": torch.full((1, 1, 1, 2, 2), 0.50),
        }
        fused = fuse_text_layer_ensemble(sem, app, config)
        self.assertFalse(bool(fused["candidate_fg"].bool().any()))
        dummy = SimpleNamespace(layer_ensemble={**fused, "semantic_p_mean": sem["p_mean"], "appearance_p_mean": app["p_mean"]})
        loss, metrics = text_layer_training_loss(dummy, config)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["dhga_text_layer_candidate_ratio"], 0.0)

    def test_voxtell_decode_can_return_native_multi_layer_logits_without_new_head(self):
        model = make_fake_text_layer_model()
        image = torch.randn(1, 1, 8, 8, 8)
        features = model.encode_once(image, (1.0, 1.0, 1.0))
        with model.decoder_deep_supervision_enabled():
            final, _prompt, layers = model.decode_from_features(features.encoder_stages, features.selected_feature, return_all_layers=True)
        self.assertEqual(len(layers), 2)
        self.assertEqual(tuple(final.shape[-3:]), (8, 8, 8))
        self.assertEqual(tuple(layers[0].shape[-3:]), (8, 8, 8))
        self.assertEqual(tuple(layers[1].shape[-3:]), (4, 4, 4))
        self.assertFalse(hasattr(model, "appearance_seg_head"))

    def test_text_layer_forward_uses_one_encoder_call_and_no_random_head(self):
        model = make_fake_text_layer_model()
        image = torch.randn(1, 1, 8, 8, 8)
        out = model.forward_text_layer_ensemble(image)
        self.assertEqual(model.encoder_calls, 1)
        self.assertIsNotNone(out.layer_ensemble)
        self.assertEqual(tuple(out.final_prob.shape[-3:]), (8, 8, 8))
        self.assertEqual(tuple(out.layer_ensemble["semantic_layer_probs"].shape), (2, 1, 1, 8, 8, 8))
        self.assertGreater(float(out.layer_ensemble["semantic_u_layer"].abs().mean()), 0.0)
        self.assertFalse(hasattr(model, "appearance_seg_head"))

    def test_legacy_decode_keeps_single_native_output(self):
        model = make_fake_text_layer_model(DHGAConfig(dhga_stage="B", dhga_stage_b_method="legacy", dhga_geometry_enabled=False, dhga_appearance_feature_layers=[0, 1]))
        image = torch.randn(1, 1, 8, 8, 8)
        features = model.encode_once(image, (1.0, 1.0, 1.0))
        final, _prompt = model.decode_from_features(features.encoder_stages, features.selected_feature)
        self.assertEqual(tuple(final.shape[-3:]), (8, 8, 8))
        self.assertFalse(model.network.decoder.deep_supervision)

    def test_text_layer_stage_b_gradients_and_frozen_router_geometry(self):
        config = DHGAConfig(
            dhga_stage="B",
            dhga_stage_b_method="text_layer_ensemble",
            dhga_geometry_enabled=False,
            dhga_appearance_feature_layers=[0, 1],
            dhga_text_layer_foreground_support_threshold=0.0,
            dhga_text_layer_candidate_max_ratio=1.0,
            dhga_text_layer_stability_threshold=1.0,
            dhga_text_layer_reliable_bg_threshold=0.49,
            dhga_text_layer_reliable_fg_threshold=0.51,
        )
        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.config = config
        trainer.device = torch.device("cpu")
        trainer.teacher = None
        trainer.global_step = 0
        trainer.model = make_fake_text_layer_model(config)
        trainer._set_stage_trainability()
        trainer._validate_stage_trainability()
        loss, metrics = trainer._stage_b_text_layer_ensemble(torch.randn(1, 1, 8, 8, 8), (1.0, 1.0, 1.0))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grad_metrics = trainer._gradient_group_metrics()
        self.assertGreater(grad_metrics["dhga_grad_semantic_lora_nonzero_tensors"], 0.0)
        self.assertGreater(grad_metrics["dhga_grad_appearance_expert_nonzero_tensors"], 0.0)
        self.assertEqual(grad_metrics["dhga_grad_router_trainable_tensors"], 0.0)
        self.assertEqual(grad_metrics["dhga_grad_geometry_trainable_tensors"], 0.0)
        self.assertEqual(metrics["dhga_text_layer_encoder_calls_per_forward"], 2.0)
        self.assertEqual(metrics["dhga_text_layer_teacher_student_forwards"], 2.0)
        self.assertIn("dhga_text_layer_teacher_candidate_fg_ratio", metrics)

    def test_text_layer_teacher_targets_drive_student_loss(self):
        config = DHGAConfig(dhga_stage_b_method="text_layer_ensemble")
        student_p = torch.full((1, 1, 1, 1, 2), 0.5, requires_grad=True)
        teacher_target = torch.tensor([[[[[1.0, 0.0]]]]])
        student = SimpleNamespace(layer_ensemble={
            "semantic_p_mean": student_p,
            "appearance_p_mean": student_p,
            "p_final": student_p,
        })
        teacher = {
            "reliable_fg": torch.tensor([[[[[1.0, 0.0]]]]]),
            "reliable_bg": torch.tensor([[[[[0.0, 1.0]]]]]),
            "candidate_fg": torch.zeros(1, 1, 1, 1, 2),
            "ignored": torch.zeros(1, 1, 1, 1, 2),
            "p_final": teacher_target,
        }
        loss, _ = text_layer_training_loss(student, config, teacher)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student_p.grad)

    def test_geometry_consumes_text_layer_candidate_and_final_probability(self):
        config = DHGAConfig(
            dhga_stage="C",
            dhga_stage_b_method="text_layer_ensemble",
            dhga_geometry_enabled=True,
            dhga_appearance_feature_layers=[0, 1],
            dhga_max_boundary_points=16,
            dhga_boundary_chunk_size=8,
        )
        model = make_fake_text_layer_model(config)
        image = torch.randn(1, 1, 8, 8, 8)
        out = model.forward_text_layer_ensemble(image, run_geometry=True)
        self.assertIn("candidate_score", out.geometry)
        self.assertIn("geometry_sampling_weight", out.geometry)
        self.assertTrue(torch.allclose(out.geometry["candidate_score"], out.layer_ensemble["candidate_score"].detach()))
        self.assertTrue(torch.all(out.geometry["geometry_sampling_weight"] >= out.layer_ensemble["candidate_score"].detach()))
        self.assertTrue(torch.allclose(out.router.fused_prob, out.layer_ensemble["p_final"]))

    def test_legacy_stage_b_trainability_remains_router_enabled(self):
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

        trainer = DHGAStageTrainer.__new__(DHGAStageTrainer)
        trainer.config = DHGAConfig(dhga_stage="B", dhga_stage_b_method="legacy")
        trainer.model = FakeStageModel()
        trainer._set_stage_trainability()
        trainable = {name for name, param in trainer.model.named_parameters() if param.requires_grad}
        self.assertIn("router.weight", trainable)
        self.assertIn("appearance_expert.weight", trainable)

    def test_text_layer_ensemble_masks_are_pairwise_disjoint(self):
        from dhga.text_layer_ensemble import fuse_text_layer_ensemble
        config = DHGAConfig(
            dhga_stage_b_method="text_layer_ensemble",
            dhga_text_layer_foreground_support_threshold=0.0,
            dhga_text_layer_disagreement_threshold=0.0,
            dhga_text_layer_candidate_max_ratio=1.0,
            dhga_text_layer_candidate_alpha=0.5,
            dhga_text_layer_stability_threshold=1.0,
            dhga_text_layer_reliable_bg_threshold=0.3,
            dhga_text_layer_reliable_fg_threshold=0.7,
        )
        torch.manual_seed(7)
        for shape in ((1, 1, 4, 4, 4), (1, 1, 8, 6, 10)):
            p_mean_sem = torch.rand(*shape)
            p_mean_app = torch.rand(*shape)
            p_max_sem = torch.rand(*shape).clamp_min(p_mean_sem)
            p_max_app = torch.rand(*shape).clamp_min(p_mean_app)
            u_sem = torch.rand(*shape) * 0.5
            u_app = torch.rand(*shape) * 0.5
            sem = {"p_mean": p_mean_sem, "p_max": p_max_sem, "u_layer": u_sem}
            app = {"p_mean": p_mean_app, "p_max": p_max_app, "u_layer": u_app}
            fused = fuse_text_layer_ensemble(sem, app, config)
            fg = fused["reliable_fg"] > 0.5
            bg = fused["reliable_bg"] > 0.5
            cands = fused["candidate_fg"] > 0.5
            self.assertFalse(bool((fg & bg).any()), msg="reliable_fg and reliable_bg must be disjoint")
            self.assertFalse(bool((fg & cands).any()), msg="reliable_fg and candidate_fg must be disjoint")
            self.assertFalse(bool((bg & cands).any()), msg="reliable_bg and candidate_fg must be disjoint")
            union = fg | bg | cands
            ignored = ~union
            self.assertTrue(bool((union | ignored).all()))

    def test_stage_c_text_layer_ensemble_produces_nonzero_perturbation(self):
        from dhga.geometry.boundary_corruption import make_local_boundary_corruption
        config = DHGAConfig(
            dhga_stage="C",
            dhga_stage_b_method="text_layer_ensemble",
            dhga_surface_tolerance_mm=2.0,
            dhga_corruption_max_offset_mm=4.0,
            dhga_corruption_modes=["inward", "outward"],
        )
        shape = (1, 1, 12, 12, 12)
        spacing = (1.0, 1.0, 1.0)
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[..., 4:8, 4:8, 4:8] = True
        teacher_sdf = mask_to_sdf(mask, spacing)
        boundary_band = (teacher_sdf.abs() <= config.dhga_surface_tolerance_mm * 3.0).float()
        self.assertGreater(float(boundary_band.sum()), 10.0, "boundary_band must cover boundary voxels")
        edge_ring = torch.zeros_like(boundary_band)
        edge_ring[..., 3:9, 3:9, 3:9] = 1.0
        edge_ring[..., 5:7, 5:7, 5:7] = 0.0
        candidate_fg = edge_ring
        stable_band = boundary_band * (candidate_fg > 0.5)
        self.assertGreater(float(stable_band.sum()), 10.0,
                           "stable_band must be non-empty where boundary_band intersects candidate_fg edge ring")
        torch.manual_seed(0)
        corrupted, recovery_target, choices = make_local_boundary_corruption(
            teacher_sdf,
            config.dhga_corruption_max_offset_mm,
            config.dhga_corruption_modes,
            stable_band=stable_band,
        )
        perturb = teacher_sdf - corrupted
        nonzero_perturb = perturb.abs() > 1e-5
        nonzero_recovery = recovery_target.abs() > 1e-5
        self.assertTrue(bool(nonzero_perturb.any()),
                        "perturb must be non-zero when stable_band is non-empty and zero-mode is disabled")
        self.assertTrue(bool(nonzero_recovery.any()),
                        "recovery_target must be non-zero when perturb is non-zero")
        self.assertTrue(torch.allclose(recovery_target[nonzero_perturb], -perturb[nonzero_perturb]),
                        "recovery_target must equal -perturb where perturb is non-zero")
        overlap = (stable_band > 0.5) & nonzero_perturb
        self.assertTrue(bool(overlap.any()),
                        "non-zero perturbation must overlap the stable_band region")

    def test_text_layer_geometry_gate_allows_displacement_to_change_final_prob(self):
        from dhga.inference import finalize_probability
        from dhga.text_layer_ensemble import build_text_layer_geometry_gate
        config = DHGAConfig(
            dhga_geometry_enabled=True,
            dhga_geometry_boundary_band_mm=6.0,
            dhga_geometry_max_displacement_mm=5.0,
            dhga_ray_step_mm=1.0,
            dhga_geometry_min_gate=0.0,
            pred_threshold=0.5,
        )
        shape = (1, 1, 9, 9, 9)
        spacing = (1.0, 1.0, 1.0)
        z = torch.arange(shape[-3]).view(1, 1, -1, 1, 1).float()
        z = z.expand(*shape)
        fused_prob = torch.sigmoid(-(z - 4.0) * 3.0)
        sdf = z - 4.0
        candidate_score = torch.zeros_like(fused_prob)
        candidate_score[..., 3:6, 3:6, 3:6] = 0.8
        norm_dis = torch.zeros_like(fused_prob)
        norm_dis[..., 3:6, 3:6, 3:6] = 0.7
        boundary_band = (sdf.abs() <= float(config.dhga_geometry_boundary_band_mm)).float()
        w_geo = build_text_layer_geometry_gate(candidate_score, norm_dis, sdf_boundary_band=boundary_band, config=config)
        self.assertGreater(float(w_geo.max()), 0.0, "w_geo must be non-zero when candidate, disagreement, and boundary overlap")
        self.assertLess(float(w_geo[0, 0, 0, 0, 0]), 1e-6, "w_geo must be zero outside the overlap")
        disp = torch.zeros_like(fused_prob)
        final_zero = finalize_probability(fused_prob, sdf, disp, w_geo, config)
        self.assertTrue(torch.allclose(final_zero, fused_prob), "zero displacement must not change final_prob")
        disp[..., 3:6, 3:6, 3:6] = 2.0
        dense_valid = torch.ones_like(fused_prob)
        expert_dis = torch.full_like(fused_prob, 0.6)
        final_with_disp = finalize_probability(
            fused_prob, sdf, disp, w_geo, config,
            dense_valid_weight=dense_valid, expert_disagreement=expert_dis,
        )
        delta = (final_with_disp - fused_prob).abs()
        self.assertGreater(float(delta.max()), 0.0,
                           "non-zero displacement with non-zero w_geo must change final_prob")
        voxel_active = (w_geo * expert_dis * boundary_band) > 0.01
        self.assertTrue(bool((delta[voxel_active] > 1e-4).any()),
                        "at active gate voxels, final_prob must differ from fused_prob")


    def test_synthetic_smoke(self):
        result = run_synthetic_smoke(DHGAConfig(), "cpu")
        self.assertEqual(result.shared_encoder_calls, 1)
        self.assertGreaterEqual(result.loss, 0.0)
        self.assertIn("dhga_expert_corr", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
