from pathlib import Path
import tempfile
import unittest

import torch

from dhga.checkpoint import load_dhga_checkpoint, save_dhga_checkpoint
from dhga.config import DHGAConfig
from dhga.experts import AppearanceExpert, SemanticExpert
from dhga.geometry.boundary_corruption import make_bidirectional_corruption
from dhga.geometry.ray_sampler import make_ray_offsets_mm, sample_along_normals
from dhga.geometry.boundary_points import extract_boundary_points, sparse_displacements_to_dense_narrowband
from dhga.geometry.sdf import mask_to_sdf, sdf_normals, update_sdf_with_displacement
from dhga.inference import finalize_mask
from dhga.losses import cross_supervision_loss
from dhga.routing import DisagreementRouter
from dhga.shared_voxtell import SharedEncoderOnce
from dhga.trainer import DHGASmokeModel, run_synthetic_smoke


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
        dense = sparse_displacements_to_dense_narrowband(points, torch.ones(1, 32), valid, (7, 7, 7))
        self.assertGreater(float(dense.abs().sum()), 0.0)

    def test_ray_sampler_coordinate_order_and_spacing(self):
        volume = torch.zeros(1, 1, 5, 6, 7)
        volume[..., 2, 3, 4] = 1.0
        points = torch.tensor([[[2.0, 3.0, 3.0]]])
        normals = torch.tensor([[[0.0, 0.0, 1.0]]])
        offsets = torch.tensor([0.0, 2.0])
        samples, valid = sample_along_normals(volume, points, normals, offsets, spacing=(1.0, 1.0, 2.0))
        self.assertTrue(bool(valid.all()))
        self.assertAlmostEqual(float(samples[0, 0, 1, 0]), 1.0, places=5)

    def test_bidirectional_corruption_recovery_sign(self):
        sdf = torch.zeros(8, 1, 3, 3, 3)
        corrupted, target, _ = make_bidirectional_corruption(sdf, 2.0, ["outward", "inward"])
        perturb = sdf - corrupted
        nonzero = perturb.abs() > 0
        self.assertTrue(bool(nonzero.any()))
        self.assertTrue(torch.allclose(target[nonzero], -perturb[nonzero]))

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

    def test_synthetic_smoke(self):
        result = run_synthetic_smoke(DHGAConfig(), "cpu")
        self.assertEqual(result.shared_encoder_calls, 1)
        self.assertGreaterEqual(result.loss, 0.0)
        self.assertIn("dhga_expert_corr", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
