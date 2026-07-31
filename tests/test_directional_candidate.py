from types import SimpleNamespace

import numpy as np
import torch

from dhga.config import DHGAConfig
from dhga.evaluation import compute_directional_candidate_metrics, gt_boundary_band
from dhga.text_layer_ensemble import build_directional_candidate_scores, fuse_text_layer_ensemble, text_layer_training_loss


def make_config() -> DHGAConfig:
    return DHGAConfig(dhga_text_layer_temperature=0.1, dhga_text_layer_disagreement_threshold=0.05, dhga_text_layer_candidate_alpha=0.0, dhga_text_layer_candidate_max_ratio=0.05, pred_threshold=0.5)


def make_summary(primary: float, secondary_a: float, secondary_b: float) -> dict[str, torch.Tensor]:
    p_last = torch.full((1, 1, 2, 2, 2), primary)
    layer_probs = torch.stack([p_last, torch.full_like(p_last, secondary_a), torch.full_like(p_last, secondary_b)], dim=0)
    p_mean = layer_probs.mean(dim=0)
    return {"layer_probs": layer_probs, "p_mean": p_mean, "p_last": p_last, "p_max": layer_probs.max(dim=0).values, "u_layer": (layer_probs - p_mean.unsqueeze(0)).abs().mean(dim=0)}


def test_directional_expansion_candidate():
    config = make_config()
    semantic = make_summary(0.40, 0.80, 0.75)
    appearance = make_summary(0.42, 0.78, 0.72)
    out = build_directional_candidate_scores(semantic, appearance, config)
    assert torch.all(out["candidate_expand_raw"] == 1)
    assert torch.all(out["candidate_expand_score"] > 0)
    assert torch.all(out["candidate_shrink_raw"] == 0)


def test_directional_shrink_candidate():
    config = make_config()
    semantic = make_summary(0.82, 0.20, 0.18)
    appearance = make_summary(0.78, 0.25, 0.22)
    out = build_directional_candidate_scores(semantic, appearance, config)
    assert torch.all(out["candidate_shrink_raw"] == 1)
    assert torch.all(out["candidate_shrink_score"] > 0)
    assert torch.all(out["candidate_expand_raw"] == 0)


def test_directional_consistent_layers_have_no_candidate():
    config = make_config()
    semantic = make_summary(0.80, 0.75, 0.72)
    appearance = make_summary(0.78, 0.74, 0.70)
    out = build_directional_candidate_scores(semantic, appearance, config)
    assert torch.count_nonzero(out["candidate_expand_score"]) == 0
    assert torch.count_nonzero(out["candidate_shrink_score"]) == 0


def test_directional_diagnostics_do_not_change_existing_fusion_or_loss():
    config = make_config()
    semantic = make_summary(0.65, 0.55, 0.45)
    appearance = make_summary(0.60, 0.50, 0.40)
    out = fuse_text_layer_ensemble(semantic, appearance, config)
    w_sem = torch.exp(-semantic["u_layer"] / config.dhga_text_layer_temperature)
    w_app = torch.exp(-appearance["u_layer"] / config.dhga_text_layer_temperature)
    expected_p_base = (w_sem * semantic["p_mean"] + w_app * appearance["p_mean"]) / (w_sem + w_app).clamp_min(1e-6)
    assert torch.allclose(out["p_base"], expected_p_base)
    assert torch.allclose(out["p_final"], expected_p_base)
    base_ensemble = {"semantic_p_mean": semantic["p_mean"], "appearance_p_mean": appearance["p_mean"], "reliable_fg": out["reliable_fg"], "reliable_bg": out["reliable_bg"], "candidate_fg": out["candidate_fg"], "ignored": out["ignored"]}
    extended_ensemble = {**base_ensemble, **{key: value for key, value in out.items() if key.startswith("directional_") or key.startswith("candidate_expand") or key.startswith("candidate_shrink")}}
    loss_a, _ = text_layer_training_loss(SimpleNamespace(layer_ensemble=base_ensemble), config)
    loss_b, _ = text_layer_training_loss(SimpleNamespace(layer_ensemble=extended_ensemble), config)
    assert torch.allclose(loss_a, loss_b)


def test_directional_candidate_metrics_reward_error_hits():
    primary = np.zeros((9, 9, 9), dtype=bool)
    primary[2:7, 2:7, 2:7] = True
    gt = primary.copy()
    gt[1, 4, 4] = True
    gt[4, 4, 4] = False
    expand_score = np.zeros_like(primary, dtype=np.float32)
    shrink_score = np.zeros_like(primary, dtype=np.float32)
    expand_score[1, 4, 4] = 1.0
    shrink_score[4, 4, 4] = 1.0
    metrics = compute_directional_candidate_metrics(primary, gt, expand_score, shrink_score, (1.0, 1.0, 1.0), 2.0)
    assert metrics["directional_expand_top05_hit_rate"] > 0
    assert metrics["directional_shrink_top05_hit_rate"] > 0
    assert metrics["directional_expand_top05_boundary_fn_coverage"] > 0
    assert metrics["directional_shrink_top05_boundary_fp_coverage"] > 0


def test_boundary_band_is_local_not_whole_volume():
    mask = np.zeros((15, 15, 15), dtype=bool)
    mask[5:10, 5:10, 5:10] = True
    band = gt_boundary_band(mask, (1.0, 1.0, 1.0), 1.0)
    assert band.any()
    assert not band.all()
    assert not band[0, 0, 0]
    assert band[5, 7, 7]
    assert not band[7, 7, 7]
