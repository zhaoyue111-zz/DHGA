from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from dhga.config import DHGAConfig


def align_layer_logits(layer_logits: list[Tensor], target_shape: tuple[int, int, int]) -> list[Tensor]:
    """Resize native VoxTell decoder layer logits to the final prediction grid."""
    if not layer_logits:
        raise ValueError("layer_logits must contain at least one tensor")
    aligned: list[Tensor] = []
    for logits in layer_logits:
        if tuple(logits.shape[-3:]) == tuple(target_shape):
            aligned.append(logits)
        else:
            aligned.append(F.interpolate(logits, size=target_shape, mode="trilinear", align_corners=False))
    return aligned


def highest_resolution_layer_index(layer_logits: list[Tensor]) -> int:
    if not layer_logits:
        raise ValueError("layer_logits must contain at least one tensor")
    volumes = [int(logits.shape[-3]) * int(logits.shape[-2]) * int(logits.shape[-1]) for logits in layer_logits]
    return max(range(len(volumes)), key=lambda idx: volumes[idx])


def order_layers_high_to_low(layer_logits: list[Tensor]) -> list[Tensor]:
    indexed = list(enumerate(layer_logits))
    indexed.sort(key=lambda item: int(item[1].shape[-3]) * int(item[1].shape[-2]) * int(item[1].shape[-1]), reverse=True)
    return [logits for _, logits in indexed]


def layer_weights_for_count(config: DHGAConfig, count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    configured = list(config.dhga_text_layer_weights)
    if configured:
        if len(configured) != count:
            raise ValueError(f"dhga_text_layer_weights length {len(configured)} does not match decoder layer count {count}")
        weights = torch.as_tensor(configured, device=device, dtype=dtype)
    else:
        weights = torch.ones(count, device=device, dtype=dtype)
    return weights / weights.sum().clamp_min(torch.finfo(dtype).eps)


def summarize_layer_probs(layer_probs: list[Tensor], config: DHGAConfig, target_shape: tuple[int, int, int] | None = None) -> dict[str, Tensor]:
    layer_probs = order_layers_high_to_low(layer_probs)
    target = target_shape or tuple(layer_probs[0].shape[-3:])
    aligned = []
    for prob in layer_probs:
        if tuple(prob.shape[-3:]) == tuple(target):
            aligned.append(prob.float())
        else:
            aligned.append(F.interpolate(prob.float(), size=target, mode="trilinear", align_corners=False))
    probs = torch.stack(aligned, dim=0)
    weights = layer_weights_for_count(config, probs.shape[0], probs.device, probs.dtype).view(-1, 1, 1, 1, 1, 1)
    p_mean = (probs * weights).sum(dim=0)
    return {
        "layer_probs": probs,
        "p_mean": p_mean,
        "p_last": probs[0],
        "p_max": probs.max(dim=0).values,
        "u_layer": (probs - p_mean.unsqueeze(0)).abs().mean(dim=0),
        "layer_weights": weights.flatten(),
    }


def summarize_layers(layer_logits: list[Tensor], config: DHGAConfig, target_shape: tuple[int, int, int] | None = None) -> dict[str, Tensor]:
    layer_logits = order_layers_high_to_low(layer_logits)
    target = target_shape or tuple(layer_logits[0].shape[-3:])
    aligned = align_layer_logits(layer_logits, target)
    return summarize_layer_probs([logits.float().sigmoid() for logits in aligned], config, target)


def _minmax_normalize_per_case(x: Tensor) -> Tensor:
    flat = x.flatten(2)
    lo = flat.amin(dim=-1, keepdim=True)
    hi = flat.amax(dim=-1, keepdim=True)
    return ((flat - lo) / (hi - lo).clamp_min(1e-6)).view_as(x).clamp(0, 1)


def _cap_candidate_ratio(score: Tensor, max_ratio: float) -> Tensor:
    max_ratio = float(max_ratio)
    if max_ratio <= 0.0:
        return torch.zeros_like(score)
    if max_ratio >= 1.0:
        return score
    flat = score.flatten(2)
    keep = max(1, int(flat.shape[-1] * max_ratio))
    values, indices = flat.topk(keep, dim=-1)
    capped = torch.zeros_like(flat)
    capped.scatter_(-1, indices, values)
    return capped.view_as(score)


def build_directional_candidate_scores(semantic: dict[str, Tensor], appearance: dict[str, Tensor], config: DHGAConfig) -> dict[str, Tensor]:
    semantic_layers = semantic.get("layer_probs")
    appearance_layers = appearance.get("layer_probs")
    reference = semantic["p_last"]
    zeros = torch.zeros_like(reference)
    temperature = max(float(config.dhga_text_layer_temperature), 1e-6)
    w_sem = torch.exp(-semantic["u_layer"] / temperature)
    w_app = torch.exp(-appearance["u_layer"] / temperature)
    primary_prob = (w_sem * semantic["p_last"] + w_app * appearance["p_last"]) / (w_sem + w_app).clamp_min(1e-6)
    if not isinstance(semantic_layers, Tensor) or not isinstance(appearance_layers, Tensor):
        return {"directional_primary_prob": primary_prob, "directional_secondary_support": zeros, "directional_gap": zeros, "candidate_expand_score": zeros, "candidate_shrink_score": zeros, "candidate_expand_raw": zeros, "candidate_shrink_raw": zeros}
    secondary_parts = []
    if semantic_layers.shape[0] > 1:
        secondary_parts.append(semantic_layers[1:])
    if appearance_layers.shape[0] > 1:
        secondary_parts.append(appearance_layers[1:])
    if not secondary_parts:
        return {"directional_primary_prob": primary_prob, "directional_secondary_support": zeros, "directional_gap": zeros, "candidate_expand_score": zeros, "candidate_shrink_score": zeros, "candidate_expand_raw": zeros, "candidate_shrink_raw": zeros}
    secondary_layers = torch.cat(secondary_parts, dim=0)
    secondary_support = torch.topk(secondary_layers, k=min(2, secondary_layers.shape[0]), dim=0).values[-1]
    directional_gap = (primary_prob - secondary_support).abs()
    pred_threshold = float(config.pred_threshold)
    disagreement_threshold = float(config.dhga_text_layer_disagreement_threshold)
    primary_fg = primary_prob >= pred_threshold
    expand_raw = (~primary_fg) & (secondary_support >= pred_threshold) & (directional_gap >= disagreement_threshold)
    shrink_raw = primary_fg & (secondary_support < pred_threshold) & (directional_gap >= disagreement_threshold)
    expand_score = (secondary_support - primary_prob).clamp_min(0.0) * expand_raw.float()
    shrink_score = (primary_prob - secondary_support).clamp_min(0.0) * shrink_raw.float()
    return {
        "directional_primary_prob": primary_prob,
        "directional_secondary_support": secondary_support,
        "directional_gap": directional_gap,
        "candidate_expand_score": expand_score,
        "candidate_shrink_score": shrink_score,
        "candidate_expand_raw": expand_raw.float(),
        "candidate_shrink_raw": shrink_raw.float(),
    }


def fuse_text_layer_ensemble(semantic: dict[str, Tensor], appearance: dict[str, Tensor], config: DHGAConfig) -> dict[str, Tensor]:
    temperature = max(float(config.dhga_text_layer_temperature), 1e-6)
    u_sem = semantic["u_layer"]
    u_app = appearance["u_layer"]
    w_sem = torch.exp(-u_sem / temperature)
    w_app = torch.exp(-u_app / temperature)
    p_sem = semantic["p_mean"]
    p_app = appearance["p_mean"]
    p_base = (w_sem * p_sem + w_app * p_app) / (w_sem + w_app).clamp_min(1e-6)
    p_max_all = torch.maximum(semantic["p_max"], appearance["p_max"])
    layer_disagreement = torch.maximum(u_sem, u_app)
    expert_disagreement = (p_sem - p_app).abs()
    disagreement = torch.maximum(layer_disagreement, expert_disagreement)
    normalized_disagreement = _minmax_normalize_per_case(disagreement)
    foreground_support = p_max_all >= float(config.dhga_text_layer_foreground_support_threshold)
    semantic_layers = semantic.get("layer_probs")
    appearance_layers = appearance.get("layer_probs")
    if isinstance(semantic_layers, Tensor) and isinstance(appearance_layers, Tensor):
        all_layer_probs = torch.cat([semantic_layers, appearance_layers], dim=0)
        support_count = (all_layer_probs >= float(config.dhga_text_layer_foreground_support_threshold)).sum(dim=0)
        candidate_support = support_count >= 2
    else:
        support_count = foreground_support.float()
        candidate_support = foreground_support
    enough_disagreement = disagreement >= float(config.dhga_text_layer_disagreement_threshold)
    candidate_score = _cap_candidate_ratio(normalized_disagreement * foreground_support.float() * enough_disagreement.float() * candidate_support.float(), config.dhga_text_layer_candidate_max_ratio)
    candidate_fg = candidate_score > 0
    alpha = float(config.dhga_text_layer_candidate_alpha)
    p_final = p_base + alpha * candidate_score * (p_max_all - p_base).clamp_min(0)
    p_final = torch.minimum(p_final, p_max_all).clamp(0, 1)
    stable = disagreement <= float(config.dhga_text_layer_stability_threshold)
    reliable_fg = (p_base >= float(config.dhga_text_layer_reliable_fg_threshold)) & stable & (~candidate_fg)
    reliable_bg = (p_base <= float(config.dhga_text_layer_reliable_bg_threshold)) & stable & (~candidate_fg)
    ignored = ~(reliable_fg | reliable_bg | candidate_fg)
    directional = build_directional_candidate_scores(semantic, appearance, config)
    return {
        "w_sem": w_sem,
        "w_app": w_app,
        "p_base": p_base,
        "p_final": p_final,
        "p_max_all": p_max_all,
        "normalized_disagreement": normalized_disagreement,
        "absolute_disagreement": disagreement,
        "layer_disagreement": layer_disagreement,
        "expert_disagreement": expert_disagreement,
        "support_count": support_count.float(),
        "candidate_support": candidate_support.float(),
        "candidate_score": candidate_score,
        "candidate_fg": candidate_fg.float(),
        "reliable_fg": reliable_fg.float(),
        "reliable_bg": reliable_bg.float(),
        "ignored": ignored.float(),
        "stable": stable.float(),
        **directional,
    }


def balanced_masked_bce(prob: Tensor, fg_mask: Tensor, bg_mask: Tensor, fg_weight: float = 0.25) -> Tensor:
    prob = prob.float().clamp(1e-6, 1.0 - 1e-6)
    fg = fg_mask.to(device=prob.device, dtype=torch.bool)
    bg = bg_mask.to(device=prob.device, dtype=torch.bool)
    zero = prob.sum() * 0.0
    has_fg = bool(fg.any().detach().cpu())
    has_bg = bool(bg.any().detach().cpu())
    if not has_bg:
        return zero
    bg_loss = -torch.log1p(-prob[bg]).mean()
    if not has_fg:
        return bg_loss
    fg_loss = -torch.log(prob[fg]).mean()
    fg_weight = float(fg_weight)
    return fg_weight * fg_loss + (1.0 - fg_weight) * bg_loss


def text_layer_training_loss(out, config: DHGAConfig, teacher_ensemble: dict[str, Tensor] | None = None, anchor_prob: Tensor | None = None) -> tuple[Tensor, dict[str, float]]:
    if out.layer_ensemble is None:
        raise ValueError("text_layer_training_loss requires DHGAForwardOutput.layer_ensemble")
    ens = out.layer_ensemble
    target_ens = teacher_ensemble or ens
    fg = target_ens["reliable_fg"].detach() > 0.5
    bg = target_ens["reliable_bg"].detach() > 0.5
    candidate = target_ens["candidate_fg"].detach() > 0.5
    semantic_prob = ens["semantic_p_mean"]
    appearance_prob = ens["appearance_p_mean"]
    sem_loss = balanced_masked_bce(semantic_prob, fg, bg)
    app_loss = balanced_masked_bce(appearance_prob, fg, bg)
    zero = semantic_prob.sum() * 0.0 + appearance_prob.sum() * 0.0
    semantic_anchor_loss = zero
    appearance_anchor_loss = zero
    volume_excess_loss = zero
    anchor_fg = torch.zeros_like(fg)
    anchor_bg = torch.zeros_like(bg)
    anchor_volume = semantic_prob.new_zeros(semantic_prob.shape[0])
    allowed_volume = semantic_prob.new_zeros(semantic_prob.shape[0])
    semantic_volume = semantic_prob.float().flatten(1).mean(dim=1)
    appearance_volume = appearance_prob.float().flatten(1).mean(dim=1)
    if anchor_prob is not None:
        anchor = anchor_prob.detach().float()
        if tuple(anchor.shape[-3:]) != tuple(semantic_prob.shape[-3:]):
            anchor = F.interpolate(anchor, size=semantic_prob.shape[-3:], mode="trilinear", align_corners=False)
        anchor = anchor.to(device=semantic_prob.device, dtype=semantic_prob.dtype).clamp(0, 1)
        anchor_fg = (anchor >= 0.9) & (~candidate)
        anchor_bg = (anchor <= 0.1) & (~candidate)
        semantic_anchor_loss = balanced_masked_bce(semantic_prob, anchor_fg, anchor_bg)
        appearance_anchor_loss = balanced_masked_bce(appearance_prob, anchor_fg, anchor_bg)
        anchor_volume = anchor.float().flatten(1).mean(dim=1)
        allowed_volume = (1.5 * anchor_volume + 0.005).clamp(max=1.0)
        volume_excess_loss = ((semantic_volume - allowed_volume).clamp_min(0.0) + (appearance_volume - allowed_volume).clamp_min(0.0)).mean()
    candidate_loss = zero
    anchor_loss = semantic_anchor_loss + float(config.dhga_appearance_anchor_weight) * appearance_anchor_loss
    loss = sem_loss + app_loss + float(config.dhga_anchor_weight) * anchor_loss + float(config.dhga_appearance_expansion_weight) * volume_excess_loss
    metrics = {
        "dhga_text_layer_semantic_loss": float(sem_loss.detach().cpu()),
        "dhga_text_layer_appearance_loss": float(app_loss.detach().cpu()),
        "dhga_text_layer_candidate_loss": float(candidate_loss.detach().cpu()),
        "dhga_text_layer_semantic_anchor_loss": float(semantic_anchor_loss.detach().cpu()),
        "dhga_text_layer_appearance_anchor_loss": float(appearance_anchor_loss.detach().cpu()),
        "dhga_text_layer_volume_excess_loss": float(volume_excess_loss.detach().cpu()),
        "dhga_text_layer_reliable_fg_ratio": _ratio(fg),
        "dhga_text_layer_reliable_bg_ratio": _ratio(bg),
        "dhga_text_layer_candidate_ratio": _ratio(candidate),
        "dhga_text_layer_anchor_fg_ratio": _ratio(anchor_fg),
        "dhga_text_layer_anchor_bg_ratio": _ratio(anchor_bg),
        "dhga_text_layer_anchor_volume": float(anchor_volume.mean().detach().cpu()),
        "dhga_text_layer_allowed_volume": float(allowed_volume.mean().detach().cpu()),
        "dhga_text_layer_semantic_soft_volume": float(semantic_volume.mean().detach().cpu()),
        "dhga_text_layer_appearance_soft_volume": float(appearance_volume.mean().detach().cpu()),
        "dhga_text_layer_candidate_supervision_active": 0.0,
        "dhga_text_layer_ignore_ratio": _ratio(target_ens["ignored"].detach() > 0.5) if "ignored" in target_ens else 0.0,
    }
    return loss, metrics


def build_text_layer_geometry_gate(candidate_score: Tensor, normalized_disagreement: Tensor, sdf_boundary_band: Tensor | None = None, fused_prob: Tensor | None = None, config: DHGAConfig | None = None, spacing: tuple[float, float, float] | None = None) -> Tensor:
    """Build explicit geometry gate for text_layer_ensemble."""
    if candidate_score.shape != normalized_disagreement.shape:
        raise ValueError("candidate_score and normalized_disagreement must match in shape")
    if sdf_boundary_band is None:
        if fused_prob is not None:
            coarse = (fused_prob - 0.5).abs()
            sdf_boundary_band = (coarse <= 0.4).to(dtype=candidate_score.dtype)
        else:
            sdf_boundary_band = torch.ones_like(candidate_score)
    else:
        if sdf_boundary_band.shape != candidate_score.shape:
            sdf_boundary_band = F.interpolate(sdf_boundary_band.float(), size=candidate_score.shape[-3:], mode="trilinear", align_corners=False)
        sdf_boundary_band = sdf_boundary_band.to(dtype=candidate_score.dtype)
    gate = (candidate_score * normalized_disagreement * sdf_boundary_band).clamp(0, 1)
    if config is not None and float(getattr(config, "dhga_geometry_min_gate", 0.0)) > 0:
        gate = torch.where(gate > float(config.dhga_geometry_min_gate), gate, torch.zeros_like(gate))
    return gate


def _ratio(mask: Tensor) -> float:
    return float(mask.float().mean().detach().cpu()) if mask.numel() else 0.0
