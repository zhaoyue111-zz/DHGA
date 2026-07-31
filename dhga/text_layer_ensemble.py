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


def fuse_text_layer_ensemble(
    semantic: dict[str, Tensor],
    appearance: dict[str, Tensor],
    config: DHGAConfig,
) -> dict[str, Tensor]:
    temperature = max(float(config.dhga_text_layer_temperature), 1e-6)
    u_sem = semantic["u_layer"]
    u_app = appearance["u_layer"]
    w_sem = torch.exp(-u_sem / temperature)
    w_app = torch.exp(-u_app / temperature)
    p_sem = semantic["p_mean"]
    p_app = appearance["p_mean"]
    p_base = (w_sem * p_sem + w_app * p_app) / (w_sem + w_app).clamp_min(1e-6)
    p_max_all = torch.maximum(semantic["p_max"], appearance["p_max"])

    disagreement = torch.maximum(u_sem, u_app)
    normalized_disagreement = _minmax_normalize_per_case(disagreement)
    foreground_support = (p_max_all >= float(config.dhga_text_layer_foreground_support_threshold)).float()
    enough_disagreement = (disagreement >= float(config.dhga_text_layer_disagreement_threshold)).float()
    candidate_score = _cap_candidate_ratio(
        normalized_disagreement * foreground_support * enough_disagreement,
        config.dhga_text_layer_candidate_max_ratio,
    )
    candidate_fg = candidate_score > 0

    alpha = float(config.dhga_text_layer_candidate_alpha)
    p_final = p_base + alpha * candidate_score * (p_max_all - p_base).clamp_min(0)
    p_final = torch.minimum(p_final, p_max_all).clamp(0, 1)

    stable = disagreement <= float(config.dhga_text_layer_stability_threshold)
    high_prob = p_base >= float(config.dhga_text_layer_reliable_fg_threshold)
    reliable_fg = high_prob & stable & (~candidate_fg)
    reliable_bg = (p_base <= float(config.dhga_text_layer_reliable_bg_threshold)) & stable & (~candidate_fg)
    ignored = ~(reliable_fg | reliable_bg | candidate_fg)
    return {
        "w_sem": w_sem,
        "w_app": w_app,
        "p_base": p_base,
        "p_final": p_final,
        "p_max_all": p_max_all,
        "normalized_disagreement": normalized_disagreement,
        "absolute_disagreement": disagreement,
        "candidate_score": candidate_score,
        "candidate_fg": candidate_fg.float(),
        "reliable_fg": reliable_fg.float(),
        "reliable_bg": reliable_bg.float(),
        "ignored": ignored.float(),
        "stable": stable.float(),
    }


def balanced_masked_bce(prob: Tensor, fg_mask: Tensor, bg_mask: Tensor) -> Tensor:
    prob = prob.float().clamp(1e-6, 1.0 - 1e-6)
    fg = fg_mask.to(device=prob.device, dtype=torch.bool)
    bg = bg_mask.to(device=prob.device, dtype=torch.bool)
    losses: list[Tensor] = []
    if bool(fg.any().detach().cpu()):
        losses.append(-torch.log(prob[fg]).mean())
    if bool(bg.any().detach().cpu()):
        losses.append(-torch.log1p(-prob[bg]).mean())
    if not losses:
        return prob.sum() * 0.0
    if len(losses) == 1:
        return losses[0]
    return 0.5 * (losses[0] + losses[1])


def text_layer_training_loss(out, config: DHGAConfig, teacher_ensemble: dict[str, Tensor] | None = None) -> tuple[Tensor, dict[str, float]]:
    if out.layer_ensemble is None:
        raise ValueError("text_layer_training_loss requires DHGAForwardOutput.layer_ensemble")
    ens = out.layer_ensemble
    target_ens = teacher_ensemble or ens
    fg = target_ens["reliable_fg"].detach() > 0.5
    bg = target_ens["reliable_bg"].detach() > 0.5
    candidate = target_ens["candidate_fg"].detach() > 0.5
    sem_loss = balanced_masked_bce(ens["semantic_p_mean"], fg, bg)
    app_loss = balanced_masked_bce(ens["appearance_p_mean"], fg, bg)
    target = target_ens["p_final"].detach()
    if bool(candidate.any().detach().cpu()):
        candidate_loss = 0.5 * (
            F.mse_loss(ens["semantic_p_mean"][candidate], target[candidate])
            + F.mse_loss(ens["appearance_p_mean"][candidate], target[candidate])
        )
    else:
        candidate_loss = target.sum() * 0.0
    loss = sem_loss + app_loss + float(config.dhga_text_layer_candidate_weight) * candidate_loss
    metrics = {
        "dhga_text_layer_semantic_loss": float(sem_loss.detach().cpu()),
        "dhga_text_layer_appearance_loss": float(app_loss.detach().cpu()),
        "dhga_text_layer_candidate_loss": float(candidate_loss.detach().cpu()),
        "dhga_text_layer_reliable_fg_ratio": _ratio(fg),
        "dhga_text_layer_reliable_bg_ratio": _ratio(bg),
        "dhga_text_layer_candidate_ratio": _ratio(candidate),
        "dhga_text_layer_ignore_ratio": _ratio(target_ens["ignored"].detach() > 0.5) if "ignored" in target_ens else 0.0,
    }
    return loss, metrics


def build_text_layer_geometry_gate(
    candidate_score: Tensor,
    normalized_disagreement: Tensor,
    sdf_boundary_band: Tensor | None = None,
    fused_prob: Tensor | None = None,
    config: DHGAConfig | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> Tensor:
    """Build explicit geometry gate for text_layer_ensemble.

    w_geo = candidate_score * normalized_disagreement * boundary_mask.

    If sdf_boundary_band is not provided but fused_prob is, a coarse boundary mask
    is derived from |fused_prob - 0.5| being small.
    """
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
            sdf_boundary_band = F.interpolate(
                sdf_boundary_band.float(),
                size=candidate_score.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        sdf_boundary_band = sdf_boundary_band.to(dtype=candidate_score.dtype)
    gate = (candidate_score * normalized_disagreement * sdf_boundary_band).clamp(0, 1)
    if config is not None and float(getattr(config, "dhga_geometry_min_gate", 0.0)) > 0:
        gate = torch.where(
            gate > float(config.dhga_geometry_min_gate),
            gate,
            torch.zeros_like(gate),
        )
    return gate


def _ratio(mask: Tensor) -> float:
    return float(mask.float().mean().detach().cpu()) if mask.numel() else 0.0
