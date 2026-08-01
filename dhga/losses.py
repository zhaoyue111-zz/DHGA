from __future__ import annotations

import torch
from torch import Tensor

from dhga.routing.disagreement_router import RouterOutput


def bce_from_prob(student_prob: Tensor, target_prob: Tensor) -> Tensor:
    prob = student_prob.float().clamp(1e-6, 1.0 - 1e-6)
    target = target_prob.detach().float()
    return -(target * prob.log() + (1.0 - target) * (1.0 - prob).log())


def weighted_bce_prob(student_prob: Tensor, teacher_prob: Tensor, weight: Tensor) -> Tensor:
    loss = bce_from_prob(student_prob, teacher_prob)
    denom = weight.detach().sum().clamp_min(1.0)
    return (loss * weight.detach()).sum() / denom


def cross_supervision_loss(sem_prob: Tensor, app_prob: Tensor, router: RouterOutput) -> Tensor:
    weight = router.cross_supervision_weight
    sem_to_app = weighted_bce_prob(sem_prob, app_prob.detach(), weight)
    app_to_sem = weighted_bce_prob(app_prob, sem_prob.detach(), weight)
    return 0.5 * (sem_to_app + app_to_sem)


def router_fusion_loss(router: RouterOutput, target_prob: Tensor, weight: Tensor | None = None) -> Tensor:
    loss = bce_from_prob(router.fused_prob, target_prob)
    if weight is None:
        return loss.mean()
    return (loss * weight.detach()).sum() / weight.detach().sum().clamp_min(1.0)


def boundary_recovery_loss(predicted_displacement: Tensor, target_displacement: Tensor, valid_mask: Tensor) -> Tensor:
    loss = (predicted_displacement - target_displacement.detach()).abs()
    return (loss * valid_mask.detach()).sum() / valid_mask.detach().sum().clamp_min(1.0)


def _masked_mean_or_zero(values: Tensor, mask: Tensor) -> Tensor:
    count = mask.detach().float().sum()
    if bool((count > 0).detach().cpu()):
        return values[mask].mean()
    return values[mask].sum() * 0.0


def balanced_boundary_recovery_loss(
    predicted_displacement: Tensor,
    target_displacement: Tensor,
    valid_mask: Tensor,
    near_zero_threshold_mm: float,
    zero_weight: float = 0.25,
) -> tuple[Tensor, dict[str, Tensor]]:
    target = target_displacement.detach()
    valid = valid_mask.detach().bool()
    threshold = max(1e-4, float(near_zero_threshold_mm))
    point_loss = (predicted_displacement - target).abs()
    nonzero_mask = valid & (target.abs() > threshold)
    zero_mask = valid & (target.abs() <= threshold)
    move_loss = _masked_mean_or_zero(point_loss, nonzero_mask)
    stay_loss = _masked_mean_or_zero(point_loss, zero_mask)
    loss = move_loss + float(zero_weight) * stay_loss
    pred_nonzero = predicted_displacement.detach().abs() > threshold
    sign_match = torch.sign(predicted_displacement.detach()) == torch.sign(target)
    sign_correct = nonzero_mask & pred_nonzero & sign_match
    nonzero_count = nonzero_mask.float().sum()
    sign_agreement = sign_correct.float().sum() / nonzero_count.clamp_min(1.0)
    return loss, {
        "nonzero_loss": move_loss.detach(),
        "zero_loss": stay_loss.detach(),
        "nonzero_count": nonzero_count.detach(),
        "zero_count": zero_mask.float().sum().detach(),
        "sign_agreement": sign_agreement.detach(),
    }


def minimal_transport_loss(displacement: Tensor, weight: Tensor | None = None) -> Tensor:
    loss = displacement.abs()
    if weight is None:
        return loss.mean()
    return (loss * weight.detach()).sum() / weight.detach().sum().clamp_min(1.0)


def diagnostics_from_probs(sem_prob: Tensor, app_prob: Tensor, router: RouterOutput) -> dict[str, float]:
    sem = sem_prob.detach().float()
    app = app_prob.detach().float()
    sem_centered = sem - sem.mean()
    app_centered = app - app.mean()
    corr = (sem_centered * app_centered).mean() / (sem_centered.square().mean().sqrt() * app_centered.square().mean().sqrt()).clamp_min(1e-6)
    return {
        "dhga_sem_fg_prob": float(sem.mean().cpu()),
        "dhga_app_fg_prob": float(app.mean().cpu()),
        "dhga_expert_corr": float(corr.cpu()),
        "dhga_disagreement_ratio": float((router.disagreement > 0.5).float().mean().cpu()),
        "dhga_stable_fg_ratio": float((router.stable_foreground > 0.5).float().mean().cpu()),
        "dhga_stable_bg_ratio": float((router.stable_background > 0.5).float().mean().cpu()),
    }
