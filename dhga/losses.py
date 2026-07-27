from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from dhga.routing.disagreement_router import RouterOutput


def weighted_bce_prob(student_prob: Tensor, teacher_prob: Tensor, weight: Tensor) -> Tensor:
    loss = F.binary_cross_entropy(student_prob.clamp(1e-6, 1 - 1e-6), teacher_prob.detach(), reduction="none")
    denom = weight.detach().sum().clamp_min(1.0)
    return (loss * weight.detach()).sum() / denom


def cross_supervision_loss(sem_prob: Tensor, app_prob: Tensor, router: RouterOutput) -> Tensor:
    consensus = (router.disagreement < 0.5).to(sem_prob.dtype)
    weight = (router.stable_foreground + router.stable_background) * (1.0 - router.disagreement) * consensus
    sem_to_app = weighted_bce_prob(sem_prob, app_prob.detach(), weight)
    app_to_sem = weighted_bce_prob(app_prob, sem_prob.detach(), weight)
    return 0.5 * (sem_to_app + app_to_sem)


def boundary_recovery_loss(predicted_displacement: Tensor, target_displacement: Tensor, valid_mask: Tensor) -> Tensor:
    loss = (predicted_displacement - target_displacement.detach()).abs()
    return (loss * valid_mask.detach()).sum() / valid_mask.detach().sum().clamp_min(1.0)


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
