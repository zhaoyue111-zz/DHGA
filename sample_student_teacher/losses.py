from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def dice_loss(student_prob: Tensor, teacher_prob: Tensor, eps: float = 1e-6) -> Tensor:
    student = student_prob.float().flatten(1)
    teacher = teacher_prob.detach().float().flatten(1)
    intersection = (student * teacher).sum(dim=1)
    denom = student.sum(dim=1) + teacher.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(student_logits: Tensor, teacher_prob: Tensor) -> tuple[Tensor, dict[str, float]]:
    student_prob = student_logits.float().sigmoid()
    target = teacher_prob.detach().float().clamp(0.0, 1.0)
    bce = F.binary_cross_entropy(student_prob.clamp(1e-6, 1.0 - 1e-6), target)
    dice = dice_loss(student_prob, target)
    loss = bce + dice
    return loss, {
        "loss": float(loss.detach().cpu()),
        "bce_loss": float(bce.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
    }
