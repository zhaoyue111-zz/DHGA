from __future__ import annotations

import torch
from torch import Tensor


def binary_dice_miou(student_prob: Tensor, teacher_prob: Tensor, threshold: float = 0.5, eps: float = 1e-6) -> dict[str, float]:
    pred = student_prob.detach() >= float(threshold)
    target = teacher_prob.detach() >= float(threshold)
    dims = tuple(range(1, pred.ndim))
    tp = (pred & target).float().sum(dim=dims)
    fp = (pred & ~target).float().sum(dim=dims)
    fn = (~pred & target).float().sum(dim=dims)
    tn = (~pred & ~target).float().sum(dim=dims)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    fg_iou = (tp + eps) / (tp + fp + fn + eps)
    bg_iou = (tn + eps) / (tn + fp + fn + eps)
    miou = 0.5 * (fg_iou + bg_iou)
    return {
        "dice": float(dice.mean().cpu()),
        "miou": float(miou.mean().cpu()),
    }
