from __future__ import annotations

import torch
from torch import Tensor

from .sdf import update_sdf_with_displacement


def make_bidirectional_corruption(
    teacher_sdf: Tensor,
    max_offset_mm: float,
    modes: list[str] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create known SDF perturbations and recovery targets.

    Positive perturbation moves the boundary outward. The recovery target is the
    inverse displacement, so outward corruption needs negative recovery and
    inward corruption needs positive recovery.
    """
    modes = modes or ["inward", "outward"]
    if not modes:
        raise ValueError("at least one corruption mode is required")
    device = teacher_sdf.device
    choices = torch.randint(0, len(modes), teacher_sdf.shape[:2], device=device)
    perturb = torch.zeros_like(teacher_sdf)
    for idx, mode in enumerate(modes):
        sign = 1.0 if mode == "outward" else -1.0 if mode == "inward" else 0.0
        if sign == 0.0:
            continue
        amp = torch.empty((*teacher_sdf.shape[:2], 1, 1, 1), device=device, dtype=teacher_sdf.dtype).uniform_(0.25, 1.0)
        perturb = torch.where((choices == idx).view(*teacher_sdf.shape[:2], 1, 1, 1), sign * amp * float(max_offset_mm), perturb)
    corrupted = update_sdf_with_displacement(teacher_sdf, perturb)
    recovery_target = -perturb
    return corrupted, recovery_target, choices
