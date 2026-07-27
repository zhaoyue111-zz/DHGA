from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .sdf import update_sdf_with_displacement


def make_local_boundary_corruption(
    teacher_sdf: Tensor,
    max_offset_mm: float,
    modes: list[str] | None = None,
    stable_band: Tensor | None = None,
    lowres_shape: tuple[int, int, int] = (6, 6, 6),
) -> tuple[Tensor, Tensor, Tensor]:
    """Create smooth local perturbations with inward, outward, and zero regions.

    Positive perturbation moves the boundary outward; recovery target is the inverse.
    """
    modes = [mode.lower().replace("_", " ").replace("-", " ") for mode in (modes or ["inward", "outward", "zero"])]
    allow_outward = any(mode in {"outward", "smooth local offset", "local bulge", "attached false positive like protrusion", "protrusion", "bulge"} for mode in modes)
    allow_inward = any(mode in {"inward", "smooth local offset", "local indentation", "missing boundary like gap", "indentation", "gap"} for mode in modes)
    allow_zero = "zero" in modes or not (allow_outward and allow_inward)
    device = teacher_sdf.device
    dtype = teacher_sdf.dtype
    noise = torch.randn((*teacher_sdf.shape[:2], *lowres_shape), device=device, dtype=dtype)
    smooth = F.interpolate(noise, size=teacher_sdf.shape[-3:], mode="trilinear", align_corners=False)
    kernel = min(5, *teacher_sdf.shape[-3:])
    if kernel % 2 == 0:
        kernel -= 1
    kernel = max(kernel, 1)
    smooth = F.avg_pool3d(smooth, kernel_size=kernel, stride=1, padding=kernel // 2)
    smooth = smooth / smooth.abs().amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    outward = (smooth > 0.25).to(dtype) if allow_outward else torch.zeros_like(smooth)
    inward = (smooth < -0.25).to(dtype) if allow_inward else torch.zeros_like(smooth)
    signed = outward - inward
    if allow_zero:
        signed = torch.where(smooth.abs() <= 0.25, torch.zeros_like(signed), signed)
    amp = torch.empty((*teacher_sdf.shape[:2], 1, 1, 1), device=device, dtype=dtype).uniform_(0.25, 1.0)
    perturb = signed * amp * float(max_offset_mm)
    if stable_band is not None:
        perturb = perturb * stable_band.to(dtype)
    corrupted = update_sdf_with_displacement(teacher_sdf, perturb)
    recovery_target = -perturb
    choices = torch.stack([(perturb < 0).float().mean(dim=(2, 3, 4)), (perturb > 0).float().mean(dim=(2, 3, 4)), (perturb == 0).float().mean(dim=(2, 3, 4))], dim=-1)
    return corrupted, recovery_target, choices


def make_bidirectional_corruption(
    teacher_sdf: Tensor,
    max_offset_mm: float,
    modes: list[str] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    return make_local_boundary_corruption(teacher_sdf, max_offset_mm, modes)
