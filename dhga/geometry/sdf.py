from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def mask_to_sdf(mask: Tensor, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tensor:
    """Return signed distance field with inside negative and outside positive."""
    from scipy import ndimage

    if mask.ndim != 5:
        raise ValueError("mask must have shape [B, C, D, H, W]")
    arrays = []
    mask_cpu = mask.detach().to("cpu").bool().numpy()
    sampling = tuple(float(v) for v in spacing)
    for batch in range(mask_cpu.shape[0]):
        channel_arrays = []
        for channel in range(mask_cpu.shape[1]):
            m = mask_cpu[batch, channel]
            outside = ndimage.distance_transform_edt(~m, sampling=sampling)
            inside = ndimage.distance_transform_edt(m, sampling=sampling)
            channel_arrays.append(outside - inside)
        arrays.append(channel_arrays)
    return torch.as_tensor(np.asarray(arrays), dtype=torch.float32, device=mask.device)


def sdf_normals(sdf: Tensor, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tensor:
    dz = _central_diff(sdf, dim=2) / float(spacing[0])
    dy = _central_diff(sdf, dim=3) / float(spacing[1])
    dx = _central_diff(sdf, dim=4) / float(spacing[2])
    grad = torch.stack([dz, dy, dx], dim=-1)
    return grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def update_sdf_with_displacement(sdf: Tensor, displacement_mm: Tensor) -> Tensor:
    """Positive displacement moves the surface outward: phi_new = phi - d."""
    return sdf - displacement_mm


def boundary_band(sdf: Tensor, radius_mm: float) -> Tensor:
    return sdf.abs() <= float(radius_mm)


def _central_diff(x: Tensor, dim: int) -> Tensor:
    padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate")
    shifted_dim = dim
    left = padded.narrow(shifted_dim, 0, x.shape[dim])
    right = padded.narrow(shifted_dim, 2, x.shape[dim])
    return (right - left) * 0.5
