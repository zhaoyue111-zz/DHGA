from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def make_ray_offsets_mm(search_radius_mm: float, step_mm: float, device: torch.device | None = None) -> Tensor:
    steps = int(round(float(search_radius_mm) / float(step_mm)))
    return torch.arange(-steps, steps + 1, device=device, dtype=torch.float32) * float(step_mm)


def sample_along_normals(volume: Tensor, points_zyx: Tensor, normals_zyx: Tensor, offsets_mm: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, Tensor]:
    """Sample [B,C,D,H,W] at physical offsets along normals.

    points_zyx is [B,N,3] in voxel index order z,y,x. grid_sample expects x,y,z
    normalized coordinates, so conversion is explicit here.
    """
    if volume.ndim != 5:
        raise ValueError("volume must have shape [B, C, D, H, W]")
    bsz, _, depth, height, width = volume.shape
    if points_zyx.shape[:2] != normals_zyx.shape[:2] or points_zyx.shape[-1] != 3:
        raise ValueError("points and normals must have shapes [B, N, 3]")
    spacing_t = torch.as_tensor(spacing, device=volume.device, dtype=volume.dtype)
    offsets = offsets_mm.to(device=volume.device, dtype=volume.dtype)
    sample_points = points_zyx.to(volume.dtype).unsqueeze(2) + normals_zyx.to(volume.dtype).unsqueeze(2) * (offsets.view(1, 1, -1, 1) / spacing_t.view(1, 1, 1, 3))
    z = sample_points[..., 0]
    y = sample_points[..., 1]
    x = sample_points[..., 2]
    valid = (z >= 0) & (z <= depth - 1) & (y >= 0) & (y <= height - 1) & (x >= 0) & (x <= width - 1)
    grid_x = 2.0 * x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * y / max(height - 1, 1) - 1.0
    grid_z = 2.0 * z / max(depth - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(bsz, -1, offsets.numel(), 1, 3)
    sampled = F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.squeeze(-1).permute(0, 2, 3, 1).contiguous()
    return sampled, valid
