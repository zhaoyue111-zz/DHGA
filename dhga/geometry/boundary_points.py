from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .sdf import boundary_band, sdf_normals


def extract_boundary_points(
    sdf: Tensor,
    radius_mm: float,
    max_points: int,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[Tensor, Tensor, Tensor]:
    """Return padded boundary point coordinates and normals in z,y,x order."""
    if sdf.ndim != 5 or sdf.shape[1] != 1:
        raise ValueError("sdf must have shape [B, 1, D, H, W]")
    band = boundary_band(sdf, radius_mm)
    normals_grid = sdf_normals(sdf, spacing).squeeze(1)
    batch_points = []
    batch_normals = []
    batch_valid = []
    for batch_idx in range(sdf.shape[0]):
        coords = band[batch_idx, 0].nonzero(as_tuple=False)
        if coords.numel() == 0:
            coords = torch.tensor([[s // 2 for s in sdf.shape[-3:]]], device=sdf.device, dtype=torch.long)
        if coords.shape[0] > max_points:
            order = torch.randperm(coords.shape[0], device=sdf.device)[:max_points]
            coords = coords[order]
        normals = normals_grid[batch_idx, coords[:, 0], coords[:, 1], coords[:, 2]]
        valid = torch.ones(coords.shape[0], device=sdf.device, dtype=torch.bool)
        pad = max_points - coords.shape[0]
        if pad > 0:
            coords = F.pad(coords, (0, 0, 0, pad))
            normals = F.pad(normals, (0, 0, 0, pad))
            valid = F.pad(valid, (0, pad), value=False)
        batch_points.append(coords.float())
        batch_normals.append(normals.float())
        batch_valid.append(valid)
    return torch.stack(batch_points), torch.stack(batch_normals), torch.stack(batch_valid)


def sparse_displacements_to_dense_narrowband(
    points_zyx: Tensor,
    displacements_mm: Tensor,
    valid_points: Tensor,
    spatial_shape: tuple[int, int, int],
    radius_voxels: int = 1,
) -> Tensor:
    """Scatter sparse point displacements into a dense narrow-band volume."""
    bsz, _, _ = points_zyx.shape
    dense = torch.zeros((bsz, 1, *spatial_shape), device=points_zyx.device, dtype=displacements_mm.dtype)
    counts = torch.zeros_like(dense)
    rounded = points_zyx.round().long()
    for dz in range(-radius_voxels, radius_voxels + 1):
        for dy in range(-radius_voxels, radius_voxels + 1):
            for dx in range(-radius_voxels, radius_voxels + 1):
                coords = rounded + torch.tensor([dz, dy, dx], device=rounded.device)
                in_bounds = (
                    valid_points
                    & (coords[..., 0] >= 0)
                    & (coords[..., 0] < spatial_shape[0])
                    & (coords[..., 1] >= 0)
                    & (coords[..., 1] < spatial_shape[1])
                    & (coords[..., 2] >= 0)
                    & (coords[..., 2] < spatial_shape[2])
                )
                for batch_idx in range(bsz):
                    c = coords[batch_idx, in_bounds[batch_idx]]
                    if c.numel() == 0:
                        continue
                    values = displacements_mm[batch_idx, in_bounds[batch_idx]]
                    dense[batch_idx, 0, c[:, 0], c[:, 1], c[:, 2]] += values
                    counts[batch_idx, 0, c[:, 0], c[:, 1], c[:, 2]] += 1
    return dense / counts.clamp_min(1.0)
