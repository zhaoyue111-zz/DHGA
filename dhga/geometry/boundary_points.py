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
    sampling_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return padded boundary point coordinates and normals in z,y,x order."""
    if sdf.ndim != 5 or sdf.shape[1] != 1:
        raise ValueError("sdf must have shape [B, 1, D, H, W]")
    if torch.all(sdf < 0) or torch.all(sdf > 0):
        bsz = sdf.shape[0]
        points = torch.zeros((bsz, max_points, 3), device=sdf.device, dtype=torch.float32)
        normals = torch.zeros_like(points)
        valid = torch.zeros((bsz, max_points), device=sdf.device, dtype=torch.bool)
        return points, normals, valid
    band = boundary_band(sdf, radius_mm)
    normals_grid = sdf_normals(sdf, spacing).squeeze(1)
    batch_points = []
    batch_normals = []
    batch_valid = []
    for batch_idx in range(sdf.shape[0]):
        coords = band[batch_idx, 0].nonzero(as_tuple=False)
        if coords.numel() == 0:
            coords = torch.zeros((0, 3), device=sdf.device, dtype=torch.long)
        if coords.shape[0] > max_points:
            if sampling_weight is not None:
                weights = sampling_weight[batch_idx, 0, coords[:, 0], coords[:, 1], coords[:, 2]].float().clamp_min(0)
                if float(weights.sum()) > 0:
                    order = torch.multinomial(weights, max_points, replacement=False)
                else:
                    order = torch.randperm(coords.shape[0], device=sdf.device)[:max_points]
            else:
                order = torch.randperm(coords.shape[0], device=sdf.device)[:max_points]
            coords = coords[order]
        normals = normals_grid[batch_idx, coords[:, 0], coords[:, 1], coords[:, 2]] if coords.numel() else torch.zeros((0, 3), device=sdf.device)
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
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    diffusion_mm: float = 2.0,
) -> Tensor:
    """Scatter sparse point displacements into a dense narrow-band volume with a physical Gaussian kernel."""
    bsz, _, _ = points_zyx.shape
    dense = torch.zeros((bsz, 1, *spatial_shape), device=points_zyx.device, dtype=displacements_mm.dtype)
    counts = torch.zeros_like(dense)
    rounded = points_zyx.round().long()
    spacing_t = torch.tensor(spacing, device=points_zyx.device, dtype=displacements_mm.dtype)
    max_vox = [max(1, int(torch.ceil(torch.tensor(diffusion_mm / max(s, 1e-6))).item())) for s in spacing]
    sigma2 = float(diffusion_mm) ** 2
    for dz in range(-max_vox[0], max_vox[0] + 1):
        for dy in range(-max_vox[1], max_vox[1] + 1):
            for dx in range(-max_vox[2], max_vox[2] + 1):
                delta = torch.tensor([dz, dy, dx], device=rounded.device)
                dist2 = ((delta.to(displacements_mm.dtype) * spacing_t) ** 2).sum()
                if float(dist2) > (float(diffusion_mm) ** 2):
                    continue
                kernel_weight = torch.exp(-0.5 * dist2 / max(sigma2, 1e-6)).to(displacements_mm.dtype)
                coords = rounded + delta
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
                    dense[batch_idx, 0, c[:, 0], c[:, 1], c[:, 2]] += values * kernel_weight
                    counts[batch_idx, 0, c[:, 0], c[:, 1], c[:, 2]] += kernel_weight
    return dense / counts.clamp_min(1.0)
