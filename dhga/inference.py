from __future__ import annotations

import torch
from torch import Tensor

from .config import DHGAConfig
from .geometry.sdf import mask_to_sdf


def finalize_probability(
    region_prob: Tensor,
    sdf: Tensor,
    dense_displacement_mm: Tensor | None,
    geometry_gate: Tensor,
    config: DHGAConfig,
    dense_valid_weight: Tensor | None = None,
) -> Tensor:
    if not config.dhga_geometry_enabled or dense_displacement_mm is None:
        return region_prob
    if torch.count_nonzero(dense_displacement_mm).item() == 0:
        return region_prob
    dense_valid = torch.ones_like(dense_displacement_mm) if dense_valid_weight is None else (dense_valid_weight > 0).to(dense_displacement_mm.dtype)
    nonzero_displacement = (dense_displacement_mm.abs() > 1e-6).to(dense_displacement_mm.dtype)
    gate = ((sdf.abs() <= config.dhga_search_radius_mm) * geometry_gate * dense_valid * nonzero_displacement).clamp(0, 1)
    if torch.count_nonzero(gate).item() == 0:
        return region_prob
    geo_prob = ((sdf - dense_displacement_mm) / max(config.dhga_ray_step_mm, 1e-6)).neg().sigmoid()
    return region_prob * (1.0 - gate) + geo_prob * gate


def finalize_mask(fused_prob: Tensor, displacement_mm: Tensor | None, config: DHGAConfig, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tensor:
    initial = fused_prob >= config.pred_threshold
    sdf = mask_to_sdf(initial, spacing)
    final_prob = finalize_probability(fused_prob, sdf, displacement_mm, torch.ones_like(fused_prob), config)
    return final_prob >= config.pred_threshold
