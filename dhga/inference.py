from __future__ import annotations

import torch
from torch import Tensor

from .config import DHGAConfig
from .geometry.sdf import mask_to_sdf


def geometry_effective_gate(
    w_geo: Tensor,
    expert_disagreement: Tensor,
    sdf: Tensor,
    config: DHGAConfig,
    dense_valid_weight: Tensor | None = None,
    dense_displacement_mm: Tensor | None = None,
) -> Tensor:
    valid = torch.ones_like(w_geo) if dense_valid_weight is None else (dense_valid_weight > 0).to(w_geo.dtype)
    if dense_displacement_mm is not None:
        valid = valid * (dense_displacement_mm.abs() > 1e-6).to(w_geo.dtype)
    boundary_mask = (sdf.abs() <= float(config.dhga_geometry_boundary_band_mm)).to(w_geo.dtype)
    gate = (w_geo * expert_disagreement * boundary_mask * valid).clamp(0, 1)
    if config.dhga_geometry_min_gate > 0:
        gate = torch.where(gate > float(config.dhga_geometry_min_gate), gate, torch.zeros_like(gate))
    return gate


def finalize_probability(
    region_prob: Tensor,
    sdf: Tensor,
    dense_displacement_mm: Tensor | None,
    geometry_gate: Tensor,
    config: DHGAConfig,
    dense_valid_weight: Tensor | None = None,
    expert_disagreement: Tensor | None = None,
) -> Tensor:
    if not config.dhga_geometry_enabled or dense_displacement_mm is None:
        return region_prob
    dense_displacement_mm = dense_displacement_mm.clamp(
        -float(config.dhga_geometry_max_displacement_mm),
        float(config.dhga_geometry_max_displacement_mm),
    )
    if torch.count_nonzero(dense_displacement_mm).item() == 0:
        return region_prob
    if expert_disagreement is None:
        expert_disagreement = torch.ones_like(geometry_gate)
    gate = geometry_effective_gate(
        geometry_gate,
        expert_disagreement,
        sdf,
        config,
        dense_valid_weight=dense_valid_weight,
        dense_displacement_mm=dense_displacement_mm,
    )
    if torch.count_nonzero(gate).item() == 0:
        return region_prob
    geo_prob = ((sdf - dense_displacement_mm) / max(config.dhga_ray_step_mm, 1e-6)).neg().sigmoid()
    return region_prob * (1.0 - gate) + geo_prob * gate


def finalize_mask(fused_prob: Tensor, displacement_mm: Tensor | None, config: DHGAConfig, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tensor:
    initial = fused_prob >= config.pred_threshold
    sdf = mask_to_sdf(initial, spacing)
    final_prob = finalize_probability(fused_prob, sdf, displacement_mm, torch.ones_like(fused_prob), config)
    return final_prob >= config.pred_threshold
