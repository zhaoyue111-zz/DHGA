from __future__ import annotations

import torch
from torch import Tensor

from .config import DHGAConfig
from .geometry.sdf import mask_to_sdf, update_sdf_with_displacement


def finalize_mask(fused_prob: Tensor, displacement_mm: Tensor | None, config: DHGAConfig, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tensor:
    initial = fused_prob >= config.pred_threshold
    if not config.dhga_geometry_enabled or displacement_mm is None:
        return initial
    if torch.count_nonzero(displacement_mm).item() == 0:
        return initial
    sdf = mask_to_sdf(initial, spacing)
    updated = update_sdf_with_displacement(sdf, displacement_mm)
    return updated < 0
