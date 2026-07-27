from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RouterOutput:
    stable_foreground: Tensor
    stable_background: Tensor
    disagreement: Tensor
    fused_prob: Tensor
    consensus_mask: Tensor
    cross_supervision_weight: Tensor
    geometry_disagreement_weight: Tensor
    w_sem: Tensor
    w_app: Tensor
    w_geo: Tensor


def _case_rank_normalize(x: Tensor) -> Tensor:
    flat = x.flatten(2)
    lo = flat.quantile(0.05, dim=-1, keepdim=True)
    hi = flat.quantile(0.95, dim=-1, keepdim=True)
    y = (flat - lo) / (hi - lo).clamp_min(1e-6)
    return y.clamp(0, 1).view_as(x)


class DisagreementRouter(nn.Module):
    """Continuous router that preserves high-disagreement voxels for geometry."""

    def __init__(
        self,
        normalization: str = "case_rank",
        foreground_quantile: float = 0.85,
        background_quantile: float = 0.15,
        disagreement_quantile: float = 0.75,
    ) -> None:
        super().__init__()
        self.normalization = normalization
        self.foreground_quantile = foreground_quantile
        self.background_quantile = background_quantile
        self.disagreement_quantile = disagreement_quantile
        self.spatial_head = nn.Sequential(
            nn.Conv3d(5, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(8, 3, kernel_size=1),
        )
        nn.init.zeros_(self.spatial_head[-1].weight)
        nn.init.zeros_(self.spatial_head[-1].bias)

    def forward(self, sem_prob: Tensor, app_prob: Tensor, sem_stability: Tensor | None = None, app_stability: Tensor | None = None) -> RouterOutput:
        if sem_prob.shape != app_prob.shape:
            raise ValueError(f"expert probability shapes differ: {sem_prob.shape} vs {app_prob.shape}")
        disagreement = (sem_prob - app_prob).abs()
        common_fg = torch.minimum(sem_prob, app_prob)
        common_bg = torch.minimum(1.0 - sem_prob, 1.0 - app_prob)
        if sem_stability is not None:
            common_fg = common_fg * sem_stability
            common_bg = common_bg * sem_stability
        if app_stability is not None:
            common_fg = common_fg * app_stability
            common_bg = common_bg * app_stability
        if self.normalization == "case_rank":
            stable_foreground = _case_rank_normalize(common_fg)
            stable_background = _case_rank_normalize(common_bg)
            disagreement_weight = _case_rank_normalize(disagreement)
        elif self.normalization == "none":
            stable_foreground = common_fg.clamp(0, 1)
            stable_background = common_bg.clamp(0, 1)
            disagreement_weight = disagreement.clamp(0, 1)
        else:
            raise ValueError("normalization must be case_rank or none")
        route_features = torch.cat([sem_prob, app_prob, disagreement_weight, stable_foreground, stable_background], dim=1)
        weights = self.spatial_head(route_features).softmax(dim=1)
        w_sem = weights[:, 0:1]
        w_app = weights[:, 1:2]
        w_geo = weights[:, 2:3]
        region_denom = (w_sem + w_app).clamp_min(1e-6)
        region_w_sem = w_sem / region_denom
        region_w_app = w_app / region_denom
        fused = region_w_sem * sem_prob + region_w_app * app_prob
        consensus_mask = (stable_foreground > stable_background) & (
            disagreement_weight < disagreement_weight.flatten(2).quantile(self.disagreement_quantile, dim=-1, keepdim=True).view(*disagreement_weight.shape[:2], 1, 1, 1)
        )
        low_disagreement_gate = (0.5 - disagreement_weight).clamp_min(0.0) * 2.0
        consensus_weight = (stable_foreground + stable_background).clamp(0, 1) * low_disagreement_gate
        geometry_weight = (w_geo * disagreement_weight).clamp(0, 1)
        return RouterOutput(
            stable_foreground,
            stable_background,
            disagreement_weight,
            fused.clamp(0, 1),
            consensus_mask,
            consensus_weight,
            geometry_weight,
            w_sem,
            w_app,
            w_geo,
        )
