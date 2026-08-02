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


# 262,144 sampled points are sufficient for stable per-case quantiles while
# avoiding torch.quantile on a complete, potentially very large 3-D volume.
_MAX_QUANTILE_SAMPLES = 262_144

# The router spatial head begins with a 3x3x3 convolution.  Whole-volume
# evaluation can contain more than one hundred million voxels, so concatenating
# all six router channels at once may require several GiB.  Training patches
# (for example 192^3) stay on the original fast path; only larger volumes are
# evaluated in depth chunks with a one-voxel halo, which preserves the exact
# receptive field of the spatial head.
_MAX_FULL_ROUTER_VOXELS = 8_388_608
_MAX_ROUTER_CHUNK_VOXELS = 2_097_152
_ROUTER_HALO = 1


def _sampled_case_quantile(
    x: Tensor,
    q: float,
    max_samples: int = _MAX_QUANTILE_SAMPLES,
) -> Tensor:
    """Estimate a per-case, per-channel quantile with bounded memory.

    Returns shape [B, C, 1]. Sampling is deterministic so evaluation and
    checkpoint resume remain reproducible.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if max_samples < 2:
        raise ValueError(f"max_samples must be at least 2, got {max_samples}")

    flat = x.flatten(2).float()
    num_voxels = flat.shape[-1]

    if num_voxels == 0:
        raise ValueError("Cannot calculate a quantile of an empty tensor")

    if num_voxels > max_samples:
        # Keep the index calculation in int64. This avoids float32 linspace
        # rounding the final index to num_voxels on very large volumes.
        positions = torch.arange(
            max_samples,
            device=flat.device,
            dtype=torch.int64,
        )
        indices = torch.div(
            positions * (num_voxels - 1),
            max_samples - 1,
            rounding_mode="floor",
        )

        indices[0] = 0
        indices[-1] = num_voxels - 1

        index_min = int(indices.min().item())
        index_max = int(indices.max().item())
        if index_min < 0 or index_max >= num_voxels:
            raise RuntimeError(
                "Quantile sampling produced invalid indices: "
                f"min={index_min}, max={index_max}, "
                f"num_voxels={num_voxels}, max_samples={max_samples}"
            )
        sampled = flat.index_select(-1, indices)
    else:
        sampled = flat

    return torch.quantile(sampled, q, dim=-1, keepdim=True)


def _case_rank_normalize(x: Tensor) -> Tensor:
    flat = x.flatten(2).float()

    lo = _sampled_case_quantile(x, 0.05)
    hi = _sampled_case_quantile(x, 0.95)

    normalized = (flat - lo) / (hi - lo).clamp_min(1e-6)
    return normalized.clamp(0.0, 1.0).view_as(x).to(dtype=x.dtype)


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
            nn.Conv3d(6, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(8, 3, kernel_size=1),
        )
        nn.init.zeros_(self.spatial_head[-1].weight)
        nn.init.zeros_(self.spatial_head[-1].bias)
        with torch.no_grad():
            self.spatial_head[-1].bias[:] = torch.tensor([0.0, 0.0, -3.0])

    def _spatial_weights(
        self,
        sem_prob: Tensor,
        app_prob: Tensor,
        disagreement_weight: Tensor,
        stable_foreground: Tensor,
        stable_background: Tensor,
        visual_context: Tensor,
    ) -> Tensor:
        """Run the spatial head without materializing a huge six-channel volume.

        For ordinary training patches this uses the original whole-volume path.
        Large evaluation volumes are split along depth.  A one-voxel halo is
        included because the only spatially non-pointwise layer is the first
        3x3x3 convolution; after cropping the halo, chunked output is equivalent
        to whole-volume output up to normal floating-point kernel variation.
        """
        num_voxels = int(sem_prob.shape[-3] * sem_prob.shape[-2] * sem_prob.shape[-1])
        features = (
            sem_prob,
            app_prob,
            disagreement_weight,
            stable_foreground,
            stable_background,
            visual_context,
        )

        # Preserve the original autograd path for training. Chunking is an
        # evaluation-only memory optimization.
        if num_voxels <= _MAX_FULL_ROUTER_VOXELS or torch.is_grad_enabled():
            route_features = torch.cat(features, dim=1)
            return self.spatial_head(route_features).softmax(dim=1)

        batch_size = sem_prob.shape[0]
        depth, height, width = sem_prob.shape[-3:]
        plane_voxels = max(int(height * width), 1)
        core_depth = max(1, _MAX_ROUTER_CHUNK_VOXELS // plane_voxels)

        weights = torch.empty(
            (batch_size, 3, depth, height, width),
            device=sem_prob.device,
            dtype=sem_prob.dtype,
        )

        for core_start in range(0, depth, core_depth):
            core_end = min(core_start + core_depth, depth)
            read_start = max(0, core_start - _ROUTER_HALO)
            read_end = min(depth, core_end + _ROUTER_HALO)

            route_chunk = torch.cat(
                tuple(feature[..., read_start:read_end, :, :] for feature in features),
                dim=1,
            )
            chunk_weights = self.spatial_head(route_chunk).softmax(dim=1)

            local_start = core_start - read_start
            local_end = local_start + (core_end - core_start)
            weights[..., core_start:core_end, :, :] = chunk_weights[
                ..., local_start:local_end, :, :
            ]

            del route_chunk, chunk_weights

        return weights

    @staticmethod
    def _fuse_probabilities(
        sem_prob: Tensor,
        app_prob: Tensor,
        w_sem: Tensor,
        w_app: Tensor,
    ) -> Tensor:
        """Fuse experts with bounded peak memory during large-volume evaluation."""
        num_voxels = int(sem_prob.shape[-3] * sem_prob.shape[-2] * sem_prob.shape[-1])
        if num_voxels <= _MAX_FULL_ROUTER_VOXELS or torch.is_grad_enabled():
            region_denom = (w_sem + w_app).clamp_min(1e-6)
            region_w_sem = w_sem / region_denom
            region_w_app = w_app / region_denom
            return region_w_sem * sem_prob + region_w_app * app_prob

        depth, height, width = sem_prob.shape[-3:]
        plane_voxels = max(int(height * width), 1)
        core_depth = max(1, _MAX_ROUTER_CHUNK_VOXELS // plane_voxels)
        fused = torch.empty_like(sem_prob)

        for core_start in range(0, depth, core_depth):
            core_end = min(core_start + core_depth, depth)
            index = (..., slice(core_start, core_end), slice(None), slice(None))
            sem_chunk = sem_prob[index]
            app_chunk = app_prob[index]
            w_sem_chunk = w_sem[index]
            w_app_chunk = w_app[index]
            denom = (w_sem_chunk + w_app_chunk).clamp_min(1e-6)
            fused[index] = (
                w_sem_chunk * sem_chunk + w_app_chunk * app_chunk
            ) / denom
            del denom

        return fused

    def forward(
        self,
        sem_prob: Tensor,
        app_prob: Tensor,
        sem_stability: Tensor | None = None,
        app_stability: Tensor | None = None,
        visual_context: Tensor | None = None,
    ) -> RouterOutput:
        if sem_prob.shape != app_prob.shape:
            raise ValueError(
                f"expert probability shapes differ: {sem_prob.shape} vs {app_prob.shape}"
            )

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

        # The normalized tensors above are the only versions needed below.
        # Releasing the raw full-volume tensors before the spatial head saves
        # three complete channels during evaluation.
        del common_fg, common_bg, disagreement

        if visual_context is None:
            visual_context = torch.zeros_like(sem_prob)
        elif visual_context.shape[-3:] != sem_prob.shape[-3:]:
            visual_context = torch.nn.functional.interpolate(
                visual_context,
                size=sem_prob.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        if visual_context.shape[1] != 1:
            visual_context = visual_context.mean(dim=1, keepdim=True)

        weights = self._spatial_weights(
            sem_prob,
            app_prob,
            disagreement_weight,
            stable_foreground,
            stable_background,
            visual_context,
        )
        w_sem = weights[:, 0:1]
        w_app = weights[:, 1:2]
        w_geo = weights[:, 2:3]

        fused = self._fuse_probabilities(sem_prob, app_prob, w_sem, w_app)

        disagreement_threshold = _sampled_case_quantile(
            disagreement_weight,
            self.disagreement_quantile,
        ).view(
            disagreement_weight.shape[0],
            disagreement_weight.shape[1],
            1,
            1,
            1,
        ).to(dtype=disagreement_weight.dtype)
        consensus_mask = (stable_foreground > stable_background) & (
            disagreement_weight < disagreement_threshold
        )

        low_disagreement_gate = (0.5 - disagreement_weight).clamp_min(0.0) * 2.0
        consensus_weight = (
            (stable_foreground + stable_background).clamp(0, 1)
            * low_disagreement_gate
        )
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