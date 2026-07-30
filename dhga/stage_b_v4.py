from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def resolve_layer_indices(layer_indices: list[int], num_features: int) -> list[int]:
    resolved: list[int] = []
    for raw_idx in layer_indices:
        idx = raw_idx if raw_idx >= 0 else num_features + raw_idx
        if idx < 0 or idx >= num_features:
            raise ValueError(f"feature layer {raw_idx} is outside {num_features} stages")
        if idx not in resolved:
            resolved.append(idx)
    if not resolved:
        raise ValueError("at least one feature layer is required")
    return resolved


def capped_fusion_size(shapes: list[tuple[int, int, int]], max_voxels: int) -> tuple[int, int, int]:
    target = max(shapes, key=lambda shape: math.prod(shape))
    d, h, w = (int(v) for v in target)
    voxels = max(d * h * w, 1)
    if voxels <= max_voxels:
        return d, h, w
    scale = (float(max_voxels) / float(voxels)) ** (1.0 / 3.0)
    return tuple(max(1, int(math.floor(v * scale))) for v in (d, h, w))


class AppearanceSegHead(nn.Module):
    def __init__(
        self,
        feature_channels: list[int],
        layer_indices: list[int],
        proj_channels: int = 16,
        max_fusion_voxels: int = 64 * 64 * 64,
    ) -> None:
        super().__init__()
        self.resolved_layer_indices = resolve_layer_indices(layer_indices, len(feature_channels))
        self.proj_channels = int(proj_channels)
        self.max_fusion_voxels = int(max_fusion_voxels)
        self.projections = nn.ModuleDict({
            str(idx): nn.Conv3d(int(feature_channels[idx]), self.proj_channels, kernel_size=1)
            for idx in self.resolved_layer_indices
        })
        fusion_channels = self.proj_channels * len(self.resolved_layer_indices)
        groups = max(group for group in range(1, min(8, self.proj_channels) + 1) if self.proj_channels % group == 0)
        self.fusion = nn.Sequential(
            nn.Conv3d(fusion_channels, self.proj_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, self.proj_channels),
            nn.GELU(),
            nn.Conv3d(self.proj_channels, self.proj_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, self.proj_channels),
            nn.GELU(),
        )
        self.logit_head = nn.Conv3d(self.proj_channels, 1, kernel_size=1)

    def forward(self, encoder_stages: list[Tensor], semantic_shape: tuple[int, int, int]) -> dict[str, Tensor]:
        selected = [encoder_stages[idx] for idx in self.resolved_layer_indices]
        fusion_size = capped_fusion_size([tuple(t.shape[-3:]) for t in selected], self.max_fusion_voxels)
        projected = []
        for idx, feature in zip(self.resolved_layer_indices, selected):
            x = self.projections[str(idx)](feature)
            if tuple(x.shape[-3:]) != fusion_size:
                x = F.interpolate(x, size=fusion_size, mode="trilinear", align_corners=False)
            projected.append(x)
        appearance_feature = self.fusion(torch.cat(projected, dim=1))
        logits_lowres = self.logit_head(appearance_feature)
        logits = logits_lowres
        if tuple(logits.shape[-3:]) != tuple(semantic_shape):
            logits = F.interpolate(logits, size=semantic_shape, mode="trilinear", align_corners=False)
        return {
            "appearance_feature": appearance_feature,
            "appearance_logits_lowres": logits_lowres,
            "appearance_logits": logits,
        }


def _l2_normalize_features(x: Tensor) -> Tensor:
    return F.normalize(x.float(), dim=1, eps=1e-6)


class TargetPrototypeBank(nn.Module):
    def __init__(
        self,
        semantic_dim: int,
        appearance_dim: int,
        semantic_fg: int = 3,
        semantic_bg: int = 5,
        appearance_fg: int = 3,
        appearance_bg: int = 5,
        tau: float = 0.1,
    ) -> None:
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("semantic_fg", torch.zeros(semantic_fg, semantic_dim))
        self.register_buffer("semantic_bg", torch.zeros(semantic_bg, semantic_dim))
        self.register_buffer("appearance_fg", torch.zeros(appearance_fg, appearance_dim))
        self.register_buffer("appearance_bg", torch.zeros(appearance_bg, appearance_dim))
        self.register_buffer("semantic_fg_valid", torch.zeros(semantic_fg, dtype=torch.bool))
        self.register_buffer("semantic_bg_valid", torch.zeros(semantic_bg, dtype=torch.bool))
        self.register_buffer("appearance_fg_valid", torch.zeros(appearance_fg, dtype=torch.bool))
        self.register_buffer("appearance_bg_valid", torch.zeros(appearance_bg, dtype=torch.bool))
        self.register_buffer("update_count", torch.zeros(4, dtype=torch.long))
        self.register_buffer("assignment_count", torch.zeros(4, max(semantic_fg, semantic_bg, appearance_fg, appearance_bg), dtype=torch.long))

    @torch.no_grad()
    def initialize_view(self, view: str, fg_samples: Tensor, bg_samples: Tensor, seed: int = 0) -> None:
        fg, fg_valid = self._view_tensors(view, True)
        bg, bg_valid = self._view_tensors(view, False)
        self._init_prototypes(fg, fg_valid, fg_samples, seed)
        self._init_prototypes(bg, bg_valid, bg_samples, seed + 17)

    @torch.no_grad()
    def _init_prototypes(self, proto: Tensor, valid: Tensor, samples: Tensor, seed: int) -> None:
        proto.zero_()
        valid.zero_()
        if samples.numel() == 0:
            return
        samples = F.normalize(samples.float().cpu(), dim=1, eps=1e-6).to(proto.device)
        count = min(proto.shape[0], samples.shape[0])
        if count <= 0:
            return
        chosen = deterministic_farthest_points(samples, count, seed)
        proto[:count].copy_(chosen.to(proto.device, proto.dtype))
        valid[:count] = True

    def prototype_probability(self, view: str, feature: Tensor, tau: float | None = None) -> tuple[Tensor, Tensor]:
        fg, fg_valid = self._view_tensors(view, True)
        bg, bg_valid = self._view_tensors(view, False)
        z = _l2_normalize_features(feature)
        fg_score = self._max_cosine(z, fg, fg_valid)
        bg_score = self._max_cosine(z, bg, bg_valid)
        any_fg = bool(fg_valid.any().detach().cpu())
        any_bg = bool(bg_valid.any().detach().cpu())
        if not any_fg and not any_bg:
            q = torch.full_like(fg_score, 0.5)
        elif not any_fg:
            q = torch.zeros_like(fg_score)
        elif not any_bg:
            q = torch.ones_like(fg_score)
        else:
            q = torch.sigmoid((fg_score - bg_score) / float(self.tau if tau is None else tau))
        confidence = 2.0 * (q - 0.5).abs()
        return q.detach(), confidence.detach()

    def _max_cosine(self, feature: Tensor, proto: Tensor, valid: Tensor) -> Tensor:
        if not bool(valid.any().detach().cpu()):
            return torch.zeros((feature.shape[0], 1, *feature.shape[-3:]), device=feature.device, dtype=feature.dtype)
        p = F.normalize(proto[valid].to(device=feature.device, dtype=feature.dtype), dim=1, eps=1e-6)
        score = torch.einsum("bcdhw,kc->bkdhw", feature, p)
        return score.max(dim=1, keepdim=True).values

    @torch.no_grad()
    def ema_update_nearest(self, view: str, feature: Tensor, label: Tensor, mask: Tensor, momentum: float = 0.99) -> None:
        for foreground in (True, False):
            proto, valid = self._view_tensors(view, foreground)
            if not bool(valid.any().cpu()):
                continue
            target = (label >= 0.5) if foreground else (label < 0.5)
            active = (mask.bool() & target.bool()).expand(-1, feature.shape[1], -1, -1, -1)
            if not bool(active.any().detach().cpu()):
                continue
            samples = feature.detach().permute(0, 2, 3, 4, 1)[active[:, 0]].float()
            center = F.normalize(samples.mean(dim=0, keepdim=True), dim=1, eps=1e-6)
            p = F.normalize(proto[valid].float(), dim=1, eps=1e-6)
            nearest_local = torch.matmul(center.to(p.device), p.t()).argmax(dim=1).item()
            valid_indices = valid.nonzero(as_tuple=False).flatten()
            proto_idx = int(valid_indices[nearest_local])
            proto[proto_idx].mul_(momentum).add_(center[0].to(proto.device, proto.dtype), alpha=1.0 - momentum)
            proto[proto_idx].copy_(F.normalize(proto[proto_idx : proto_idx + 1].float(), dim=1, eps=1e-6)[0].to(proto.dtype))
            row = self._assignment_row(view, foreground)
            self.update_count[row] += 1
            self.assignment_count[row, proto_idx] += int(samples.shape[0])

    def _assignment_row(self, view: str, foreground: bool) -> int:
        if view == "semantic":
            return 0 if foreground else 1
        if view == "appearance":
            return 2 if foreground else 3
        raise ValueError(f"unknown prototype view {view}")

    def _view_tensors(self, view: str, foreground: bool) -> tuple[Tensor, Tensor]:
        if view == "semantic" and foreground:
            return self.semantic_fg, self.semantic_fg_valid
        if view == "semantic" and not foreground:
            return self.semantic_bg, self.semantic_bg_valid
        if view == "appearance" and foreground:
            return self.appearance_fg, self.appearance_fg_valid
        if view == "appearance" and not foreground:
            return self.appearance_bg, self.appearance_bg_valid
        raise ValueError(f"unknown prototype view {view}")

    def valid_counts(self) -> dict[str, int]:
        return {
            "semantic_fg": int(self.semantic_fg_valid.sum().cpu()),
            "semantic_bg": int(self.semantic_bg_valid.sum().cpu()),
            "appearance_fg": int(self.appearance_fg_valid.sum().cpu()),
            "appearance_bg": int(self.appearance_bg_valid.sum().cpu()),
        }


def deterministic_farthest_points(samples: Tensor, count: int, seed: int = 0) -> Tensor:
    samples = F.normalize(samples.float(), dim=1, eps=1e-6)
    if samples.shape[0] <= count:
        return samples
    generator = torch.Generator(device=samples.device)
    generator.manual_seed(int(seed))
    first = int(torch.randint(samples.shape[0], (1,), generator=generator, device=samples.device).item())
    selected = [first]
    min_dist = 1.0 - torch.matmul(samples, samples[first])
    for _ in range(1, count):
        idx = int(min_dist.argmax().item())
        selected.append(idx)
        dist = 1.0 - torch.matmul(samples, samples[idx])
        min_dist = torch.minimum(min_dist, dist)
    return samples[torch.tensor(selected, device=samples.device)]


@dataclass
class ReliabilityMasks:
    joint: Tensor
    joint_fg: Tensor
    joint_bg: Tensor
    semantic_only: Tensor
    appearance_only: Tensor
    conflict: Tensor
    reject: Tensor
    sem_target: Tensor
    app_target: Tensor
    sem_weight: Tensor
    app_weight: Tensor


def make_reliability_masks(
    q_sem: Tensor,
    q_app: Tensor,
    sem_prob: Tensor,
    app_prob: Tensor,
    high: float = 0.8,
    low: float = 0.2,
) -> ReliabilityMasks:
    sem_rel = (q_sem >= high) | (q_sem <= low)
    app_rel = (q_app >= high) | (q_app <= low)
    sem_label = q_sem >= 0.5
    app_label = q_app >= 0.5
    labels_same = sem_label == app_label
    joint = sem_rel & app_rel & labels_same
    proto_fg = (sem_label & sem_rel) | (app_label & app_rel)
    proto_bg = ((~sem_label) & sem_rel) | ((~app_label) & app_rel)
    strong_fg_conflict = proto_fg & (sem_prob < 0.2) & (app_prob < 0.2)
    strong_bg_conflict = proto_bg & (sem_prob > 0.8) & (app_prob > 0.8)
    conflict = (sem_rel & app_rel & ~labels_same) | strong_fg_conflict | strong_bg_conflict
    joint = joint & ~conflict
    semantic_only = sem_rel & ~app_rel & ~conflict
    appearance_only = app_rel & ~sem_rel & ~conflict
    used = joint | semantic_only | appearance_only | conflict
    reject = ~used
    sem_target = torch.where(joint, sem_label.float(), sem_label.float())
    app_target = torch.where(joint, app_label.float(), app_label.float())
    sem_weight = (joint.float() + 0.5 * semantic_only.float()).detach()
    app_weight = (joint.float() + 0.5 * appearance_only.float()).detach()
    return ReliabilityMasks(
        joint=joint.detach(),
        joint_fg=(joint & sem_label).detach(),
        joint_bg=(joint & ~sem_label).detach(),
        semantic_only=semantic_only.detach(),
        appearance_only=appearance_only.detach(),
        conflict=conflict.detach(),
        reject=reject.detach(),
        sem_target=sem_target.detach(),
        app_target=app_target.detach(),
        sem_weight=sem_weight,
        app_weight=app_weight,
    )


def balanced_masked_bce(prob: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    if not bool((weight > 0).any().detach().cpu()):
        return prob.sum() * 0.0
    prob = prob.float().clamp(1e-5, 1.0 - 1e-5)
    target = target.float()
    weight = weight.float()
    pos = weight * (target >= 0.5).float()
    neg = weight * (target < 0.5).float()
    loss = F.binary_cross_entropy(prob, target, reduction="none")
    total = prob.sum() * 0.0
    if bool((pos > 0).any().detach().cpu()):
        total = total + 0.5 * (loss * pos).sum() / pos.sum().clamp_min(1.0)
    if bool((neg > 0).any().detach().cpu()):
        total = total + 0.5 * (loss * neg).sum() / neg.sum().clamp_min(1.0)
    return total


def masked_dice_loss(prob: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    active = (weight > 0) & (target >= 0.5)
    if not bool(active.any().detach().cpu()):
        return prob.sum() * 0.0
    p = prob.float() * weight.float()
    t = target.float() * weight.float()
    intersection = (p * t).sum()
    denom = p.sum() + t.sum()
    return 1.0 - (2.0 * intersection + 1e-6) / denom.clamp_min(1e-6)


def stage_b_v4_expert_loss(prob: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    return balanced_masked_bce(prob, target, weight) + 0.5 * masked_dice_loss(prob, target, weight)


def prototype_reliability_fusion(
    sem_prob: Tensor,
    app_prob: Tensor,
    q_sem: Tensor,
    q_app: Tensor,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    sem_conflict = (((q_sem >= 0.5) & (sem_prob < 0.2)) | ((q_sem < 0.5) & (sem_prob > 0.8))).float()
    app_conflict = (((q_app >= 0.5) & (app_prob < 0.2)) | ((q_app < 0.5) & (app_prob > 0.8))).float()
    r_sem = (2.0 * (q_sem - 0.5).abs()) * (0.5 + 0.5 * 2.0 * (sem_prob - 0.5).abs()) * (1.0 - sem_conflict)
    r_app = (2.0 * (q_app - 0.5).abs()) * (0.5 + 0.5 * 2.0 * (app_prob - 0.5).abs()) * (1.0 - app_conflict)
    denom = r_sem + r_app
    low = denom < eps
    w_sem = torch.where(low, torch.full_like(denom, 0.5), r_sem / denom.clamp_min(eps))
    w_app = torch.where(low, torch.full_like(denom, 0.5), r_app / denom.clamp_min(eps))
    fused = (w_sem * sem_prob + w_app * app_prob).clamp(0, 1)
    geo_trigger = (sem_prob - app_prob).abs() * (1.0 - torch.maximum(r_sem, r_app).clamp(0, 1))
    return {"fused_prob": fused, "w_sem": w_sem, "w_app": w_app, "r_sem": r_sem, "r_app": r_app, "geo_trigger": geo_trigger}


def tensor_corr(a: Tensor, b: Tensor, mask: Tensor | None = None) -> Tensor:
    x = a.detach().float()
    y = b.detach().float()
    if mask is not None:
        m = mask.detach().bool().expand_as(x)
        if not bool(m.any().cpu()):
            return x.new_tensor(0.0)
        x = x[m]
        y = y[m]
    else:
        x = x.flatten()
        y = y.flatten()
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (x.square().mean().sqrt() * y.square().mean().sqrt()).clamp_min(1e-6)
