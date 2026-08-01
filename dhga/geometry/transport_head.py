from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GeometryTransportHead(nn.Module):
    """Light ray-token head initialized to prefer zero displacement."""
    """对每一条射线 token，输出各个偏移位置的分类概率分布，算出期望位移（毫米）"""
    def __init__(self, in_channels: int, offsets_mm: Tensor, hidden_channels: int = 32) -> None:
        super().__init__()
        self.register_buffer("offsets_mm", offsets_mm.float())
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        center = int(torch.argmin(self.offsets_mm.abs()).item())
        self.center_index = center
        self.center_bias = nn.Parameter(torch.tensor(2.0))
        step = torch.diff(self.offsets_mm).abs().min().clamp_min(1e-6) if self.offsets_mm.numel() > 1 else torch.tensor(1.0)
        prior = -1.0 * self.offsets_mm.abs() / step # # 距离0越远，先验分值越低
        prior[center] = 0.0
        self.register_buffer("zero_displacement_prior", prior.float())

    def forward(self, ray_tokens: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        if ray_tokens.ndim != 4:
            raise ValueError("ray_tokens must have shape [B, N, K, C]")
        bsz, n_points, n_offsets, channels = ray_tokens.shape
        logits = self.net(ray_tokens.view(bsz * n_points, n_offsets, channels).permute(0, 2, 1)).squeeze(1)
        logits = logits + self.zero_displacement_prior.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        logits[:, self.center_index] = logits[:, self.center_index] + self.center_bias.to(logits.dtype)
        if valid_mask is not None:
            flat_valid = valid_mask.view(bsz * n_points, n_offsets)
            any_valid = flat_valid.any(dim=-1, keepdim=True)
            flat_valid = torch.where(any_valid, flat_valid, torch.zeros_like(flat_valid).scatter(1, torch.full((flat_valid.shape[0], 1), self.center_index, device=flat_valid.device), True))
            logits = logits.masked_fill(~flat_valid, -1e4)
        else:
            any_valid = torch.ones((bsz * n_points, 1), device=ray_tokens.device, dtype=torch.bool)
        probs = F.softmax(logits, dim=-1).view(bsz, n_points, n_offsets)
        offsets = self.offsets_mm.to(device=ray_tokens.device, dtype=ray_tokens.dtype)
        expected = (probs * offsets.view(1, 1, -1)).sum(dim=-1)
        expected = torch.where(any_valid.view(bsz, n_points), expected, torch.zeros_like(expected))
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = torch.where(any_valid.view(bsz, n_points), entropy, torch.zeros_like(entropy))
        return {"logits": logits.view(bsz, n_points, n_offsets), "prob": probs, "expected_displacement_mm": expected, "entropy": entropy, "valid_points": any_valid.view(bsz, n_points)}
