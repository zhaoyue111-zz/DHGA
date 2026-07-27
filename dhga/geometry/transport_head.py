from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GeometryTransportHead(nn.Module):
    """Light ray-token head initialized to prefer zero displacement."""

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

    def forward(self, ray_tokens: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        if ray_tokens.ndim != 4:
            raise ValueError("ray_tokens must have shape [B, N, K, C]")
        bsz, n_points, n_offsets, channels = ray_tokens.shape
        logits = self.net(ray_tokens.view(bsz * n_points, n_offsets, channels).permute(0, 2, 1)).squeeze(1)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask.view(bsz * n_points, n_offsets), -1e4)
        probs = F.softmax(logits, dim=-1).view(bsz, n_points, n_offsets)
        offsets = self.offsets_mm.to(device=ray_tokens.device, dtype=ray_tokens.dtype)
        expected = (probs * offsets.view(1, 1, -1)).sum(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
        return {"logits": logits.view(bsz, n_points, n_offsets), "prob": probs, "expected_displacement_mm": expected, "entropy": entropy}
