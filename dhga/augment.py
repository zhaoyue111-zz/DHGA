from __future__ import annotations

import torch
from torch import Tensor


def weak_strong_intensity_views(image: Tensor, noise_std: float = 0.03, gamma_range: tuple[float, float] = (0.9, 1.1)) -> tuple[Tensor, Tensor, dict]:
    weak = image
    strong = image.clone()
    scale = 1.0 + (torch.rand((image.shape[0], 1, 1, 1, 1), device=image.device, dtype=image.dtype) - 0.5) * 0.1
    strong = strong * scale
    if noise_std > 0:
        strong = strong + torch.randn_like(strong) * float(noise_std)
    min_v = strong.amin(dim=(2, 3, 4), keepdim=True)
    max_v = strong.amax(dim=(2, 3, 4), keepdim=True)
    norm = (strong - min_v) / (max_v - min_v).clamp_min(1e-6)
    gamma = torch.empty((image.shape[0], 1, 1, 1, 1), device=image.device, dtype=image.dtype).uniform_(*gamma_range)
    strong = norm.clamp(0, 1).pow(gamma) * (max_v - min_v) + min_v
    return weak, strong, {"scale": scale.detach(), "gamma": gamma.detach(), "noise_std": noise_std}
