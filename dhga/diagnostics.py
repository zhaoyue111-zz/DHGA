from __future__ import annotations

import torch
from torch import Tensor, nn


def grad_norms(modules: dict[str, nn.Module]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, module in modules.items():
        total = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().norm().cpu()) ** 2
        values[f"{name}_grad_norm"] = total ** 0.5
    return values


def displacement_diagnostics(displacement: Tensor, entropy: Tensor | None = None) -> dict[str, float]:
    disp = displacement.detach().float()
    out = {
        "dhga_abs_displacement_mm": float(disp.abs().mean().cpu()),
        "dhga_positive_displacement_ratio": float((disp > 1e-4).float().mean().cpu()),
        "dhga_negative_displacement_ratio": float((disp < -1e-4).float().mean().cpu()),
        "dhga_near_zero_displacement_ratio": float((disp.abs() <= 1e-4).float().mean().cpu()),
    }
    if entropy is not None:
        out["dhga_displacement_entropy"] = float(entropy.detach().float().mean().cpu())
    return out
