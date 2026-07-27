from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor, nn


class EMATeacher:
    """EMA over trainable DHGA parameters only; frozen VoxTell weights stay shared."""

    def __init__(self, student: nn.Module, decay: float = 0.99) -> None:
        self.decay = float(decay)
        self.names = [name for name, param in student.named_parameters() if param.requires_grad]
        self.shadow: dict[str, Tensor] = {
            name: param.detach().clone()
            for name, param in student.named_parameters()
            if name in self.names
        }

    def state_dict(self) -> dict[str, Tensor | list[str] | float]:
        return {"decay": self.decay, "names": list(self.names), "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.decay = float(state["decay"])
        self.names = list(state["names"])
        self.shadow = {k: v.detach().clone() for k, v in state["shadow"].items()}

    def update(self, student: nn.Module) -> None:
        params = dict(student.named_parameters())
        for name in self.names:
            if name not in params:
                continue
            value = params[name].detach()
            if name not in self.shadow:
                self.shadow[name] = value.clone()
            else:
                self.shadow[name] = self.shadow[name].to(device=value.device, dtype=value.dtype)
                self.shadow[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)

    def sync_from(self, student: nn.Module) -> None:
        """Hard-copy current student trainable parameters into the EMA shadow."""
        params = dict(student.named_parameters())
        self.shadow = {
            name: params[name].detach().clone()
            for name in self.names
            if name in params
        }

    @contextmanager
    def apply_to(self, student: nn.Module):
        params = dict(student.named_parameters())
        originals: dict[str, Tensor] = {}
        with torch.no_grad():
            for name in self.names:
                if name in params and name in self.shadow:
                    originals[name] = params[name].detach().clone()
                    params[name].copy_(self.shadow[name].to(device=params[name].device, dtype=params[name].dtype))
        try:
            yield student
        finally:
            with torch.no_grad():
                for name, value in originals.items():
                    params[name].copy_(value)
