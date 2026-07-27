from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import DHGAConfig


DHGA_PREFIX = "dhga"


def save_dhga_checkpoint(path: str | Path, modules: dict[str, nn.Module], config: DHGAConfig, metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "format": "dhga_checkpoint_v1",
        "config": config.to_dict(),
        "metadata": metadata or {},
        "state_dicts": {name: module.state_dict() for name, module in modules.items()},
    }
    torch.save(payload, path)


def load_dhga_checkpoint(path: str | Path, modules: dict[str, nn.Module]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "dhga_checkpoint_v1":
        raise RuntimeError("Not a DHGA checkpoint; refusing silent migration")
    missing_modules = sorted(set(payload["state_dicts"]) - set(modules))
    unexpected_modules = sorted(set(modules) - set(payload["state_dicts"]))
    if missing_modules or unexpected_modules:
        raise RuntimeError(f"DHGA module mismatch missing={missing_modules} unexpected={unexpected_modules}")
    for name, module in modules.items():
        module.load_state_dict(payload["state_dicts"][name], strict=True)
    return payload
