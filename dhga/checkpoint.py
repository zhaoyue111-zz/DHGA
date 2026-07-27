from __future__ import annotations

from pathlib import Path
from typing import Any
import random
import numpy as np

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


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: DHGAConfig,
    optimizer: torch.optim.Optimizer | None = None,
    ema: nn.Module | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "format": "dhga_training_checkpoint_v1",
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    ema: nn.Module | None = None,
    scaler: Any | None = None,
    load_training_state: bool = True,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") == "dhga_checkpoint_v1":
        state = payload["state_dicts"].get("dhga_model")
        if state is None:
            raise RuntimeError("DHGA checkpoint does not contain dhga_model")
        model.load_state_dict(state, strict=True)
        return {"epoch": 0, "global_step": 0, "metadata": payload.get("metadata", {})}
    if payload.get("format") != "dhga_training_checkpoint_v1":
        raise RuntimeError("Not a DHGA training checkpoint")
    metadata = payload.get("metadata", {})
    checkpoint_stage = metadata.get("stage") or payload.get("config", {}).get("dhga_stage")
    if expected_stage is not None and checkpoint_stage is not None and str(checkpoint_stage) != str(expected_stage):
        raise RuntimeError(f"Checkpoint stage {checkpoint_stage} does not match current stage {expected_stage}")
    model.load_state_dict(payload["model"], strict=True)
    if load_training_state and optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if load_training_state and ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"], strict=True)
    if load_training_state and scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if load_training_state and payload.get("rng_state"):
        torch.set_rng_state(payload["rng_state"]["torch"])
        if payload["rng_state"].get("python") is not None:
            random.setstate(payload["rng_state"]["python"])
        if payload["rng_state"].get("numpy") is not None:
            np.random.set_state(payload["rng_state"]["numpy"])
        if torch.cuda.is_available() and payload["rng_state"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])
    return payload


def load_dhga_checkpoint(path: str | Path, modules: dict[str, nn.Module]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "dhga_checkpoint_v1":
        raise RuntimeError("Not a DHGA checkpoint; refusing silent migration")
    missing_modules = sorted(set(payload["state_dicts"]) - set(modules))
    unexpected_modules = sorted(set(modules) - set(payload["state_dicts"]))
    if missing_modules or unexpected_modules:
        raise RuntimeError(f"DHGA module mismatch missing={missing_modules} unexpected={unexpected_modules}")
    for name, module in modules.items():
        module.load_state_dict(payload["state_dicts"][name], strict=True)
    return payload
