from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class SharedVoxTellFeatures:
    image: Tensor
    encoder_stages: list[Tensor]
    selected_feature: Tensor | None = None
    text_embedding: Tensor | None = None
    mask_embedding: Tensor | None = None
    prompt_decoder_memory: Tensor | None = None
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    crop_bbox: Any = None
    original_shape: tuple[int, int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SharedEncoderOnce(nn.Module):
    """Small testable wrapper proving both experts consume one shared feature cache."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_calls = 0

    def forward(self, image: Tensor, **metadata: Any) -> SharedVoxTellFeatures:
        self.num_calls += 1
        stages = self.encoder(image)
        if isinstance(stages, Tensor):
            stages = [stages]
        stages = list(stages)
        return SharedVoxTellFeatures(
            image=image,
            encoder_stages=stages,
            selected_feature=stages[-1] if stages else None,
            spacing=tuple(metadata.pop("spacing", (1.0, 1.0, 1.0))),
            crop_bbox=metadata.pop("crop_bbox", None),
            original_shape=metadata.pop("original_shape", tuple(image.shape[-3:])),
            metadata=metadata,
        )


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def trainable_parameter_summary(named_modules: dict[str, nn.Module]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    all_names: list[str] = []
    for prefix, module in named_modules.items():
        names = [f"{prefix}.{name}" for name, param in module.named_parameters() if param.requires_grad]
        summary[prefix] = {
            "trainable_params": sum(param.numel() for param in module.parameters() if param.requires_grad),
            "trainable_names": names[:64],
            "num_trainable_names": len(names),
        }
        all_names.extend(names)
    summary["total_trainable_params"] = sum(
        param.numel() for module in named_modules.values() for param in module.parameters() if param.requires_grad
    )
    summary["all_trainable_names"] = all_names[:256]
    return summary
