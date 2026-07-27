from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dhga.shared_voxtell import SharedVoxTellFeatures


class AppearanceFeatureAdapter(nn.Module):
    """Light residual 3D feature adapter used on selected VoxTell skip features."""

    def __init__(self, channels: int, hidden_ratio: float = 0.25, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, int(channels * hidden_ratio))
        groups = max(group for group in range(1, min(8, hidden) + 1) if hidden % group == 0)
        self.block = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=hidden),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.block[-1].weight)

    def forward(self, feature: Tensor) -> Tensor:
        return feature + self.block(feature)


class AppearanceExpert(nn.Module):
    """Feature-adapter-biased expert with a path distinct from semantic LoRA."""

    def __init__(
        self,
        feature_channels: list[int],
        layer_indices: list[int],
        hidden_ratio: float = 0.25,
        dropout: float = 0.0,
        decoder_forward: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.decoder_forward = decoder_forward
        self.resolved_layer_indices: list[int] = []
        self.adapters = nn.ModuleDict()
        for raw_idx in self.layer_indices:
            idx = raw_idx if raw_idx >= 0 else len(feature_channels) + raw_idx
            if idx < 0 or idx >= len(feature_channels):
                raise ValueError(f"appearance feature layer {raw_idx} is outside {len(feature_channels)} stages")
            self.resolved_layer_indices.append(idx)
            self.adapters[str(idx)] = AppearanceFeatureAdapter(feature_channels[idx], hidden_ratio, 0.0)

    def adapt_features(self, features: SharedVoxTellFeatures, feature_dropout: float = 0.0) -> SharedVoxTellFeatures:
        adapted = list(features.encoder_stages)
        for key, adapter in self.adapters.items():
            idx = int(key)
            adapted_feature = adapter(adapted[idx])
            if self.training and feature_dropout > 0:
                adapted_feature = F.dropout3d(adapted_feature, p=feature_dropout, training=True)
            adapted[idx] = adapted_feature
        selected_idx = features.metadata.get("selected_feature_idx")
        selected = adapted[selected_idx] if selected_idx is not None else features.selected_feature
        return SharedVoxTellFeatures(
            image=features.image,
            encoder_stages=adapted,
            selected_feature=selected,
            text_embedding=features.text_embedding,
            mask_embedding=features.mask_embedding,
            prompt_decoder_memory=features.prompt_decoder_memory,
            spacing=features.spacing,
            crop_bbox=features.crop_bbox,
            original_shape=features.original_shape,
            metadata=dict(features.metadata),
        )

    def forward(self, features: SharedVoxTellFeatures, logits: Tensor | None = None) -> dict[str, Tensor]:
        if logits is None:
            if self.decoder_forward is None:
                raise ValueError("AppearanceExpert requires logits or a decoder_forward module")
            logits = self.decoder_forward(self.adapt_features(features))
        return {"logits": logits, "prob": logits.float().sigmoid()}
