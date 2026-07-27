from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import DHGAConfig
from .experts import AppearanceExpert, SemanticExpert
from .geometry import GeometryTransportHead, make_ray_offsets_mm
from .losses import cross_supervision_loss, diagnostics_from_probs
from .routing import DisagreementRouter
from .shared_voxtell import SharedEncoderOnce, trainable_parameter_summary


@dataclass
class DHGASmokeResult:
    loss: float
    diagnostics: dict[str, float]
    trainable_summary: dict
    shared_encoder_calls: int


class TinySegDecoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class DHGASmokeModel(nn.Module):
    """Random-tensor DHGA graph used for static/smoke validation only."""

    def __init__(self, config: DHGAConfig) -> None:
        super().__init__()
        self.config = config
        self.shared = SharedEncoderOnce(
            nn.Sequential(
                nn.Conv3d(1, 4, 3, padding=1),
                nn.GELU(),
                nn.Conv3d(4, 8, 3, padding=1),
            )
        )
        self.semantic_decoder = TinySegDecoder(8)
        self.semantic_expert = SemanticExpert()
        self.appearance_expert = AppearanceExpert([8], [0], config.dhga_appearance_hidden_ratio, config.dhga_appearance_feature_dropout)
        self.appearance_decoder = TinySegDecoder(8)
        self.router = DisagreementRouter(config.dhga_router_normalization)
        offsets = make_ray_offsets_mm(config.dhga_search_radius_mm, config.dhga_ray_step_mm)
        self.geometry = GeometryTransportHead(4, offsets)

    def forward(self, image: Tensor) -> tuple[Tensor, dict[str, float]]:
        features = self.shared(image)
        sem_logits = self.semantic_decoder(features.encoder_stages[-1])
        sem = self.semantic_expert(features, sem_logits)
        app_features = self.appearance_expert.adapt_features(features, self.config.dhga_appearance_feature_dropout)
        app_logits = self.appearance_decoder(app_features.encoder_stages[-1])
        app = self.appearance_expert(features, app_logits)
        router = self.router(sem["prob"], app["prob"])
        loss = cross_supervision_loss(sem["prob"], app["prob"], router)
        return loss, diagnostics_from_probs(sem["prob"], app["prob"], router)

    def module_summary(self) -> dict:
        return trainable_parameter_summary(
            {
                "dhga.semantic_expert": self.semantic_decoder,
                "dhga.appearance_expert": self.appearance_expert,
                "dhga.appearance_decoder": self.appearance_decoder,
                "dhga.router": self.router,
                "dhga.geometry": self.geometry,
            }
        )


def run_synthetic_smoke(config: DHGAConfig, device: str = "cpu") -> DHGASmokeResult:
    config.validate()
    torch.manual_seed(7)
    model = DHGASmokeModel(config).to(device)
    image = torch.randn(2, 1, 12, 14, 10, device=device)
    loss, diagnostics = model(image)
    loss.backward()
    return DHGASmokeResult(
        loss=float(loss.detach().cpu()),
        diagnostics=diagnostics,
        trainable_summary=model.module_summary(),
        shared_encoder_calls=model.shared.num_calls,
    )
