from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class DHGAConfig:
    voxtell_repo: str = "/data/zy/VoxTell_from_disk"
    model_dir: str = "/data/zy/VoxTell_from_disk/model"
    data_dir: str = ""
    split_manifest: str = ""
    sequences: str = ""
    prompt_templates: list[str] = field(default_factory=lambda: ["{}"])
    text_encoding_model: str = "Qwen/Qwen3-Embedding-4B"
    device: str = "cuda"
    epochs: int = 1
    steps_per_volume: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    amp: bool = True
    seed: int = 0
    max_cases: int = 0
    dhga_enabled: bool = True
    dhga_stage: str = "B"
    dhga_freeze_voxtell: bool = True
    dhga_semantic_adapter_rank: int = 4
    dhga_semantic_adapter_target: str = "cross"
    dhga_appearance_feature_layers: list[int] = field(default_factory=lambda: [-1, -2, -3])
    dhga_appearance_hidden_ratio: float = 0.25
    dhga_appearance_feature_dropout: float = 0.05
    dhga_use_ema_teacher: bool = True
    dhga_ema_decay: float = 0.99
    dhga_anchor_weight: float = 0.25
    dhga_cross_supervision_weight: float = 1.0
    dhga_weak_strong_weight: float = 0.5
    dhga_router_normalization: str = "case_rank"
    dhga_geometry_enabled: bool = True
    dhga_search_radius_mm: float = 6.0
    dhga_ray_step_mm: float = 1.0
    dhga_max_boundary_points: int = 4096
    dhga_boundary_chunk_size: int = 1024
    dhga_corruption_max_offset_mm: float = 3.0
    dhga_corruption_modes: list[str] = field(default_factory=lambda: ["inward", "outward"])
    dhga_boundary_recovery_weight: float = 1.0
    dhga_transport_equivariance_weight: float = 0.1
    dhga_prompt_ranking_weight: float = 0.05
    dhga_minimal_transport_weight: float = 0.01
    dhga_transport_smoothness_weight: float = 0.0
    dhga_debug_outputs: bool = False
    pred_threshold: float = 0.5

    def validate(self) -> None:
        if self.dhga_stage not in {"A", "B", "C", "D"}:
            raise ValueError("dhga_stage must be one of A, B, C, D")
        if self.epochs < 0 or self.steps_per_volume < 0:
            raise ValueError("epochs and steps_per_volume must be non-negative")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.dhga_semantic_adapter_rank <= 0:
            raise ValueError("dhga_semantic_adapter_rank must be positive")
        if self.dhga_semantic_adapter_target not in {"cross", "self", "both"}:
            raise ValueError("dhga_semantic_adapter_target must be cross, self, or both")
        if not 0.0 <= self.dhga_appearance_feature_dropout < 1.0:
            raise ValueError("dhga_appearance_feature_dropout must be in [0, 1)")
        if not 0.0 < self.dhga_ema_decay < 1.0:
            raise ValueError("dhga_ema_decay must be in (0, 1)")
        if self.dhga_router_normalization not in {"case_rank", "none"}:
            raise ValueError("dhga_router_normalization must be case_rank or none")
        if self.dhga_search_radius_mm <= 0 or self.dhga_ray_step_mm <= 0:
            raise ValueError("ray radius and step must be positive")
        if self.dhga_boundary_chunk_size <= 0 or self.dhga_max_boundary_points <= 0:
            raise ValueError("boundary chunk and max points must be positive")
        if self.dhga_corruption_max_offset_mm <= 0:
            raise ValueError("corruption max offset must be positive")
        if not 0.0 < self.pred_threshold < 1.0:
            raise ValueError("pred_threshold must be in (0, 1)")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: dict) -> "DHGAConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        config = cls(**{key: value for key, value in values.items() if key in known})
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "DHGAConfig":
        import json

        return cls.from_mapping(json.loads(Path(path).read_text()))


def parse_int_list(values: Iterable[str] | None, default: list[int]) -> list[int]:
    if values is None:
        return list(default)
    parsed: list[int] = []
    for value in values:
        parsed.extend(int(item) for item in str(value).replace(",", " ").split())
    return parsed
