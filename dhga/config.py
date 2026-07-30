from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class DHGAConfig:
    voxtell_repo: str = "/data/zy/VoxTell_from_disk"
    model_dir: str = "/data/zy/VoxTell_from_disk/model"
    data_dir: str = "/data/zy/CT_MRI_DATA_3D/images"
    split_manifest: str = "/data/zy/DHGA/worst_zeroshot_split_p0/worst_zeroshot_split.json"
    sequences: str = "P0"
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
    val_label_dir: str = "/data/zy/CT_MRI_DATA_3D/labels/P0"
    label_values: list[int] = field(default_factory=lambda: [5])
    tensorboard_enabled: bool = True
    tensorboard_log_interval: int = 1
    tensorboard_image_interval: int = 50
    dhga_stage_b_method: str = "legacy"
    dhga_stage_b_anchor_candidate_patches: int = 24
    dhga_stage_b_include_background_patch: bool = True
    dhga_validation_interval_epochs: int = 2
    init_checkpoint: str = ""
    resume_checkpoint: str = ""
    dhga_ema_warmup_steps: int = 10
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
    dhga_appearance_anchor_weight: float = 0.25
    dhga_appearance_expansion_weight: float = 0.1
    dhga_cross_supervision_weight: float = 1.0
    dhga_cross_supervision_min_weight: float = 0.05
    dhga_weak_strong_weight: float = 0.5
    dhga_router_target_weight: float = 0.5
    dhga_router_normalization: str = "case_rank"
    dhga_text_layer_weights: list[float] = field(default_factory=list)
    dhga_text_layer_temperature: float = 0.1
    dhga_text_layer_foreground_support_threshold: float = 0.5
    dhga_text_layer_candidate_max_ratio: float = 0.1
    dhga_text_layer_candidate_alpha: float = 0.5
    dhga_text_layer_stability_threshold: float = 0.08
    dhga_text_layer_reliable_fg_threshold: float = 0.8
    dhga_text_layer_reliable_bg_threshold: float = 0.2
    dhga_text_layer_candidate_weight: float = 0.1
    dhga_geometry_enabled: bool = True
    dhga_search_radius_mm: float = 6.0
    dhga_surface_tolerance_mm: float = 1.0
    dhga_ray_step_mm: float = 1.0
    dhga_max_boundary_points: int = 4096
    dhga_boundary_chunk_size: int = 1024
    dhga_geometry_feature_layer: int = -1
    dhga_geometry_feature_channels: int = 8
    dhga_displacement_diffusion_mm: float = 2.0
    dhga_geometry_max_displacement_mm: float = 3.0
    dhga_geometry_boundary_band_mm: float = 6.0
    dhga_geometry_min_gate: float = 1e-4
    dhga_corruption_max_offset_mm: float = 3.0
    dhga_corruption_modes: list[str] = field(default_factory=lambda: ["inward", "outward"])
    dhga_boundary_recovery_weight: float = 1.0
    dhga_transport_equivariance_weight: float = 0.1
    dhga_prompt_ranking_weight: float = 0.0
    dhga_minimal_transport_weight: float = 0.01
    dhga_transport_smoothness_weight: float = 0.0
    dhga_debug_outputs: bool = False
    pred_threshold: float = 0.5

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.init_checkpoint and self.resume_checkpoint:
            raise ValueError("--init_checkpoint and --resume_checkpoint are mutually exclusive")
        if self.dhga_stage not in {"A", "B", "C", "D"}:
            raise ValueError("dhga_stage must be one of A, B, C, D")
        if self.epochs < 0 or self.steps_per_volume < 0:
            raise ValueError("epochs and steps_per_volume must be non-negative")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.tensorboard_log_interval <= 0 or self.tensorboard_image_interval <= 0:
            raise ValueError("tensorboard intervals must be positive")
        if self.dhga_stage_b_method not in {"legacy", "text_layer_ensemble"}:
            raise ValueError("dhga_stage_b_method must be legacy or text_layer_ensemble")
        if self.dhga_stage_b_anchor_candidate_patches <= 0:
            raise ValueError("dhga_stage_b_anchor_candidate_patches must be positive")
        if self.dhga_validation_interval_epochs <= 0:
            raise ValueError("dhga_validation_interval_epochs must be positive")
        if self.dhga_semantic_adapter_rank <= 0:
            raise ValueError("dhga_semantic_adapter_rank must be positive")
        if self.dhga_semantic_adapter_target not in {"cross", "self", "both"}:
            raise ValueError("dhga_semantic_adapter_target must be cross, self, or both")
        if not 0.0 <= self.dhga_appearance_feature_dropout < 1.0:
            raise ValueError("dhga_appearance_feature_dropout must be in [0, 1)")
        if not 0.0 < self.dhga_ema_decay < 1.0:
            raise ValueError("dhga_ema_decay must be in (0, 1)")
        if self.dhga_anchor_weight < 0 or self.dhga_appearance_anchor_weight < 0:
            raise ValueError("anchor weights must be non-negative")
        if self.dhga_appearance_expansion_weight < 0:
            raise ValueError("dhga_appearance_expansion_weight must be non-negative")
        if self.dhga_cross_supervision_weight < 0:
            raise ValueError("dhga_cross_supervision_weight must be non-negative")
        if not 0.0 <= self.dhga_cross_supervision_min_weight <= 1.0:
            raise ValueError("dhga_cross_supervision_min_weight must be in [0, 1]")
        if self.dhga_router_target_weight < 0:
            raise ValueError("dhga_router_target_weight must be non-negative")
        if self.dhga_router_normalization not in {"case_rank", "none"}:
            raise ValueError("dhga_router_normalization must be case_rank or none")
        if any(weight < 0 for weight in self.dhga_text_layer_weights):
            raise ValueError("dhga_text_layer_weights must be non-negative")
        if self.dhga_text_layer_weights and sum(self.dhga_text_layer_weights) <= 0:
            raise ValueError("dhga_text_layer_weights must contain a positive sum")
        if self.dhga_text_layer_temperature <= 0:
            raise ValueError("dhga_text_layer_temperature must be positive")
        for name in (
            "dhga_text_layer_foreground_support_threshold",
            "dhga_text_layer_candidate_max_ratio",
            "dhga_text_layer_candidate_alpha",
            "dhga_text_layer_stability_threshold",
            "dhga_text_layer_reliable_fg_threshold",
            "dhga_text_layer_reliable_bg_threshold",
            "dhga_text_layer_candidate_weight",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.dhga_text_layer_candidate_max_ratio > 1:
            raise ValueError("dhga_text_layer_candidate_max_ratio must be <= 1")
        if not 0 <= self.dhga_text_layer_reliable_bg_threshold < self.dhga_text_layer_reliable_fg_threshold <= 1:
            raise ValueError("text-layer reliable thresholds must satisfy 0 <= bg < fg <= 1")
        if self.dhga_search_radius_mm <= 0 or self.dhga_ray_step_mm <= 0:
            raise ValueError("ray radius and step must be positive")
        if self.dhga_surface_tolerance_mm <= 0:
            raise ValueError("surface tolerance must be positive")
        if self.dhga_surface_tolerance_mm > self.dhga_search_radius_mm:
            raise ValueError("surface tolerance must not exceed ray search radius")
        if self.dhga_geometry_feature_channels <= 0:
            raise ValueError("dhga_geometry_feature_channels must be positive")
        if self.dhga_displacement_diffusion_mm <= 0:
            raise ValueError("dhga_displacement_diffusion_mm must be positive")
        if self.dhga_geometry_max_displacement_mm <= 0:
            raise ValueError("dhga_geometry_max_displacement_mm must be positive")
        if self.dhga_geometry_boundary_band_mm <= 0:
            raise ValueError("dhga_geometry_boundary_band_mm must be positive")
        if self.dhga_geometry_min_gate < 0:
            raise ValueError("dhga_geometry_min_gate must be non-negative")
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


def parse_float_list(values: Iterable[str] | None, default: list[float]) -> list[float]:
    if values is None:
        return list(default)
    parsed: list[float] = []
    for value in values:
        parsed.extend(float(item) for item in str(value).replace(",", " ").split())
    return parsed
