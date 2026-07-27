from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from .augment import weak_strong_intensity_views
from .checkpoint import save_dhga_checkpoint
from .config import DHGAConfig
from .experts import AppearanceExpert, SemanticExpert
from .geometry import GeometryTransportHead, make_ray_offsets_mm, mask_to_sdf
from .geometry.boundary_corruption import make_bidirectional_corruption
from .geometry.ray_sampler import sample_along_normals
from .losses import boundary_recovery_loss, cross_supervision_loss, diagnostics_from_probs, minimal_transport_loss, weighted_bce_prob
from .routing import DisagreementRouter
from .shared_voxtell import SharedEncoderOnce, trainable_parameter_summary
from .teacher import EMATeacher
from .voxtell_model import DHGAVoxTellModel, build_dhga_voxtell_model
from voxtell_sfda.adapter import load_split_manifest, random_slicer, set_seed


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


class DHGAStageTrainer:
    def __init__(self, config: DHGAConfig, prompts: list[str], save_dir: str | Path) -> None:
        config.validate()
        set_seed(config.seed)
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.predictor, self.prompts = build_dhga_voxtell_model(config, prompts)
        self.device = next(self.model.parameters()).device
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
            foreach=False,
        )
        self.teacher = EMATeacher(self.model, config.dhga_ema_decay) if config.dhga_use_ema_teacher else None
        self._init_io()
        print(json.dumps(self.model.trainable_summary(), indent=2))

    def _init_io(self) -> None:
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

        self.reader = NibabelIOWithReorient()
        self.pad_nd_image = pad_nd_image

    def _load_train_paths(self) -> list[Path]:
        if not self.config.split_manifest or not self.config.data_dir:
            raise ValueError("Real DHGA training requires --split_manifest and --data_dir")
        paths = load_split_manifest(
            Path(self.config.split_manifest),
            "train",
            Path(self.config.data_dir),
            self.config.sequences,
        )
        if self.config.max_cases > 0:
            paths = paths[: self.config.max_cases]
        return paths

    def _load_volume(self, path: Path) -> Tensor:
        image, _ = self.reader.read_images([str(path)])
        image, _, _ = self.predictor.preprocess(image)
        image, _ = self.pad_nd_image(image, tuple(int(v) for v in self.predictor.patch_size), "constant", {"value": 0}, True, None)
        return image

    def fit(self) -> None:
        (self.save_dir / "dhga_config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        train_paths = self._load_train_paths()
        history = []
        self.model.train()
        for epoch in tqdm(range(self.config.epochs), desc=f"DHGA Stage {self.config.dhga_stage}", dynamic_ncols=True):
            random.shuffle(train_paths)
            for path in train_paths:
                volume = self._load_volume(path)
                patch_size = tuple(int(v) for v in self.predictor.patch_size)
                slicers = self.predictor._internal_get_sliding_window_slicers(volume.shape[1:])
                random.shuffle(slicers)
                if self.config.steps_per_volume > 0:
                    slicers = slicers[: self.config.steps_per_volume]
                for slicer in slicers:
                    patch = torch.clone(volume[slicer][None], memory_format=torch.contiguous_format).to(self.device)
                    loss, metrics = self.training_step(patch, spacing=(1.0, 1.0, 1.0))
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
                    self.optimizer.step()
                    if self.teacher is not None:
                        self.teacher.update(self.model)
                    record = {"epoch": epoch + 1, "case": str(path), **metrics}
                    history.append(record)
        save_dhga_checkpoint(
            self.save_dir / "checkpoint_final.pt",
            {
                "dhga_model": self.model,
            },
            self.config,
            {"stage": self.config.dhga_stage, "history": history[-256:]},
        )
        (self.save_dir / "history.json").write_text(json.dumps(history, indent=2))

    def training_step(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        stage = self.config.dhga_stage
        if stage == "A":
            return self._stage_a(patch, spacing)
        if stage == "B":
            return self._stage_b(patch, spacing)
        if stage == "C":
            return self._stage_c(patch, spacing)
        if stage == "D":
            return self._stage_d(patch, spacing)
        raise ValueError(f"Unsupported stage {stage}")

    def _stage_a(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        out = self.model(patch, spacing, run_geometry=False)
        loss = (out.semantic_prob - out.anchor_prob.detach()).abs().mean() * 0.0
        metrics = diagnostics_from_probs(out.semantic_prob, out.appearance_prob, out.router)
        metrics["dhga_stage_a_anchor_delta"] = float((out.semantic_prob - out.anchor_prob).abs().mean().detach().cpu())
        metrics["loss"] = float(loss.detach().cpu())
        return loss, metrics

    def _stage_b(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        weak, strong, _ = weak_strong_intensity_views(patch)
        weak_out = self.model(weak, spacing, run_geometry=False)
        strong_out = self.model(strong, spacing, run_geometry=False)
        stable_anchor = ((weak_out.anchor_prob.detach() > 0.9) | (weak_out.anchor_prob.detach() < 0.1)).float()
        anchor_loss = weighted_bce_prob(weak_out.semantic_prob, weak_out.anchor_prob.detach(), stable_anchor)
        weak_strong = F.mse_loss(strong_out.semantic_prob, weak_out.semantic_prob.detach()) + F.mse_loss(strong_out.appearance_prob, weak_out.appearance_prob.detach())
        cross = cross_supervision_loss(weak_out.semantic_prob, weak_out.appearance_prob, weak_out.router)
        loss = (
            self.config.dhga_anchor_weight * anchor_loss
            + self.config.dhga_weak_strong_weight * weak_strong
            + self.config.dhga_cross_supervision_weight * cross
        )
        metrics = diagnostics_from_probs(weak_out.semantic_prob, weak_out.appearance_prob, weak_out.router)
        metrics.update({
            "loss": float(loss.detach().cpu()),
            "dhga_anchor_loss": float(anchor_loss.detach().cpu()),
            "dhga_weak_strong_loss": float(weak_strong.detach().cpu()),
            "dhga_cross_supervision_loss": float(cross.detach().cpu()),
        })
        return loss, metrics

    def _stage_c(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        out = self.model(patch, spacing, run_geometry=False)
        teacher_sdf = mask_to_sdf(out.router.fused_prob.detach() >= self.config.pred_threshold, spacing)
        corrupted_sdf, recovery_target, choices = make_bidirectional_corruption(
            teacher_sdf,
            self.config.dhga_corruption_max_offset_mm,
            self.config.dhga_corruption_modes,
        )
        geometry = self.model.run_geometry(
            patch,
            out.semantic_prob,
            out.appearance_prob,
            out.router,
            out.features.mask_embedding if out.features.mask_embedding is not None else self.model.text_embeddings,
            spacing,
            initial_sdf=corrupted_sdf,
        )
        sampled_target, valid_target = sample_along_normals(
            recovery_target,
            geometry["boundary_points_zyx"],
            geometry["boundary_normals_zyx"],
            torch.zeros(1, device=patch.device),
            spacing,
        )
        target = sampled_target[:, :, 0, 0]
        valid = valid_target[:, :, 0] & geometry["valid_boundary_points"]
        recovery = boundary_recovery_loss(geometry["sparse_displacement_mm"], target, valid.float())
        minimal = minimal_transport_loss(geometry["sparse_displacement_mm"], valid.float())
        loss = self.config.dhga_boundary_recovery_weight * recovery + self.config.dhga_minimal_transport_weight * minimal
        return loss, {
            "loss": float(loss.detach().cpu()),
            "dhga_boundary_recovery_loss": float(recovery.detach().cpu()),
            "dhga_minimal_transport_loss": float(minimal.detach().cpu()),
            "dhga_inward_corruptions": float((choices == 0).float().mean().detach().cpu()),
            "dhga_valid_boundary_points": float(valid.float().mean().detach().cpu()),
        }

    def _stage_d(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        out = self.model(patch, spacing, run_geometry=True)
        cross = cross_supervision_loss(out.semantic_prob, out.appearance_prob, out.router)
        displacement = out.geometry.get("sparse_displacement_mm", patch.new_zeros(1))
        minimal = minimal_transport_loss(displacement)
        loss = self.config.dhga_cross_supervision_weight * cross + self.config.dhga_minimal_transport_weight * minimal
        metrics = diagnostics_from_probs(out.semantic_prob, out.appearance_prob, out.router)
        metrics.update({
            "loss": float(loss.detach().cpu()),
            "dhga_cross_supervision_loss": float(cross.detach().cpu()),
            "dhga_minimal_transport_loss": float(minimal.detach().cpu()),
            "dhga_abs_displacement_mm": float(displacement.detach().abs().mean().cpu()),
        })
        return loss, metrics
