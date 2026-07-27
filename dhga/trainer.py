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
from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .config import DHGAConfig
from .experts import AppearanceExpert, SemanticExpert
from .geometry import GeometryTransportHead, make_ray_offsets_mm, mask_to_sdf
from .geometry.boundary_corruption import make_local_boundary_corruption
from .geometry.ray_sampler import sample_along_normals
from .losses import boundary_recovery_loss, cross_supervision_loss, diagnostics_from_probs, minimal_transport_loss, router_fusion_loss, weighted_bce_prob
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
        self._set_stage_trainability()
        self.optimizer = None if config.dhga_stage == "A" else self._build_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        self.teacher = EMATeacher(self.model, config.dhga_ema_decay) if config.dhga_use_ema_teacher else None
        self.start_epoch = 0
        self.global_step = 0
        self._load_initial_or_resume()
        self._init_io()
        print(json.dumps(self.model.trainable_summary(), indent=2))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters for Stage {self.config.dhga_stage}")
        return torch.optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay, foreach=False)

    def _set_stage_trainability(self) -> None:
        for param in self.model.parameters():
            param.requires_grad_(False)
        stage = self.config.dhga_stage
        if stage == "B":
            for module in self.model.injected_lora.values():
                for name, param in module.named_parameters():
                    if not name.startswith("base."):
                        param.requires_grad_(True)
            for param in self.model.appearance_expert.parameters():
                param.requires_grad_(True)
            for param in self.model.router.parameters():
                param.requires_grad_(True)
        elif stage == "C":
            for param in self.model.geometry_head.parameters():
                param.requires_grad_(True)
            for param in self.model.ray_tokens.parameters():
                param.requires_grad_(True)
            for param in self.model.geometry_visual_proj.parameters():
                param.requires_grad_(True)
        elif stage == "D":
            for param in self.model.router.parameters():
                param.requires_grad_(True)
            for param in self.model.geometry_head.parameters():
                param.requires_grad_(True)
            for param in self.model.ray_tokens.parameters():
                param.requires_grad_(True)
            for param in self.model.geometry_visual_proj.parameters():
                param.requires_grad_(True)
        elif stage == "A":
            pass
        else:
            raise ValueError(f"Unsupported stage {stage}")

    def _load_initial_or_resume(self) -> None:
        if self.config.resume_checkpoint:
            payload = load_training_checkpoint(
                self.config.resume_checkpoint,
                self.model,
                self.optimizer,
                self.teacher if self.teacher is not None else None,
                self.scaler,
                load_training_state=True,
                expected_stage=self.config.dhga_stage,
            )
            self.start_epoch = int(payload.get("epoch", 0))
            self.global_step = int(payload.get("global_step", 0))
            return
        if self.config.init_checkpoint:
            load_training_checkpoint(
                self.config.init_checkpoint,
                self.model,
                load_training_state=False,
            )
            if self.teacher is not None:
                self.teacher.sync_from(self.model)
            self._set_stage_trainability()
            self.optimizer = None if self.config.dhga_stage == "A" else self._build_optimizer()

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

    def _load_volume(self, path: Path) -> tuple[Tensor, tuple[float, float, float]]:
        image, props = self.reader.read_images([str(path)])
        spacing = self._spacing_zyx_from_properties(props)
        image, _, _ = self.predictor.preprocess(image)
        image, _ = self.pad_nd_image(image, tuple(int(v) for v in self.predictor.patch_size), "constant", {"value": 0}, True, None)
        return image, spacing

    def _spacing_zyx_from_properties(self, props: dict) -> tuple[float, float, float]:
        spacing = props.get("spacing") or props.get("sitk_stuff", {}).get("spacing")
        if spacing is None:
            return (1.0, 1.0, 1.0)
        values = tuple(float(v) for v in spacing)
        if len(values) != 3:
            return (1.0, 1.0, 1.0)
        return values

    def fit(self) -> None:
        (self.save_dir / "dhga_config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        train_paths = self._load_train_paths()
        history = []
        self.model.train()
        for epoch in tqdm(range(self.start_epoch, self.config.epochs), desc=f"DHGA Stage {self.config.dhga_stage}", dynamic_ncols=True):
            random.shuffle(train_paths)
            for path in train_paths:
                volume, spacing = self._load_volume(path)
                patch_size = tuple(int(v) for v in self.predictor.patch_size)
                slicers = self.predictor._internal_get_sliding_window_slicers(volume.shape[1:])
                random.shuffle(slicers)
                if self.config.dhga_stage in {"C", "D"}:
                    slicers = self._teacher_boundary_guided_slicers(volume, spacing, slicers)
                if self.config.steps_per_volume > 0:
                    slicers = slicers[: self.config.steps_per_volume]
                for slicer in slicers:
                    patch = torch.clone(volume[slicer][None], memory_format=torch.contiguous_format).to(self.device)
                    if self.config.dhga_stage == "A":
                        with torch.no_grad():
                            loss, metrics = self.training_step(patch, spacing=spacing)
                    else:
                        with torch.autocast(self.device.type, enabled=self.config.amp and self.device.type == "cuda"):
                            loss, metrics = self.training_step(patch, spacing=spacing)
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.global_step += 1
                        if self.teacher is not None:
                            self.teacher.update(self.model)
                    record = {"epoch": epoch + 1, "case": str(path), **metrics}
                    history.append(record)
            save_training_checkpoint(
                self.save_dir / "checkpoint_last.pt",
                self.model,
                self.config,
                optimizer=self.optimizer,
                ema=self.teacher if self.teacher is not None else None,
                scaler=self.scaler,
                epoch=epoch + 1,
                global_step=self.global_step,
                metadata={"stage": self.config.dhga_stage},
            )
        save_training_checkpoint(
            self.save_dir / "checkpoint_final.pt",
            self.model,
            self.config,
            optimizer=self.optimizer,
            ema=self.teacher if self.teacher is not None else None,
            scaler=self.scaler,
            epoch=self.config.epochs,
            global_step=self.global_step,
            metadata={"stage": self.config.dhga_stage, "history": history[-256:]},
        )
        (self.save_dir / "history.json").write_text(json.dumps(history, indent=2))

    def training_step(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        stage = self.config.dhga_stage
        if stage == "A":
            return self._stage_a(patch, spacing)
        if stage == "B":
            return self._stage_b(patch, spacing)
        if stage == "C":
            recovery, minimal, metrics = self._stage_c(patch, spacing)
            loss = self._combine_stage_c_loss(recovery, minimal)
            metrics["loss"] = float(loss.detach().cpu())
            return loss, metrics
        if stage == "D":
            return self._stage_d(patch, spacing)
        raise ValueError(f"Unsupported stage {stage}")

    def _teacher_boundary_guided_slicers(self, volume, spacing: tuple[float, float, float], slicers: list) -> list:
        if not slicers or self.config.steps_per_volume <= 0:
            return slicers
        candidate_count = min(len(slicers), max(self.config.steps_per_volume * 4, self.config.steps_per_volume))
        candidates = slicers[:candidate_count]
        scored = []
        with torch.no_grad():
            for slicer in candidates:
                patch = torch.as_tensor(volume[slicer][None], device=self.device, dtype=torch.float32)
                out = self._teacher_forward(patch, spacing, run_geometry=False)
                sdf = mask_to_sdf(out.router.fused_prob.detach() >= self.config.pred_threshold, spacing)
                band = (sdf.abs() <= self.config.dhga_surface_tolerance_mm * 2.0).float()
                score = float((band * out.router.geometry_disagreement_weight.detach().clamp_min(0.05)).mean().cpu())
                scored.append((score, slicer))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [slicer for _, slicer in scored]
        selected.extend(slicer for slicer in slicers[candidate_count:] if slicer not in selected)
        return selected

    def _teacher_forward(self, patch: Tensor, spacing: tuple[float, float, float], run_geometry: bool = False):
        module = self.model
        was_training = module.training
        module.eval()
        with torch.no_grad():
            if self.teacher is not None and self.global_step >= self.config.dhga_ema_warmup_steps:
                with self.teacher.apply_to(module):
                    out = module(patch, spacing, run_geometry=run_geometry)
            else:
                out = module(patch, spacing, run_geometry=run_geometry)
        module.train(was_training)
        return out

    def _stage_a(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        out = self.model.baseline_forward(patch, spacing) if hasattr(self.model, "baseline_forward") else self.model(patch, spacing, run_geometry=False)
        loss = (out.semantic_prob - out.anchor_prob.detach()).abs().mean() * 0.0
        metrics = diagnostics_from_probs(out.semantic_prob, out.appearance_prob, out.router)
        metrics["dhga_stage_a_anchor_delta"] = float((out.semantic_prob - out.anchor_prob).abs().mean().detach().cpu())
        metrics["dhga_stage_a_forced_baseline"] = 1.0
        metrics["loss"] = float(loss.detach().cpu())
        return loss, metrics

    def _stage_b(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        weak, strong, _ = weak_strong_intensity_views(patch)
        teacher_out = self._teacher_forward(weak, spacing, run_geometry=False)
        strong_out = self.model(strong, spacing, run_geometry=False)
        stable_anchor = ((teacher_out.anchor_prob.detach() > 0.9) | (teacher_out.anchor_prob.detach() < 0.1)).float()
        anchor_loss = weighted_bce_prob(strong_out.semantic_prob, teacher_out.anchor_prob.detach(), stable_anchor)
        weak_strong = F.mse_loss(strong_out.semantic_prob, teacher_out.semantic_prob.detach()) + F.mse_loss(strong_out.appearance_prob, teacher_out.appearance_prob.detach())
        weight = teacher_out.router.cross_supervision_weight.detach()
        sem_from_teacher_app = weighted_bce_prob(strong_out.semantic_prob, teacher_out.appearance_prob.detach(), weight)
        app_from_teacher_sem = weighted_bce_prob(strong_out.appearance_prob, teacher_out.semantic_prob.detach(), weight)
        cross = 0.5 * (sem_from_teacher_app + app_from_teacher_sem)
        router_loss = router_fusion_loss(strong_out.router, 0.5 * (teacher_out.semantic_prob + teacher_out.appearance_prob), weight)
        loss = (
            self.config.dhga_anchor_weight * anchor_loss
            + self.config.dhga_weak_strong_weight * weak_strong
            + self.config.dhga_cross_supervision_weight * cross
            + 0.1 * router_loss
        )
        metrics = diagnostics_from_probs(strong_out.semantic_prob, strong_out.appearance_prob, teacher_out.router)
        metrics.update({
            "loss": float(loss.detach().cpu()),
            "dhga_anchor_loss": float(anchor_loss.detach().cpu()),
            "dhga_weak_strong_loss": float(weak_strong.detach().cpu()),
            "dhga_cross_supervision_loss": float(cross.detach().cpu()),
            "dhga_router_fusion_loss": float(router_loss.detach().cpu()),
        })
        return loss, metrics

    def _stage_c(
        self,
        patch: Tensor,
        spacing: tuple[float, float, float],
        teacher_out=None,
        student_out=None,
        teacher_sdf: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        reused_teacher = teacher_out is not None
        reused_student = student_out is not None
        if teacher_out is None:
            teacher_out = self._teacher_forward(patch, spacing, run_geometry=False)
        if student_out is None:
            student_out = self.model(patch, spacing, run_geometry=False)
        if teacher_sdf is None:
            teacher_sdf = mask_to_sdf(teacher_out.router.fused_prob.detach() >= self.config.pred_threshold, spacing)
        stable_band = (teacher_sdf.abs() <= self.config.dhga_surface_tolerance_mm * 3.0) * (teacher_out.router.cross_supervision_weight.detach() > 0)
        corrupted_sdf, recovery_target, choices = make_local_boundary_corruption(
            teacher_sdf,
            self.config.dhga_corruption_max_offset_mm,
            self.config.dhga_corruption_modes,
            stable_band=stable_band,
        )
        geometry = self.model.run_geometry(
            patch,
            student_out.semantic_prob,
            student_out.appearance_prob,
            student_out.router,
            self.model.text_embeddings,
            spacing,
            initial_sdf=corrupted_sdf,
            visual_feature=student_out.features.encoder_stages[self.model.geometry_feature_idx],
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
        return recovery, minimal, {
            "dhga_boundary_recovery_loss": float(recovery.detach().cpu()),
            "dhga_minimal_transport_loss": float(minimal.detach().cpu()),
            "dhga_inward_corruptions": float(choices[..., 0].mean().detach().cpu()),
            "dhga_outward_corruptions": float(choices[..., 1].mean().detach().cpu()),
            "dhga_zero_corruptions": float(choices[..., 2].mean().detach().cpu()),
            "dhga_valid_boundary_points": float(valid.float().mean().detach().cpu()),
            "dhga_reused_teacher_forward": float(reused_teacher),
            "dhga_reused_student_forward": float(reused_student),
        }

    def _combine_stage_c_loss(self, recovery: Tensor, minimal: Tensor) -> Tensor:
        return self.config.dhga_boundary_recovery_weight * recovery + self.config.dhga_minimal_transport_weight * minimal

    def _stage_d(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        teacher_out = self._teacher_forward(patch, spacing, run_geometry=False)
        out = self.model(patch, spacing, run_geometry=True)
        teacher_sdf = mask_to_sdf(teacher_out.router.fused_prob.detach() >= self.config.pred_threshold, spacing)
        router_target = self._stage_d_router_target_loss(out, teacher_out)
        displacement = out.geometry.get("sparse_displacement_mm", patch.new_zeros(1))
        minimal = minimal_transport_loss(displacement)
        recovery, recovery_minimal, recovery_metrics = self._stage_c(
            patch,
            spacing,
            teacher_out=teacher_out,
            student_out=out,
            teacher_sdf=teacher_sdf,
        )
        recovery_aux = self._combine_stage_c_loss(recovery, recovery_minimal)
        equiv = patch.new_zeros(())
        weak, strong, _ = weak_strong_intensity_views(patch)
        out_a = self.model(weak, spacing, run_geometry=True)
        out_b = self.model(strong, spacing, run_geometry=True)
        if out_a.geometry and out_b.geometry:
            dense_a = out_a.geometry["dense_displacement_mm"].detach()
            dense_b = out_b.geometry["dense_displacement_mm"]
            common = ((dense_a.abs() > 0) | (dense_b.abs() > 0)).float()
            equiv = ((dense_b - dense_a).square() * common).sum() / common.sum().clamp_min(1.0)
        ranking = self._prompt_ranking_loss(out) if self.config.dhga_prompt_ranking_weight > 0 else patch.new_zeros(())
        loss = (
            self.config.dhga_cross_supervision_weight * router_target
            + recovery_aux
            + self.config.dhga_transport_equivariance_weight * equiv
            + self.config.dhga_prompt_ranking_weight * ranking
            + self.config.dhga_minimal_transport_weight * minimal
        )
        metrics = diagnostics_from_probs(out.semantic_prob, out.appearance_prob, out.router)
        metrics.update({
            "loss": float(loss.detach().cpu()),
            "dhga_router_target_loss": float(router_target.detach().cpu()),
            "dhga_minimal_transport_loss": float(minimal.detach().cpu()),
            "dhga_boundary_recovery_loss": float(recovery.detach().cpu()),
            "dhga_boundary_recovery_aux_loss": float(recovery_aux.detach().cpu()),
            "dhga_recovery_minimal_loss": float(recovery_minimal.detach().cpu()),
            "dhga_reused_teacher_forward": recovery_metrics.get("dhga_reused_teacher_forward", 0.0),
            "dhga_reused_student_forward": recovery_metrics.get("dhga_reused_student_forward", 0.0),
            "dhga_transport_equivariance_loss": float(equiv.detach().cpu()),
            "dhga_prompt_ranking_loss": float(ranking.detach().cpu()),
            "dhga_prompt_ranking_enabled": float(self.config.dhga_prompt_ranking_weight > 0),
            "dhga_abs_displacement_mm": float(displacement.detach().abs().mean().cpu()),
        })
        return loss, metrics

    def _prompt_ranking_loss(self, out) -> Tensor:
        return out.semantic_prob.new_zeros(())

    def _stage_d_router_target_loss(self, student_out, teacher_out) -> Tensor:
        with torch.no_grad():
            teacher_disagreement = (teacher_out.semantic_prob - teacher_out.appearance_prob).abs()
            sem_reliability = 1.0 - (teacher_out.semantic_prob - teacher_out.anchor_prob).abs()
            app_reliability = 1.0 - (teacher_out.appearance_prob - teacher_out.anchor_prob).abs()
            geo_reliability = teacher_disagreement
            target = torch.cat([sem_reliability, app_reliability, geo_reliability], dim=1).clamp_min(0.0)
            target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weight = teacher_out.router.stable_foreground + teacher_out.router.stable_background + teacher_out.router.disagreement
            weight = weight.detach().clamp(0, 1)
        pred = torch.cat([student_out.router.w_sem, student_out.router.w_app, student_out.router.w_geo], dim=1)
        loss = (pred - target.detach()).square().sum(dim=1, keepdim=True)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)
