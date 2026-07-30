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
from .geometry import GeometryTransportHead, extract_boundary_points, make_ray_offsets_mm, mask_to_sdf
from .geometry.boundary_corruption import make_local_boundary_corruption
from .geometry.ray_sampler import sample_along_normals
from .losses import boundary_recovery_loss, cross_supervision_loss, diagnostics_from_probs, minimal_transport_loss, weighted_bce_prob
from .routing import DisagreementRouter
from .shared_voxtell import SharedEncoderOnce, trainable_parameter_summary
from .teacher import EMATeacher
from .text_layer_ensemble import text_layer_training_loss
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
        self._validate_stage_trainability()
        self.optimizer = None if config.dhga_stage == "A" else self._build_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        self.teacher = EMATeacher(self.model, config.dhga_ema_decay) if config.dhga_use_ema_teacher else None
        self.start_epoch = 0
        self.global_step = 0
        self.best_validation_score: float | None = None
        self.best_epoch: int | None = None
        self.stage_b_anchor_cache: dict[str, list[dict]] = {}
        self._load_initial_or_resume()
        self._init_io()
        self.writer = self._init_tensorboard()
        print(json.dumps(self.model.trainable_summary(), indent=2))
        print(json.dumps(self._trainable_group_summary(), indent=2))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters for Stage {self.config.dhga_stage}")
        return torch.optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay, foreach=False)

    def _stage_parameter_groups(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        lora_ids: set[int] = set()
        for module in getattr(self.model, "injected_lora", {}).values():
            for name, param in module.named_parameters():
                if not name.startswith("base."):
                    lora_ids.add(id(param))
        groups = {
            "semantic_lora": [],
            "appearance_expert": [],
            "router": [],
            "geometry": [],
            "other": [],
        }
        for name, param in self.model.named_parameters():
            if id(param) in lora_ids or ("lora" in name.lower() and ".base." not in name):
                groups["semantic_lora"].append((name, param))
            elif name.startswith("appearance_expert."):
                groups["appearance_expert"].append((name, param))
            elif name.startswith("router."):
                groups["router"].append((name, param))
            elif name.startswith(("geometry_head.", "ray_tokens.", "geometry_visual_proj.")):
                groups["geometry"].append((name, param))
            else:
                groups["other"].append((name, param))
        return groups

    def _trainable_group_summary(self) -> dict[str, dict[str, object]]:
        summary: dict[str, dict[str, object]] = {}
        for group, items in self._stage_parameter_groups().items():
            trainable = [(name, param) for name, param in items if param.requires_grad]
            summary[f"stage_{self.config.dhga_stage}.{group}"] = {
                "trainable_params": int(sum(param.numel() for _, param in trainable)),
                "trainable_tensors": len(trainable),
                "sample_names": [name for name, _ in trainable[:12]],
            }
        return summary

    def _validate_stage_trainability(self) -> None:
        groups = self._stage_parameter_groups()
        trainable = {group: [(name, param) for name, param in items if param.requires_grad] for group, items in groups.items()}
        stage = self.config.dhga_stage
        required = {
            "B": ("semantic_lora", "appearance_expert") if self.config.dhga_stage_b_method == "text_layer_ensemble" else ("semantic_lora", "appearance_expert", "router"),
            "C": ("geometry",),
            "D": ("router", "geometry"),
        }.get(stage, ())
        missing = [group for group in required if not trainable[group]]
        if missing:
            available = {group: [name for name, _ in items[:8]] for group, items in trainable.items() if items}
            raise RuntimeError(
                f"Stage {stage} has no trainable parameters for required group(s): {missing}. "
                f"Available trainable groups: {available}"
            )
        if stage == "B" and trainable["geometry"]:
            raise RuntimeError("Stage B unexpectedly enables geometry parameters")
        if stage in {"C", "D"} and (trainable["semantic_lora"] or trainable["appearance_expert"]):
            raise RuntimeError(f"Stage {stage} unexpectedly enables region-expert parameters")

    def _gradient_group_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for group, items in self._stage_parameter_groups().items():
            trainable_items = [(name, param) for name, param in items if param.requires_grad]
            total_sq = 0.0
            tensors_with_grad = 0
            tensors_nonzero_grad = 0
            for _, param in trainable_items:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                tensors_with_grad += 1
                grad_norm = grad.norm()
                total_sq += float(grad_norm.cpu()) ** 2
                if bool((grad.abs() > 0).any().cpu()):
                    tensors_nonzero_grad += 1
            metrics[f"dhga_grad_{group}_norm"] = total_sq ** 0.5
            metrics[f"dhga_grad_{group}_tensors"] = float(tensors_with_grad)
            metrics[f"dhga_grad_{group}_nonzero_tensors"] = float(tensors_nonzero_grad)
            metrics[f"dhga_grad_{group}_trainable_tensors"] = float(len(trainable_items))
            metrics[f"dhga_grad_{group}_trainable_params"] = float(sum(param.numel() for _, param in trainable_items))
        return metrics

    def _validate_required_gradient_flow(self, metrics: dict[str, float]) -> None:
        required = {
            "B": ("semantic_lora", "appearance_expert") if self.config.dhga_stage_b_method == "text_layer_ensemble" else ("semantic_lora", "appearance_expert", "router"),
            "C": ("geometry",),
            "D": ("router", "geometry"),
        }.get(self.config.dhga_stage, ())
        missing = [
            group
            for group in required
            if metrics.get(f"dhga_grad_{group}_tensors", 0.0) <= 0.0
        ]
        if missing:
            debug = {
                group: {
                    "trainable_tensors": metrics.get(f"dhga_grad_{group}_trainable_tensors", 0.0),
                    "grad_tensors": metrics.get(f"dhga_grad_{group}_tensors", 0.0),
                    "nonzero_grad_tensors": metrics.get(f"dhga_grad_{group}_nonzero_tensors", 0.0),
                    "grad_norm": metrics.get(f"dhga_grad_{group}_norm", 0.0),
                }
                for group in missing
            }
            debug["autograd"] = {
                key: value
                for key, value in metrics.items()
                if key.startswith("dhga_debug_")
            }
            raise RuntimeError(
                f"Stage {self.config.dhga_stage} loss is not connected to required trainable group(s): {debug}"
            )

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
            if self.config.dhga_stage_b_method == "legacy":
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
            metadata = payload.get("metadata", {})
            if isinstance(metadata.get("best_score"), (int, float)):
                self.best_validation_score = float(metadata["best_score"])
            if isinstance(metadata.get("best_epoch"), (int, float)):
                self.best_epoch = int(metadata["best_epoch"])
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
            self._validate_stage_trainability()
            self.optimizer = None if self.config.dhga_stage == "A" else self._build_optimizer()

    def _init_io(self) -> None:
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

        self.reader = NibabelIOWithReorient()
        self.pad_nd_image = pad_nd_image

    def _init_tensorboard(self):
        if not self.config.tensorboard_enabled:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            print(f"TensorBoard disabled: {exc}")
            return None
        log_dir = self.save_dir / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_text("dhga/config", json.dumps(self.config.to_dict(), indent=2), 0)
        return writer

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
        try:
            for epoch in tqdm(range(self.start_epoch, self.config.epochs), desc=f"DHGA Stage {self.config.dhga_stage}", dynamic_ncols=True):
                random.shuffle(train_paths)
                if self.config.dhga_stage == "B" and self.config.dhga_stage_b_method == "legacy":
                    epoch_total = len(train_paths) * self._stage_b_patches_per_case()
                else:
                    epoch_total = len(train_paths) * self.config.steps_per_volume if self.config.steps_per_volume > 0 else None
                epoch_records = []
                with tqdm(total=epoch_total, desc=f"epoch {epoch + 1}/{self.config.epochs}", dynamic_ncols=True, leave=False) as patch_bar:
                    for path in train_paths:
                        volume, spacing = self._load_volume(path)
                        patch_size = tuple(int(v) for v in self.predictor.patch_size)
                        slicers = self.predictor._internal_get_sliding_window_slicers(volume.shape[1:])
                        random.shuffle(slicers)
                        if self.config.dhga_stage == "B" and self.config.dhga_stage_b_method == "legacy":
                            selected_slicers = self._stage_b_anchor_guided_slicers(path, volume, spacing, slicers)
                        else:
                            if self.config.dhga_stage in {"C", "D"}:
                                slicers = self._teacher_boundary_guided_slicers(volume, spacing, slicers)
                            if self.config.steps_per_volume > 0:
                                slicers = slicers[: self.config.steps_per_volume]
                            selected_slicers = [(slicer, "default") for slicer in slicers]
                        for slicer, patch_kind in selected_slicers:
                            patch = torch.clone(volume[slicer][None], memory_format=torch.contiguous_format).to(self.device)
                            if self.config.dhga_stage == "A":
                                with torch.no_grad():
                                    loss, metrics = self.training_step(patch, spacing=spacing)
                            else:
                                self._set_stage_trainability()
                                with torch.autocast(self.device.type, enabled=self.config.amp and self.device.type == "cuda"):
                                    loss, metrics = self.training_step(patch, spacing=spacing)
                                self.optimizer.zero_grad(set_to_none=True)
                                self.scaler.scale(loss).backward()
                                self.scaler.unscale_(self.optimizer)
                                metrics.update(self._gradient_group_metrics())
                                self._validate_required_gradient_flow(metrics)
                                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.global_step += 1
                                if self.teacher is not None:
                                    self.teacher.update(self.model)
                            record = {"epoch": epoch + 1, "case": str(path), **metrics}
                            record["patch_kind"] = patch_kind
                            history.append(record)
                            epoch_records.append(record)
                            log_step = self.global_step if self.config.dhga_stage != "A" else len(history)
                            self._log_tensorboard_scalars(metrics, log_step)
                            self._maybe_log_visual_masks(patch, spacing, log_step)
                            patch_bar.set_postfix({"loss": f"{float(metrics.get('loss', 0.0)):.4f}", "kind": patch_kind, "case": Path(path).name[:18]})
                            patch_bar.update(1)
                epoch_metrics = self._log_epoch_metrics(epoch + 1, epoch_records)
                if self.config.dhga_stage in {"B", "C"} and (epoch + 1) % self.config.dhga_validation_interval_epochs == 0:
                    self._run_periodic_test_evaluation(epoch + 1, epoch_metrics)
        finally:
            if self.writer is not None:
                self.writer.flush()
        save_training_checkpoint(
            self.save_dir / "checkpoint_final.pt",
            self.model,
            self.config,
            optimizer=self.optimizer,
            ema=self.teacher if self.teacher is not None else None,
            scaler=self.scaler,
            epoch=self.config.epochs,
            global_step=self.global_step,
            metadata={
                "stage": self.config.dhga_stage,
                "history": history[-256:],
                "best_metric": "mean_fused_dice" if self.config.dhga_stage == "B" else None,
                "best_score": self.best_validation_score,
                "best_epoch": self.best_epoch,
            },
        )
        (self.save_dir / "history.json").write_text(json.dumps(history, indent=2))
        if self.writer is not None:
            self.writer.close()

    def _log_tensorboard_scalars(self, metrics: dict[str, float], step: int) -> None:
        if self.writer is None or step % self.config.tensorboard_log_interval != 0:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", float(value), step)
        if self.optimizer is not None:
            self.writer.add_scalar("train/lr", float(self.optimizer.param_groups[0]["lr"]), step)

    def _maybe_log_visual_masks(self, patch: Tensor, spacing: tuple[float, float, float], step: int) -> None:
        if self.writer is None or step <= 0:
            return
        if step != 1 and step % self.config.tensorboard_image_interval != 0:
            return
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out = self.model(patch[:1], spacing, run_geometry=False)
        self.model.train(was_training)
        image = patch[:1, :1].detach().float()
        sem_mask = (out.semantic_prob[:1] >= self.config.pred_threshold).float()
        app_mask = (out.appearance_prob[:1] >= self.config.pred_threshold).float()
        center = image.shape[-3] // 2
        panel = torch.cat(
            [
                self._normalize_image_slice(image[0, 0, center]),
                sem_mask[0, 0, center].detach().cpu(),
                app_mask[0, 0, center].detach().cpu(),
            ],
            dim=1,
        ).unsqueeze(0)
        self.writer.add_image("masks/image_semantic_appearance_center_z", panel, step)

    def _normalize_image_slice(self, image_slice: Tensor) -> Tensor:
        x = image_slice.detach().float().cpu()
        lo = torch.quantile(x.flatten(), 0.01)
        hi = torch.quantile(x.flatten(), 0.99)
        return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)

    def _stage_b_patches_per_case(self) -> int:
        if self.config.steps_per_volume > 0:
            return max(1, int(self.config.steps_per_volume))
        return 5 if self.config.dhga_stage_b_include_background_patch else 4

    def _stage_b_patch_kind_counts(self, total: int) -> dict[str, int]:
        total = max(0, int(total))
        if total == 0:
            return {"foreground": 0, "boundary": 0, "background": 0}
        if self.config.dhga_stage_b_include_background_patch:
            if total < 5:
                counts = {"foreground": 0, "boundary": 0, "background": 0}
                priority = ("foreground", "boundary", "background", "foreground")
                for idx in range(total):
                    counts[priority[idx]] += 1
                return counts
            counts = {
                "foreground": (total * 3) // 5,
                "boundary": total // 5,
                "background": total // 5,
            }
            remainder = total - sum(counts.values())
            for kind in ("foreground", "boundary", "background"):
                if remainder <= 0:
                    break
                counts[kind] += 1
                remainder -= 1
            return counts
        if total < 4:
            return {"foreground": total, "boundary": 0, "background": 0}
        counts = {
            "foreground": (total * 3) // 4,
            "boundary": total // 4,
            "background": 0,
        }
        remainder = total - counts["foreground"] - counts["boundary"]
        for kind in ("foreground", "boundary"):
            if remainder <= 0:
                break
            counts[kind] += 1
            remainder -= 1
        return counts

    def _stage_b_anchor_guided_slicers(self, path: Path | str, volume, spacing: tuple[float, float, float], slicers: list) -> list[tuple[tuple, str]]:
        if not slicers:
            return []
        cache_key = str(path)
        if cache_key not in self.stage_b_anchor_cache:
            candidate_count = min(len(slicers), max(self.config.dhga_stage_b_anchor_candidate_patches, self._stage_b_patches_per_case()))
            candidates = slicers[:candidate_count]
            scored = []
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                for candidate_idx, slicer in enumerate(candidates):
                    patch = torch.as_tensor(volume[slicer][None], device=self.device, dtype=torch.float32)
                    out = self.model.baseline_forward(patch, spacing) if hasattr(self.model, "baseline_forward") else self.model(patch, spacing, run_geometry=False)
                    anchor = out.anchor_prob.detach().float()
                    body_ratio = float((patch[:, :1].detach().float().abs() > 1e-3).float().mean().cpu())
                    fg_score = float(anchor.mean().cpu())
                    boundary_score = float((1.0 - (anchor - 0.5).abs() * 2.0).clamp(0, 1).mean().cpu())
                    scored.append({
                        "slicer": slicer,
                        "candidate_idx": candidate_idx,
                        "fg": fg_score,
                        "boundary": boundary_score,
                        "bg": (1.0 - fg_score) * body_ratio,
                        "body": body_ratio,
                    })
            self.model.train(was_training)
            self.stage_b_anchor_cache[cache_key] = scored
        scored = self.stage_b_anchor_cache[cache_key]
        selected: list[tuple[tuple, str]] = []
        used: set[int] = set()
        target_counts = self._stage_b_patch_kind_counts(self._stage_b_patches_per_case())

        def add_random_from_top(score_key: str, kind: str, body_min: float = 0.0) -> None:
            ranked = [
                (idx, item)
                for idx, item in sorted(enumerate(scored), key=lambda pair: pair[1][score_key], reverse=True)
                if idx not in used and item.get("body", 0.0) >= body_min
            ]
            if not ranked:
                ranked = [
                    (idx, item)
                    for idx, item in sorted(enumerate(scored), key=lambda pair: pair[1][score_key], reverse=True)
                    if idx not in used
                ]
            if not ranked:
                return
            pool_size = min(len(ranked), max(2, min(8, max(1, len(ranked) // 4))))
            idx, item = random.choice(ranked[:pool_size])
            selected.append((item["slicer"], kind))
            used.add(idx)

        def add_best_unused(kind: str) -> None:
            for idx, item in enumerate(scored):
                if idx not in used:
                    selected.append((item["slicer"], kind))
                    used.add(idx)
                    return

        for _ in range(target_counts["foreground"]):
            add_random_from_top("fg", "foreground")
        for _ in range(target_counts["boundary"]):
            add_random_from_top("boundary", "boundary")
        for _ in range(target_counts["background"]):
            add_random_from_top("bg", "background", body_min=0.02)
        while len(selected) < self._stage_b_patches_per_case():
            before = len(selected)
            add_best_unused("fallback")
            if len(selected) == before:
                break
        return selected

    def _log_epoch_metrics(self, epoch: int, records: list[dict]) -> dict[str, float]:
        numeric: dict[str, list[float]] = {}
        for record in records:
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    numeric.setdefault(key, []).append(float(value))
        metrics: dict[str, float] = {}
        for key, values in numeric.items():
            values = sorted(values)
            count = len(values)
            if count == 0:
                continue
            p90_idx = min(count - 1, int(0.9 * (count - 1)))
            mid = count // 2
            median = values[mid] if count % 2 else 0.5 * (values[mid - 1] + values[mid])
            metrics[f"epoch_{key}_mean"] = sum(values) / count
            metrics[f"epoch_{key}_median"] = median
            metrics[f"epoch_{key}_p90"] = values[p90_idx]
            metrics[f"epoch_{key}_max"] = values[-1]
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f"epoch/{key}", value, epoch)
        return metrics

    def _run_periodic_test_evaluation(self, epoch: int, epoch_metrics: dict[str, float] | None = None) -> None:
        label_dir = Path(self.config.val_label_dir)
        if not self.config.val_label_dir or not label_dir.exists():
            print(f"Skip test evaluation at epoch {epoch}: val_label_dir not found: {self.config.val_label_dir}")
            return
        from .config import DHGAConfig
        from .evaluation import DHGAEvaluator

        values = self.config.to_dict()
        values["init_checkpoint"] = ""
        values["resume_checkpoint"] = ""
        if self.config.dhga_stage == "B":
            values["dhga_geometry_enabled"] = False
        eval_config = DHGAConfig.from_mapping(values)
        eval_dir = self.save_dir / f"eval_epoch_{epoch:04d}"
        was_training = self.model.training
        evaluator = DHGAEvaluator(
            eval_config,
            self.prompts,
            eval_dir,
            self.config.val_label_dir,
            self.config.label_values,
            model=self.model,
            predictor=self.predictor,
        )
        try:
            metrics = evaluator.evaluate_split("test", 0)
        finally:
            self.model.train(was_training)
        if self.writer is not None:
            for key, value in metrics.items():
                if key != "rows" and isinstance(value, (int, float)) and value is not None:
                    self.writer.add_scalar(f"test/{key}", float(value), epoch)
        score = metrics.get("mean_fused_dice")
        if not isinstance(score, (int, float)):
            score = metrics.get("mean_dice")
        if self.config.dhga_stage == "B" and isinstance(score, (int, float)):
            save_training_checkpoint(
                self.save_dir / "last_stage_b.pt",
                self.model,
                self.config,
                optimizer=self.optimizer,
                ema=self.teacher if self.teacher is not None else None,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                metadata={
                    "stage": self.config.dhga_stage,
                    "best_metric": "mean_fused_dice",
                    "score": float(score),
                    "best_score": self.best_validation_score,
                    "best_epoch": self.best_epoch,
                    "epoch_metrics": epoch_metrics or {},
                    "test_metrics": {k: v for k, v in metrics.items() if k != "rows"},
                },
            )
        if self.config.dhga_stage == "B" and isinstance(score, (int, float)) and (self.best_validation_score is None or float(score) > self.best_validation_score):
            self.best_validation_score = float(score)
            self.best_epoch = int(epoch)
            save_training_checkpoint(
                self.save_dir / "best_stage_b.pt",
                self.model,
                self.config,
                optimizer=self.optimizer,
                ema=self.teacher if self.teacher is not None else None,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                metadata={
                    "stage": self.config.dhga_stage,
                    "best_metric": "mean_fused_dice",
                    "best_score": self.best_validation_score,
                    "best_epoch": self.best_epoch,
                    "epoch_metrics": epoch_metrics or {},
                    "test_metrics": {k: v for k, v in metrics.items() if k != "rows"},
                },
            )

    def training_step(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        stage = self.config.dhga_stage
        if stage == "A":
            return self._stage_a(patch, spacing)
        if stage == "B":
            if self.config.dhga_stage_b_method == "text_layer_ensemble":
                return self._stage_b_text_layer_ensemble(patch, spacing)
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

    def _stage_b_text_layer_ensemble(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        weak, strong, _ = weak_strong_intensity_views(patch)
        before_encoder_calls = int(getattr(self.model, "encoder_calls", 0))
        with torch.autocast(self.device.type, enabled=False):
            teacher_out = self._teacher_forward(weak.float(), spacing, run_geometry=False)
        out = self.model.forward_text_layer_ensemble(strong, spacing, run_geometry=False) if hasattr(self.model, "forward_text_layer_ensemble") else self.model(strong, spacing, run_geometry=False)
        if teacher_out.layer_ensemble is None:
            raise RuntimeError("text_layer_ensemble Stage B requires teacher layer_ensemble outputs")
        loss, metrics = text_layer_training_loss(out, self.config, teacher_out.layer_ensemble)
        ens = out.layer_ensemble or {}
        teacher_ens = teacher_out.layer_ensemble or {}
        sem_layers = ens.get("semantic_layer_probs")
        app_layers = ens.get("appearance_layer_probs")
        if isinstance(sem_layers, Tensor):
            for idx in range(sem_layers.shape[0]):
                metrics[f"dhga_text_layer_semantic_layer_{idx}_fg_ratio"] = float((sem_layers[idx] >= self.config.pred_threshold).float().mean().detach().cpu())
        if isinstance(app_layers, Tensor):
            for idx in range(app_layers.shape[0]):
                metrics[f"dhga_text_layer_appearance_layer_{idx}_fg_ratio"] = float((app_layers[idx] >= self.config.pred_threshold).float().mean().detach().cpu())
        for prefix, tensor in (("u_sem", ens.get("semantic_u_layer")), ("u_app", ens.get("appearance_u_layer"))):
            if isinstance(tensor, Tensor):
                values = tensor.detach().float().flatten()
                metrics[f"dhga_text_layer_{prefix}_mean"] = float(values.mean().cpu()) if values.numel() else 0.0
                metrics[f"dhga_text_layer_{prefix}_p90"] = float(torch.quantile(values, 0.9).cpu()) if values.numel() else 0.0
        for key in ("semantic_p_last", "appearance_p_last", "semantic_p_mean", "appearance_p_mean", "p_base", "p_final"):
            value = ens.get(key)
            if isinstance(value, Tensor):
                metrics[f"dhga_text_layer_{key}_pred_volume"] = float((value >= self.config.pred_threshold).float().mean().detach().cpu())
        for key in ("reliable_fg", "reliable_bg", "candidate_fg", "p_final"):
            value = teacher_ens.get(key)
            if isinstance(value, Tensor):
                if key == "p_final":
                    metrics["dhga_text_layer_teacher_target_volume"] = float((value >= self.config.pred_threshold).float().mean().detach().cpu())
                else:
                    metrics[f"dhga_text_layer_teacher_{key}_ratio"] = float((value > 0.5).float().mean().detach().cpu())
        metrics["dhga_text_layer_base_to_enhanced_abs_delta"] = float((ens["p_final"] - ens["p_base"]).abs().mean().detach().cpu()) if "p_final" in ens and "p_base" in ens else 0.0
        metrics["dhga_text_layer_encoder_calls_per_forward"] = float(int(getattr(self.model, "encoder_calls", 0)) - before_encoder_calls)
        metrics["dhga_text_layer_teacher_student_forwards"] = 2.0
        metrics["dhga_text_layer_method_enabled"] = 1.0
        metrics["dhga_text_layer_legacy_anchor_loss_active"] = 0.0
        metrics["dhga_text_layer_legacy_router_loss_active"] = 0.0
        metrics["loss"] = float(loss.detach().cpu())
        return loss, metrics

    def _stage_b(self, patch: Tensor, spacing: tuple[float, float, float]) -> tuple[Tensor, dict[str, float]]:
        weak, strong, _ = weak_strong_intensity_views(patch)
        with torch.autocast(self.device.type, enabled=False): # 不使用混合精度
            teacher_out = self._teacher_forward(weak.float(), spacing, run_geometry=False)
        strong_out = self.model(strong, spacing, run_geometry=False)
        stable_anchor = ((teacher_out.anchor_prob.detach() > 0.9) | (teacher_out.anchor_prob.detach() < 0.1)).float()
        semantic_anchor = weighted_bce_prob(strong_out.semantic_prob, teacher_out.anchor_prob.detach(), stable_anchor)
        appearance_anchor = weighted_bce_prob(strong_out.appearance_prob, teacher_out.anchor_prob.detach(), stable_anchor)
        anchor_loss = semantic_anchor + self.config.dhga_appearance_anchor_weight * appearance_anchor
        stable_background = (teacher_out.router.stable_background.detach() * (teacher_out.anchor_prob.detach() < 0.1).float()).clamp(0, 1)
        appearance_stable_bg = weighted_bce_prob(strong_out.appearance_prob, torch.zeros_like(strong_out.appearance_prob), stable_background)
        appearance_volume_excess = (
            strong_out.appearance_prob.float().flatten(1).mean(dim=1)
            - teacher_out.anchor_prob.detach().float().flatten(1).mean(dim=1)
        ).clamp_min(0.0).mean()
        appearance_expansion = appearance_volume_excess + 0.5 * appearance_stable_bg
        weak_strong = F.mse_loss(strong_out.semantic_prob, teacher_out.semantic_prob.detach()) + F.mse_loss(strong_out.appearance_prob, teacher_out.appearance_prob.detach()) # 同专家一致性
        cross_weight = teacher_out.router.cross_supervision_weight.detach().clamp_min(self.config.dhga_cross_supervision_min_weight)
        sem_from_teacher_app = weighted_bce_prob(strong_out.semantic_prob, teacher_out.appearance_prob.detach(), cross_weight) # Semantic学习teacher Appearance
        app_from_teacher_sem = weighted_bce_prob(strong_out.appearance_prob, teacher_out.semantic_prob.detach(), cross_weight) # Appearance学习Teacher Semantic
        cross = 0.5 * (sem_from_teacher_app + app_from_teacher_sem)
        router_loss = self._router_target_loss(strong_out, teacher_out, min_weight=0.05)
        loss = (
            self.config.dhga_anchor_weight * anchor_loss
            + self.config.dhga_appearance_expansion_weight * appearance_expansion
            + self.config.dhga_weak_strong_weight * weak_strong
            + self.config.dhga_cross_supervision_weight * cross
            + self.config.dhga_router_target_weight * router_loss
        )
        student_sem = strong_out.semantic_prob.detach().float()
        student_app = strong_out.appearance_prob.detach().float()
        sem_centered = student_sem - student_sem.mean()
        app_centered = student_app - student_app.mean()
        student_corr = (sem_centered * app_centered).mean() / (sem_centered.square().mean().sqrt() * app_centered.square().mean().sqrt()).clamp_min(1e-6)
        student_disagreement = strong_out.router.disagreement.detach().float()
        teacher_disagreement = teacher_out.router.disagreement.detach().float()
        student_high_disagreement = student_disagreement > 0.5
        teacher_high_disagreement = teacher_disagreement > 0.5
        student_w_geo = strong_out.router.w_geo.detach().float()
        teacher_w_geo = teacher_out.router.w_geo.detach().float()
        metrics = {
            "dhga_debug_loss_requires_grad": float(loss.requires_grad),
            "dhga_debug_semantic_prob_requires_grad": float(strong_out.semantic_prob.requires_grad),
            "dhga_debug_appearance_prob_requires_grad": float(strong_out.appearance_prob.requires_grad),
            "dhga_debug_router_w_sem_requires_grad": float(strong_out.router.w_sem.requires_grad),
            "dhga_debug_router_w_geo_requires_grad": float(strong_out.router.w_geo.requires_grad),
            "dhga_student_sem_fg_prob": float(student_sem.mean().cpu()),
            "dhga_student_app_fg_prob": float(student_app.mean().cpu()),
            "dhga_student_expert_corr": float(student_corr.cpu()),
            "dhga_student_w_geo_mean": float(student_w_geo.mean().cpu()),
            "dhga_student_w_geo_disagreement_mean": float((student_w_geo * student_disagreement).sum().cpu() / student_disagreement.sum().clamp_min(1e-6).cpu()),
            "dhga_student_w_geo_high_disagreement_mean": float(student_w_geo[student_high_disagreement].mean().cpu()) if bool(student_high_disagreement.any().cpu()) else 0.0,
            "dhga_teacher_disagreement_ratio": float((teacher_out.router.disagreement > 0.5).float().mean().detach().cpu()),
            "dhga_teacher_stable_fg_ratio": float((teacher_out.router.stable_foreground > 0.5).float().mean().detach().cpu()),
            "dhga_teacher_stable_bg_ratio": float((teacher_out.router.stable_background > 0.5).float().mean().detach().cpu()),
            "dhga_teacher_w_geo_mean": float(teacher_w_geo.mean().cpu()),
            "dhga_teacher_w_geo_disagreement_mean": float((teacher_w_geo * teacher_disagreement).sum().cpu() / teacher_disagreement.sum().clamp_min(1e-6).cpu()),
            "dhga_teacher_w_geo_high_disagreement_mean": float(teacher_w_geo[teacher_high_disagreement].mean().cpu()) if bool(teacher_high_disagreement.any().cpu()) else 0.0,
            "dhga_teacher_cross_supervision_weight": float(teacher_out.router.cross_supervision_weight.detach().mean().cpu()),
            "dhga_cross_supervision_effective_weight": float(cross_weight.detach().mean().cpu()),
            "dhga_cross_supervision_min_effective_weight": float(cross_weight.detach().min().cpu()),
            "dhga_router_supervision_weight": float(self._router_supervision_weight(teacher_out, 0.05).detach().mean().cpu()),
            "dhga_router_target_weight": float(self.config.dhga_router_target_weight),
        }
        metrics.update({
            "loss": float(loss.detach().cpu()),
            "dhga_anchor_loss": float(anchor_loss.detach().cpu()),
            "dhga_semantic_anchor_loss": float(semantic_anchor.detach().cpu()),
            "dhga_appearance_anchor_loss": float(appearance_anchor.detach().cpu()),
            "dhga_appearance_expansion_loss": float(appearance_expansion.detach().cpu()),
            "dhga_appearance_volume_excess_loss": float(appearance_volume_excess.detach().cpu()),
            "dhga_appearance_stable_background_loss": float(appearance_stable_bg.detach().cpu()),
            "dhga_weak_strong_loss": float(weak_strong.detach().cpu()),
            "dhga_cross_supervision_loss": float(cross.detach().cpu()),
            "dhga_router_target_loss": float(router_loss.detach().cpu()),
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
            candidate_score=student_out.layer_ensemble.get("candidate_score").detach() if student_out.layer_ensemble else None,
            candidate_fg=student_out.layer_ensemble.get("candidate_fg").detach() if student_out.layer_ensemble else None,
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
        boundary_points, boundary_normals, valid_boundary = extract_boundary_points(
            teacher_sdf,
            self.config.dhga_surface_tolerance_mm,
            self.config.dhga_max_boundary_points,
            spacing,
            sampling_weight=teacher_out.router.geometry_disagreement_weight,
        )
        out_a = self.model(weak, spacing, run_geometry=False)
        out_b = self.model(strong, spacing, run_geometry=False)
        geometry_a = self.model.run_geometry(
            weak,
            out_a.semantic_prob,
            out_a.appearance_prob,
            out_a.router,
            self.model.text_embeddings,
            spacing,
            initial_sdf=teacher_sdf,
            visual_feature=out_a.features.encoder_stages[self.model.geometry_feature_idx],
            boundary_points=boundary_points,
            boundary_normals=boundary_normals,
            valid_boundary_points=valid_boundary,
        )
        geometry_b = self.model.run_geometry(
            strong,
            out_b.semantic_prob,
            out_b.appearance_prob,
            out_b.router,
            self.model.text_embeddings,
            spacing,
            initial_sdf=teacher_sdf,
            visual_feature=out_b.features.encoder_stages[self.model.geometry_feature_idx],
            boundary_points=boundary_points,
            boundary_normals=boundary_normals,
            valid_boundary_points=valid_boundary,
        )
        if geometry_a and geometry_b:
            dense_a = geometry_a["dense_displacement_mm"].detach()
            dense_b = geometry_b["dense_displacement_mm"]
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
            "dhga_shared_equivariance_boundary_points": float(valid_boundary.float().mean().detach().cpu()),
            "dhga_transport_equivariance_loss": float(equiv.detach().cpu()),
            "dhga_prompt_ranking_loss": float(ranking.detach().cpu()),
            "dhga_prompt_ranking_enabled": float(self.config.dhga_prompt_ranking_weight > 0),
            "dhga_abs_displacement_mm": float(displacement.detach().abs().mean().cpu()),
        })
        return loss, metrics

    def _prompt_ranking_loss(self, out) -> Tensor:
        return out.semantic_prob.new_zeros(())

    def _router_supervision_weight(self, teacher_out, min_weight: float = 0.0) -> Tensor:
        foreground = torch.maximum(
            teacher_out.anchor_prob.detach(),
            torch.maximum(teacher_out.semantic_prob.detach(), teacher_out.appearance_prob.detach()),
        )
        foreground = torch.maximum(foreground, teacher_out.router.stable_foreground.detach()).clamp(0, 1)
        disagreement = teacher_out.router.disagreement.detach().clamp(0, 1)
        high_disagreement = disagreement.square()
        weight = (0.5 * foreground + 0.5 * disagreement + high_disagreement).clamp(0, 1)
        return weight.detach().clamp(0, 1).clamp_min(float(min_weight))

    def _router_target_loss(self, student_out, teacher_out, min_weight: float = 0.0) -> Tensor:
        with torch.no_grad():
            target = self._router_reliability_target(teacher_out)
            weight = self._router_supervision_weight(teacher_out, min_weight)
        pred = torch.cat([student_out.router.w_sem, student_out.router.w_app, student_out.router.w_geo], dim=1)
        loss = (pred - target.detach()).square().sum(dim=1, keepdim=True)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _router_reliability_target(self, teacher_out) -> Tensor:
        teacher_disagreement = (teacher_out.semantic_prob - teacher_out.appearance_prob).abs()
        sem_reliability = 1.0 - (teacher_out.semantic_prob - teacher_out.anchor_prob).abs()
        app_reliability = 1.0 - (teacher_out.appearance_prob - teacher_out.anchor_prob).abs()
        normalized_disagreement = torch.maximum(
            teacher_disagreement.clamp(0, 1),
            teacher_out.router.disagreement.detach().clamp(0, 1),
        )
        geo_reliability = normalized_disagreement.square()
        target = torch.cat([sem_reliability, app_reliability, geo_reliability], dim=1).clamp_min(0.0)
        return target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _stage_d_router_target_loss(self, student_out, teacher_out) -> Tensor:
        return self._router_target_loss(student_out, teacher_out, min_weight=0.0)
