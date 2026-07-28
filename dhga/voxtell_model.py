from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from dhga.config import DHGAConfig
from dhga.inference import finalize_probability
from dhga.experts import AppearanceExpert
from dhga.geometry import (
    GeometryTransportHead,
    extract_boundary_points,
    make_ray_offsets_mm,
    mask_to_sdf,
    sample_along_normals,
    sparse_displacements_to_dense_narrowband,
)
from dhga.routing import DisagreementRouter, RouterOutput
from dhga.shared_voxtell import SharedVoxTellFeatures, freeze_module, trainable_parameter_summary
from voxtell_sfda.adapter import build_prompt_variants, load_prompts, prepare_voxtell_import
from voxtell_sfda.lora import inject_lora_into_voxtell_decoder, mark_only_lora_trainable


@dataclass
class DHGAForwardOutput:
    semantic_logits: Tensor
    appearance_logits: Tensor
    semantic_prob: Tensor
    appearance_prob: Tensor
    anchor_prob: Tensor
    router: RouterOutput
    geometry: dict[str, Tensor]
    final_prob: Tensor
    features: SharedVoxTellFeatures


class PromptConditionedRayTokens(nn.Module):
    def __init__(self, prompt_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)
        self.prompt_dim = prompt_dim

    def forward(
        self,
        sampled_image: Tensor,
        sampled_sem: Tensor,
        sampled_app: Tensor,
        sampled_disagreement: Tensor,
        sampled_fused: Tensor,
        sampled_geo_gate: Tensor,
        sampled_sdf: Tensor,
        sampled_visual: Tensor,
        offsets_mm: Tensor,
        prompt_embedding: Tensor,
    ) -> Tensor:
        bsz, n_points, n_offsets, _ = sampled_image.shape
        prompt = prompt_embedding.float()
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0).expand(bsz, -1, -1)
        elif prompt.ndim == 3 and prompt.shape[0] == 1 and bsz != 1:
            prompt = prompt.expand(bsz, -1, -1)
        if prompt.ndim != 3:
            raise ValueError("prompt_embedding must have shape [N, D] or [B, N, D]")
        prompt = prompt.mean(dim=1)
        prompt = self.prompt_proj(prompt).view(bsz, 1, 1, -1).expand(bsz, n_points, n_offsets, -1)
        offsets = offsets_mm.to(sampled_image.device, sampled_image.dtype).view(1, 1, n_offsets, 1).expand(bsz, n_points, -1, -1)
        return torch.cat(
            [
                sampled_image.float(),
                sampled_sem.float(),
                sampled_app.float(),
                sampled_disagreement.float(),
                sampled_fused.float(),
                sampled_geo_gate.float(),
                sampled_sdf.float(),
                sampled_visual.float(),
                offsets.float(),
                prompt.float(),
            ],
            dim=-1,
        )


class DHGAVoxTellModel(nn.Module):
    """DHGA model around one frozen VoxTell encoder and two heterogeneous expert paths."""

    def __init__(self, network: nn.Module, text_embeddings: Tensor, config: DHGAConfig, num_classes: int, num_templates: int) -> None:
        super().__init__()
        if config.dhga_geometry_enabled and num_classes != 1:
            raise ValueError("Current DHGA geometry is single-class binary; pass exactly one prompt or disable geometry")
        self.network = network
        self.config = config
        self.num_classes = num_classes
        self.num_templates = num_templates
        self.register_buffer("text_embeddings", text_embeddings.detach().clone(), persistent=False)
        self.encoder_calls = 0
        self._printed_encoder_plan = False
        if config.dhga_freeze_voxtell:
            freeze_module(self.network)
        self.injected_lora = inject_lora_into_voxtell_decoder(  # decoder的cross attention增加lora
            self.network,
            rank=config.dhga_semantic_adapter_rank,
            alpha=float(config.dhga_semantic_adapter_rank) * 2.0,
            dropout=0.0,
            target=config.dhga_semantic_adapter_target,
        )
        mark_only_lora_trainable(self.network)
        channels = self._infer_encoder_channels()
        self.appearance_expert = AppearanceExpert(
            channels,
            config.dhga_appearance_feature_layers,
            config.dhga_appearance_hidden_ratio,
            config.dhga_appearance_feature_dropout,
        )
        self.router = DisagreementRouter(config.dhga_router_normalization)
        self.ray_offsets_mm = make_ray_offsets_mm(config.dhga_search_radius_mm, config.dhga_ray_step_mm)
        self.ray_tokens = PromptConditionedRayTokens(text_embeddings.shape[-1])
        visual_channels = self._feature_channels_for_layer(channels, config.dhga_geometry_feature_layer)
        self.geometry_visual_proj = nn.Conv3d(visual_channels, config.dhga_geometry_feature_channels, kernel_size=1)
        token_dim = 1 + 1 + 1 + 1 + 1 + 1 + 1 + config.dhga_geometry_feature_channels + 1 + 16
        self.geometry_head = GeometryTransportHead(token_dim, self.ray_offsets_mm)

    @classmethod
    def from_voxtell(
        cls,
        voxtell_repo: str,
        model_dir: str,
        prompts: list[str],
        config: DHGAConfig,
    ) -> tuple["DHGAVoxTellModel", Any]:
        prepare_voxtell_import(voxtell_repo)
        from voxtell.inference.predictor_multiclass import VoxTellPredictor

        device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        predictor = VoxTellPredictor(
            model_dir=model_dir,
            device=device,
            text_encoding_model=config.text_encoding_model,
            perform_everything_on_device=False,
        )
        prompt_variants, num_templates = build_prompt_variants(prompts, config.prompt_templates)
        text_embeddings = predictor.embed_text_prompts(prompt_variants).to(device)
        dtype = next(predictor.network.project_text_embed.parameters()).dtype
        text_embeddings = text_embeddings.to(dtype=dtype)
        model = cls(predictor.network.to(device), text_embeddings, config, len(prompts), num_templates).to(device)
        return model, predictor

    def _infer_encoder_channels(self) -> list[int]:
        channels = []
        for stage in getattr(self.network.encoder, "stages", []):
            out_channels = None
            for module in reversed(list(stage.modules())):
                if isinstance(module, nn.Conv3d):
                    out_channels = module.out_channels
                    break
            channels.append(int(out_channels or 1))
        if not channels:
            channels = [1]
        return channels

    @contextmanager
    def lora_disabled(self):
        old = {name: module.scaling for name, module in self.injected_lora.items()}
        for module in self.injected_lora.values():
            module.scaling = 0.0
        try:
            yield
        finally:
            for name, value in old.items():
                self.injected_lora[name].scaling = value

    @contextmanager
    def appearance_disabled(self):
        old_training = self.appearance_expert.training
        old_requires_grad = [param.requires_grad for param in self.appearance_expert.parameters()]
        self.appearance_expert.eval()
        for param in self.appearance_expert.parameters():
            param.requires_grad_(False)
        try:
            yield
        finally:
            for param, state in zip(self.appearance_expert.parameters(), old_requires_grad):
                param.requires_grad_(state)
            self.appearance_expert.train(old_training)

    def forward(self, image: Tensor, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0), run_geometry: bool = True) -> DHGAForwardOutput:
        features = self.encode_once(image, spacing)
        if not self.config.dhga_enabled:
            return self.baseline_forward_from_features(features)
        semantic_logits, projected_prompt = self.decode_from_features(features.encoder_stages, features.selected_feature)
        with self.lora_disabled():
            app_features = self.appearance_expert.adapt_features(features, self.config.dhga_appearance_feature_dropout)
            appearance_logits, _ = self.decode_from_features(app_features.encoder_stages, app_features.selected_feature)
        sem_prob = self._class_probs(semantic_logits)
        app_prob = self._class_probs(appearance_logits)
        with torch.no_grad(), self.lora_disabled(), self.appearance_disabled():
            anchor_logits, _ = self.decode_from_features(features.encoder_stages, features.selected_feature)
            anchor_prob = self._class_probs(anchor_logits)
        visual_feature = features.encoder_stages[self.geometry_feature_idx]
        router_visual = self.geometry_visual_proj(visual_feature).detach().mean(dim=1, keepdim=True)
        router = self.router(sem_prob, app_prob, visual_context=router_visual)
        geometry = self.run_geometry(image, sem_prob, app_prob, router, self.text_embeddings, spacing, visual_feature=visual_feature) if run_geometry and self.config.dhga_geometry_enabled else {}
        final_prob = router.fused_prob
        if "dense_displacement_mm" in geometry:
            sdf = mask_to_sdf(router.fused_prob >= self.config.pred_threshold, spacing)
            final_prob = finalize_probability(
                router.fused_prob,
                sdf,
                geometry["dense_displacement_mm"],
                router.w_geo,
                self.config,
                geometry.get("dense_valid_weight"),
            )
        return DHGAForwardOutput(
            semantic_logits=semantic_logits,
            appearance_logits=appearance_logits,
            semantic_prob=sem_prob,
            appearance_prob=app_prob,
            anchor_prob=anchor_prob,
            router=router,
            geometry=geometry,
            final_prob=final_prob,
            features=features,
        )

    def baseline_forward(self, image: Tensor, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> DHGAForwardOutput:
        return self.baseline_forward_from_features(self.encode_once(image, spacing))

    def baseline_forward_from_features(self, features: SharedVoxTellFeatures) -> DHGAForwardOutput:
        with torch.no_grad(), self.lora_disabled(), self.appearance_disabled():
            base_logits, _ = self.decode_from_features(features.encoder_stages, features.selected_feature)
            base_prob = self._class_probs(base_logits)
            router = self.router(base_prob, base_prob)
        return DHGAForwardOutput(
            semantic_logits=base_logits,
            appearance_logits=base_logits,
            semantic_prob=base_prob,
            appearance_prob=base_prob,
            anchor_prob=base_prob,
            router=router,
            geometry={},
            final_prob=base_prob,
            features=features,
        )

    def encode_once(self, image: Tensor, spacing: tuple[float, float, float]) -> SharedVoxTellFeatures:
        self.encoder_calls += 1
        selected_stages: list[int] = []
        skips, selected_feature, selected_idx = self._encoder_selected_skips(image, selected_stages)
        return SharedVoxTellFeatures(
            image=image,
            encoder_stages=skips,
            selected_feature=selected_feature,
            spacing=spacing,
            original_shape=tuple(image.shape[-3:]),
            metadata={"selected_feature_idx": selected_idx},
        )

    def decode_from_features(self, skips: list[Tensor], selected_feature: Tensor | None) -> tuple[Tensor, Tensor]:
        if selected_feature is None:
            selected_feature = skips[-1]
        bottleneck_embed = rearrange(selected_feature, "b c d h w -> b h w d c")
        bottleneck_embed = self.network.project_bottleneck_embed(bottleneck_embed)
        _, h_dim, w_dim, d_dim, c_dim = bottleneck_embed.shape
        memory = rearrange(bottleneck_embed, "b h w d c -> (h w d) b c")
        pos_embed = self._pos_embed_for_shape(h_dim, w_dim, d_dim, c_dim, memory)
        text_embedding = self.text_embeddings
        if text_embedding.ndim == 4:
            text_embedding = text_embedding.squeeze(2)
        if text_embedding.shape[0] == 1 and skips[-1].shape[0] != 1:
            text_embedding = text_embedding.expand(skips[-1].shape[0], -1, -1)
        text_embed = text_embedding.permute(1, 0, 2)
        text_embed = text_embed.to(dtype=next(self.network.project_text_embed.parameters()).dtype)
        projected_prompt = self.network.project_text_embed(text_embed)
        mask_embedding, _ = self.network.transformer_decoder(
            tgt=projected_prompt,
            memory=memory,
            pos=pos_embed,
            memory_key_padding_mask=None,
        )
        mask_embedding = mask_embedding.permute(1, 0, 2)
        mask_embeddings = [projection(mask_embedding) for projection in self.network.project_to_decoder_channels]
        outs = []
        for prompt_idx in range(text_embedding.shape[1]):
            prompt_embeds = [m[:, prompt_idx : prompt_idx + 1] for m in mask_embeddings]
            num_skips = len(skips)

            def run_decoder(*args):
                return tuple(self.network.decoder(list(args[:num_skips]), list(args[num_skips:])))

            if torch.is_grad_enabled():
                scale_outs = checkpoint(run_decoder, *skips, *prompt_embeds, use_reentrant=False)
            else:
                scale_outs = run_decoder(*skips, *prompt_embeds)
            outs.append(scale_outs)
        outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]
        return outs[-1], projected_prompt.permute(1, 0, 2)

    def run_geometry(
        self,
        image: Tensor,
        sem_prob: Tensor,
        app_prob: Tensor,
        router: RouterOutput,
        prompt_embedding: Tensor,
        spacing: tuple[float, float, float],
        initial_sdf: Tensor | None = None,
        visual_feature: Tensor | None = None,
        visual_feature_is_projected: bool = False,
        boundary_points: Tensor | None = None,
        boundary_normals: Tensor | None = None,
        valid_boundary_points: Tensor | None = None,
    ) -> dict[str, Tensor]:
        sdf = initial_sdf if initial_sdf is not None else mask_to_sdf(router.fused_prob >= self.config.pred_threshold, spacing)
        if boundary_points is None or boundary_normals is None or valid_boundary_points is None:
            points, normals, valid_points = extract_boundary_points(
                sdf,
                self.config.dhga_surface_tolerance_mm,
                self.config.dhga_max_boundary_points,
                spacing,
                sampling_weight=router.geometry_disagreement_weight,
            )
        else:
            points, normals, valid_points = boundary_points, boundary_normals, valid_boundary_points
        offsets = self.ray_offsets_mm.to(image.device)
        sampled_image, valid_ray = sample_along_normals(image[:, :1], points, normals, offsets, spacing)
        sampled_sem, valid_sem = sample_along_normals(sem_prob, points, normals, offsets, spacing)
        sampled_app, valid_app = sample_along_normals(app_prob, points, normals, offsets, spacing)
        sampled_dis, valid_dis = sample_along_normals(router.disagreement, points, normals, offsets, spacing)
        sampled_fused, valid_fused = sample_along_normals(router.fused_prob, points, normals, offsets, spacing)
        sampled_geo_gate, valid_geo_gate = sample_along_normals(router.w_geo, points, normals, offsets, spacing)
        sampled_sdf, valid_sdf = sample_along_normals(sdf, points, normals, offsets, spacing)
        if visual_feature is None:
            raise RuntimeError("run_geometry requires explicit visual_feature from the shared encoder")
        if not visual_feature_is_projected:
            visual_feature = self.geometry_visual_proj(visual_feature)
        sampled_visual, valid_visual = self._sample_visual_feature(visual_feature, points, normals, offsets, spacing, tuple(image.shape[-3:]))
        valid = valid_ray & valid_sem & valid_app & valid_dis & valid_fused & valid_geo_gate & valid_sdf & valid_visual & valid_points.unsqueeze(-1)
        tokens = self.ray_tokens(
            sampled_image,
            sampled_sem,
            sampled_app,
            sampled_dis,
            sampled_fused,
            sampled_geo_gate,
            sampled_sdf,
            sampled_visual,
            offsets,
            prompt_embedding,
        )
        pred = self._geometry_forward_chunked(tokens, valid)
        dense, dense_weight = sparse_displacements_to_dense_narrowband(
            points,
            pred["expected_displacement_mm"] * self._sample_point_weight(router.geometry_disagreement_weight, points),
            valid_points,
            tuple(image.shape[-3:]),
            spacing=spacing,
            diffusion_mm=self.config.dhga_displacement_diffusion_mm,
            return_weight=True,
        )
        return {
            "sdf": sdf,
            "boundary_points_zyx": points,
            "boundary_normals_zyx": normals,
            "valid_boundary_points": valid_points,
            "ray_valid_mask": valid,
            "ray_tokens": tokens,
            "sparse_displacement_mm": pred["expected_displacement_mm"],
            "dense_displacement_mm": dense,
            "dense_valid_weight": dense_weight,
            "displacement_entropy": pred["entropy"],
        }

    def _sample_point_weight(self, weight_volume: Tensor, points: Tensor) -> Tensor:
        rounded = points.round().long()
        out = torch.zeros(points.shape[:2], device=points.device, dtype=weight_volume.dtype)
        for batch_idx in range(points.shape[0]):
            coords = rounded[batch_idx]
            valid = (
                (coords[:, 0] >= 0)
                & (coords[:, 0] < weight_volume.shape[-3])
                & (coords[:, 1] >= 0)
                & (coords[:, 1] < weight_volume.shape[-2])
                & (coords[:, 2] >= 0)
                & (coords[:, 2] < weight_volume.shape[-1])
            )
            c = coords[valid]
            if c.numel():
                out[batch_idx, valid] = weight_volume[batch_idx, 0, c[:, 0], c[:, 1], c[:, 2]]
        return out.clamp(0, 1)

    def _geometry_forward_chunked(self, tokens: Tensor, valid: Tensor) -> dict[str, Tensor]:
        chunk = int(self.config.dhga_boundary_chunk_size)
        if tokens.shape[1] <= chunk:
            return self.geometry_head(tokens, valid)
        outputs: dict[str, list[Tensor]] = {"logits": [], "prob": [], "expected_displacement_mm": [], "entropy": []}
        for start in range(0, tokens.shape[1], chunk):
            part = self.geometry_head(tokens[:, start : start + chunk], valid[:, start : start + chunk])
            for key in outputs:
                outputs[key].append(part[key])
        return {key: torch.cat(values, dim=1) for key, values in outputs.items()}

    def _sample_visual_feature(
        self,
        visual_feature: Tensor,
        points: Tensor,
        normals: Tensor,
        offsets: Tensor,
        spacing: tuple[float, float, float],
        image_shape: tuple[int, int, int],
    ) -> tuple[Tensor, Tensor]:
        scale = torch.tensor(
            [
                (visual_feature.shape[-3] - 1) / max(image_shape[0] - 1, 1),
                (visual_feature.shape[-2] - 1) / max(image_shape[1] - 1, 1),
                (visual_feature.shape[-1] - 1) / max(image_shape[2] - 1, 1),
            ],
            device=points.device,
            dtype=points.dtype,
        )
        feature_points = points * scale.view(1, 1, 3)
        feature_spacing = tuple(float(spacing[idx]) / max(float(scale[idx]), 1e-6) for idx in range(3))
        return sample_along_normals(visual_feature, feature_points, normals, offsets, feature_spacing)

    def _class_probs(self, logits: Tensor) -> Tensor:
        probs = logits.float().sigmoid()
        if self.num_templates > 1:
            probs = probs.view(probs.shape[0], self.num_classes, self.num_templates, *probs.shape[2:]).mean(dim=2)
        return probs

    def _encoder_selected_skips(self, patch: Tensor, selected_decoder_stages: list[int]) -> tuple[list[Tensor], Tensor, int]:
        encoder = self.network.encoder
        num_encoder_stages = len(encoder.stages)
        selected_feature_idx = int(self.network.selected_decoder_layer)
        if selected_feature_idx < 0:
            selected_feature_idx += num_encoder_stages
        keep_from_stage = 0
        if selected_decoder_stages:
            keep_from_stage = max(0, num_encoder_stages - (max(selected_decoder_stages) + 2))
        x = patch
        if encoder.stem is not None:
            x = encoder.stem(x)
        skips = []
        selected_feature = None
        for stage_idx, stage in enumerate(encoder.stages):
            x = stage(x)
            if stage_idx == selected_feature_idx:
                selected_feature = x
            if stage_idx >= keep_from_stage:
                skips.append(x)
        if selected_feature is None:
            raise RuntimeError(f"selected_decoder_layer={self.network.selected_decoder_layer} was not produced")
        return skips, selected_feature, selected_feature_idx

    def _feature_channels_for_layer(self, channels: list[int], raw_idx: int) -> int:
        idx = raw_idx if raw_idx >= 0 else len(channels) + raw_idx
        if idx < 0 or idx >= len(channels):
            raise ValueError(f"geometry feature layer {raw_idx} is outside {len(channels)} stages")
        self.geometry_feature_idx = idx
        return channels[idx]

    def _pos_embed_for_shape(self, h_dim: int, w_dim: int, d_dim: int, c_dim: int, reference: Tensor) -> Tensor:
        expected_tokens = h_dim * w_dim * d_dim
        if self.network.pos_embed.shape[0] == expected_tokens and self.network.pos_embed.shape[-1] == c_dim:
            return self.network.pos_embed.to(device=reference.device, dtype=reference.dtype)
        from positional_encodings.torch_encodings import PositionalEncoding3D

        pos_encoder = PositionalEncoding3D(c_dim).to(reference.device)
        pos = pos_encoder(torch.zeros(1, h_dim, w_dim, d_dim, c_dim, device=reference.device, dtype=torch.float32))
        return rearrange(pos, "b h w d c -> (h w d) b c").to(dtype=reference.dtype)

    def trainable_summary(self) -> dict[str, Any]:
        return trainable_parameter_summary(
            {
                "dhga.semantic_lora": self.network,
                "dhga.appearance_expert": self.appearance_expert,
                "dhga.router": self.router,
                "dhga.geometry_head": self.geometry_head,
                "dhga.ray_tokens": self.ray_tokens,
            }
        )


def build_dhga_voxtell_model(config: DHGAConfig, prompts_or_file: list[str]) -> tuple[DHGAVoxTellModel, Any, list[str]]:
    prompts = load_prompts(prompts_or_file)
    if config.dhga_geometry_enabled and len(prompts) != 1:
        raise ValueError("Current DHGA geometry is single-class binary; pass exactly one prompt or disable geometry")
    model, predictor = DHGAVoxTellModel.from_voxtell(config.voxtell_repo, config.model_dir, prompts, config)
    return model, predictor, prompts
