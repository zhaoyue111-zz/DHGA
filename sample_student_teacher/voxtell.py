from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def prepare_voxtell_import(voxtell_repo: str) -> None:
    repo = str(Path(voxtell_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_prompts(path_or_values: Sequence[str]) -> list[str]:
    if len(path_or_values) == 1 and Path(path_or_values[0]).is_file():
        path = Path(path_or_values[0])
        if path.suffix.lower() == ".json":
            import json

            data = json.loads(path.read_text())
            if not isinstance(data, list):
                raise ValueError("Prompt JSON must be a list of strings")
            return [str(item) for item in data]
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [str(item) for item in path_or_values]


def build_prompt_variants(prompts: Sequence[str], templates: Sequence[str] | None = None) -> tuple[list[str], int]:
    templates = list(templates) if templates else ["{}"]
    return [template.format(prompt) for prompt in prompts for template in templates], len(templates)


class SimpleVoxTellSegmenter(nn.Module):
    """Minimal VoxTell binary segmentation wrapper used by the LoRA SFDA trainer."""

    def __init__(self, network: nn.Module, text_embeddings: Tensor, num_classes: int, num_templates: int) -> None:
        super().__init__()
        if num_classes != 1:
            raise ValueError("This minimal project supports binary segmentation with exactly one prompt")
        self.network = network
        self.num_classes = int(num_classes)
        self.num_templates = int(num_templates)
        self.register_buffer("text_embeddings", text_embeddings.detach().clone(), persistent=False)

    @classmethod
    def from_voxtell(
        cls,
        voxtell_repo: str,
        model_dir: str,
        prompts: Sequence[str],
        prompt_templates: Sequence[str] | None,
        text_encoding_model: str,
        device: torch.device,
    ) -> tuple["SimpleVoxTellSegmenter", object, list[str]]:
        prepare_voxtell_import(voxtell_repo)
        from voxtell.inference.predictor_multiclass import VoxTellPredictor

        prompts = load_prompts(list(prompts))
        predictor = VoxTellPredictor(
            model_dir=model_dir,
            device=device,
            text_encoding_model=text_encoding_model,
            perform_everything_on_device=False,
        )
        prompt_variants, num_templates = build_prompt_variants(prompts, prompt_templates)
        text_embeddings = predictor.embed_text_prompts(prompt_variants).to(device)
        dtype = next(predictor.network.project_text_embed.parameters()).dtype
        text_embeddings = text_embeddings.to(dtype=dtype)
        model = cls(predictor.network.to(device), text_embeddings, len(prompts), num_templates).to(device)
        return model, predictor, prompts

    def forward(self, image: Tensor) -> Tensor:
        skips, selected_feature = self._encode(image)
        logits = self._decode(skips, selected_feature)
        if self.num_templates > 1:
            logits = logits.view(logits.shape[0], self.num_classes, self.num_templates, *logits.shape[2:]).mean(dim=2)
        return logits

    def _encode(self, patch: Tensor) -> tuple[list[Tensor], Tensor]:
        encoder = self.network.encoder
        selected_idx = int(self.network.selected_decoder_layer)
        num_stages = len(encoder.stages)
        if selected_idx < 0:
            selected_idx += num_stages
        x = patch
        if encoder.stem is not None:
            x = encoder.stem(x)
        skips: list[Tensor] = []
        selected_feature = None
        for stage_idx, stage in enumerate(encoder.stages):
            x = stage(x)
            if stage_idx == selected_idx:
                selected_feature = x
            skips.append(x)
        if selected_feature is None:
            raise RuntimeError(f"selected_decoder_layer={self.network.selected_decoder_layer} was not produced")
        return skips, selected_feature

    def _decode(self, skips: list[Tensor], selected_feature: Tensor) -> Tensor:
        bottleneck = selected_feature.permute(0, 2, 3, 4, 1).contiguous()
        bottleneck = self.network.project_bottleneck_embed(bottleneck)
        bsz, h_dim, w_dim, d_dim, c_dim = bottleneck.shape
        memory = bottleneck.reshape(bsz, h_dim * w_dim * d_dim, c_dim).permute(1, 0, 2).contiguous()
        pos_embed = self._pos_embed_for_shape(h_dim, w_dim, d_dim, c_dim, memory)
        text_embedding = self.text_embeddings
        if text_embedding.ndim == 4:
            text_embedding = text_embedding.squeeze(2)
        if text_embedding.shape[0] == 1 and skips[-1].shape[0] != 1:
            text_embedding = text_embedding.expand(skips[-1].shape[0], -1, -1)
        text_embed = text_embedding.permute(1, 0, 2).contiguous()
        text_embed = text_embed.to(dtype=next(self.network.project_text_embed.parameters()).dtype)
        projected_prompt = self.network.project_text_embed(text_embed)
        mask_embedding, _ = self.network.transformer_decoder(
            tgt=projected_prompt,
            memory=memory,
            pos=pos_embed,
            memory_key_padding_mask=None,
        )
        mask_embedding = mask_embedding.permute(1, 0, 2).contiguous()
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
        return outs[-1]

    def _pos_embed_for_shape(self, h_dim: int, w_dim: int, d_dim: int, c_dim: int, reference: Tensor) -> Tensor:
        expected_tokens = h_dim * w_dim * d_dim
        pos_embed = getattr(self.network, "pos_embed", None)
        if pos_embed is not None and pos_embed.shape[0] == expected_tokens and pos_embed.shape[-1] == c_dim:
            return pos_embed.to(device=reference.device, dtype=reference.dtype)
        from positional_encodings.torch_encodings import PositionalEncoding3D

        pos_encoder = PositionalEncoding3D(c_dim).to(reference.device)
        pos = pos_encoder(torch.zeros(1, h_dim, w_dim, d_dim, c_dim, device=reference.device, dtype=torch.float32))
        return pos.reshape(1, expected_tokens, c_dim).permute(1, 0, 2).to(dtype=reference.dtype)
