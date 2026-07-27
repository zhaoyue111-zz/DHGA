import json
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from torch.utils.checkpoint import checkpoint
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from tqdm import tqdm
from .lora import (
    inject_lora_into_voxtell_decoder,
    lora_parameters,
    lora_state_dict,
    mark_only_lora_trainable,
)


@dataclass
class VoxTellSFDAConfig:
    voxtell_repo: str
    model_dir: str
    data_dir: str
    split_manifest: str
    sequences: str
    prompts: List[str]
    save_dir: str
    val_label_dir: str = ""
    label_values: List[int] = None
    val_interval: int = 1
    val_max_cases: int = 0
    pred_threshold: float = 0.5
    largest_component: bool = False
    save_val_masks: bool = True
    prompt_templates: List[str] = None
    text_encoding_model: str = "Qwen/Qwen3-Embedding-4B"
    device: str = "cuda"
    epochs: int = 1
    steps_per_volume: int = 0
    lr: float = 1e-4
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    lora_target: str = "cross"
    entropy_weight: float = 1.0
    alpha_global: float = 1.0
    consistency_weight: float = 0.0
    source_consistency_weight: float = 0.0
    pseudo_weight: float = 0.0
    foreground_prior_weight: float = 0.0
    foreground_prior: float = 0.02
    pseudo_threshold: float = 0.9
    noise_std: float = 0.03
    local_entropy_mode: str = "class_softmax"
    train_decoder_output_stages: List[int] = None
    amp: bool = True
    seed: int = 0
    max_cases: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(path_or_values: Sequence[str]) -> List[str]:
    if len(path_or_values) == 1 and Path(path_or_values[0]).is_file():
        path = Path(path_or_values[0])
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            if not isinstance(data, list):
                raise ValueError("Prompt JSON must be a list of strings")
            return [str(item) for item in data]
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [str(item) for item in path_or_values]


def build_prompt_variants(prompts: Sequence[str], templates: Sequence[str] = None) -> Tuple[List[str], int]:
    templates = list(templates) if templates else ["{}"]
    variants = []
    for prompt in prompts:
        for template in templates:
            variants.append(template.format(prompt))
    return variants, len(templates)


def discover_images(data_dir: str) -> List[Path]:
    root = Path(data_dir)
    files = sorted(root.rglob("*.nii")) + sorted(root.rglob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No .nii or .nii.gz files found under {data_dir}")
    return files


def parse_sequences(sequences: str | Iterable[str] | None) -> List[str] | None:
    if sequences is None:
        return None
    if isinstance(sequences, str):
        parts = sequences.replace(",", " ").split()
        return parts or None
    return list(sequences)


def discover_image_paths(data_root: Path, sequences: Iterable[str] | None = None) -> list[Path]:
    data_root = data_root.expanduser()
    sequences = parse_sequences(sequences)

    if any(data_root.glob("*.nii.gz")) or any(data_root.glob("*.nii")):
        paths = sorted(data_root.glob("*.nii.gz")) + sorted(data_root.glob("*.nii"))
        return paths

    if (data_root / "images").is_dir():
        image_root = data_root / "images"
    else:
        image_root = data_root

    selected = list(sequences) if sequences else sorted(p.name for p in image_root.iterdir() if p.is_dir())
    paths: list[Path] = []
    for sequence in selected:
        sequence_dir = image_root / sequence
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"Missing image sequence directory: {sequence_dir}")
        paths.extend(sorted(sequence_dir.glob("*.nii.gz")))
    if not paths:
        raise RuntimeError(f"No .nii.gz files found under {image_root}")
    return paths

def load_split_manifest(
    manifest_path: Path,
    split: str,
    data_root: Path | None = None,
    sequences: Iterable[str] | None = None,
) -> list[Path]:
    with manifest_path.open() as f:
        manifest = json.load(f)

    split_key = split if split in manifest else f"{split}_cases"
    if split_key not in manifest:
        available = ", ".join(sorted(k for k, v in manifest.items() if isinstance(v, list)))
        raise KeyError(f"Split '{split}' not found in {manifest_path}. Available list fields: {available}")

    values = manifest[split_key]
    raw_paths = [Path(p) for p in values]
    if any(not p.is_absolute() for p in raw_paths):
        if data_root is None:
            raise ValueError(f"Split '{split_key}' in {manifest_path} contains relative case names; data_root is required")
        case_to_path = {p.name: p for p in discover_image_paths(data_root, sequences)}
        missing_cases = [str(p) for p in raw_paths if p.name not in case_to_path]
        if missing_cases:
            preview = ", ".join(missing_cases[:5])
            suffix = "" if len(missing_cases) <= 5 else f", ... ({len(missing_cases)} missing total)"
            raise FileNotFoundError(f"Missing cases from split '{split_key}' in {manifest_path}: {preview}{suffix}")
        paths = [case_to_path[p.name] for p in raw_paths]
    else:
        paths = raw_paths
    if not paths:
        raise RuntimeError(f"Split '{split_key}' in {manifest_path} is empty")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
        raise FileNotFoundError(f"Missing image files from split '{split_key}' in {manifest_path}: {preview}{suffix}")
    return paths


def match_label_path(image_path: Path, label_dir: str) -> Path:
    label_path = Path(label_dir) / image_path.name
    if label_path.exists():
        return label_path
    if image_path.name.endswith(".nii.gz"):
        alt = Path(label_dir) / image_path.name[:-3]
    else:
        alt = Path(label_dir) / f"{image_path.name}.gz"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"No matching label for {image_path.name} under {label_dir}")


def sigmoid_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.sigmoid()
    eps = torch.finfo(probs.dtype).eps
    return -(probs * (probs + eps).log() + (1 - probs) * (1 - probs + eps).log())


def softmax_entropy(logits: torch.Tensor, dim: int) -> torch.Tensor:
    return -(logits.softmax(dim) * logits.log_softmax(dim)).sum(dim)


def foreground_dice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum(dtype=np.float64)
    pred_sum = pred.sum(dtype=np.float64)
    gt_sum = gt.sum(dtype=np.float64)
    dice = (2.0 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0

    fg_union = np.logical_or(pred, gt).sum(dtype=np.float64)
    iou = intersection / fg_union if fg_union > 0 else 1.0
    return float(dice), float(iou)


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num_components = ndimage.label(mask, structure=structure)
    if num_components <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(counts.argmax())
    return labeled == largest_label


def pseudo_label_loss(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probs = logits.sigmoid().detach()
    high_conf = (probs >= threshold) | (probs <= (1.0 - threshold))
    if not torch.any(high_conf):
        return logits.new_zeros(())
    targets = (probs >= 0.5).to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return loss[high_conf].mean()


def random_intensity_augment(patch: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return patch
    noise = torch.randn_like(patch) * noise_std
    scale = 1.0 + (torch.rand((patch.shape[0], 1, 1, 1, 1), device=patch.device) - 0.5) * 0.1
    return patch * scale + noise


def random_slicer(image_shape: Sequence[int], patch_size: Sequence[int]) -> Tuple[slice, ...]:
    starts = []
    for dim, size in zip(image_shape, patch_size):
        if dim <= size:
            starts.append(0)
        else:
            starts.append(random.randint(0, dim - size))
    return tuple([slice(None), *[slice(start, start + size) for start, size in zip(starts, patch_size)]]) # a[2:5]=a[slice(2,5)] slice是pythonb内置类型


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_")


def prepare_voxtell_import(voxtell_repo: str) -> None:
    repo = str(Path(voxtell_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class VoxTellSFDAAdapter:
    def __init__(self, config: VoxTellSFDAConfig) -> None:
        self.config = config
        set_seed(config.seed)
        prepare_voxtell_import(config.voxtell_repo)

        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from voxtell.inference.predictor_multiclass import VoxTellPredictor

        self.reader = NibabelIOWithReorient()
        self.pad_nd_image = pad_nd_image
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        self.predictor = VoxTellPredictor(
            model_dir=config.model_dir,
            device=self.device,
            text_encoding_model=config.text_encoding_model,
            perform_everything_on_device=False,
        )
        print(f"Loaded VoxTell checkpoint from: {Path(config.model_dir) / 'checkpoint_final.pth'}")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.network = self.predictor.network.to(self.device)
        self.network.eval()

        self.injected = inject_lora_into_voxtell_decoder(
            self.network,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            target=config.lora_target,
        )
        self.network = self.network.to(self.device)
        mark_only_lora_trainable(self.network)
        self.optimizer = torch.optim.AdamW(lora_parameters(self.network), lr=config.lr, weight_decay=0.0, foreach=False)
        self.base_prompts = list(config.prompts)
        self.label_values = config.label_values or [5]
        if len(self.label_values) != len(self.base_prompts):
            if len(self.base_prompts) == 1 and len(self.label_values) == 1:
                pass
            else:
                raise ValueError("label_values length must match prompts length")
        self.prompt_variants, self.num_templates = build_prompt_variants(
            self.base_prompts,
            config.prompt_templates,
        )
        self.num_classes = len(self.base_prompts)
        self.text_embeddings = self.predictor.embed_text_prompts(self.prompt_variants).to(self.device)
        text_project_dtype = next(self.network.project_text_embed.parameters()).dtype
        self.text_embeddings = self.text_embeddings.to(dtype=text_project_dtype)
        self._printed_padding_info = False
        self._printed_encoder_plan = False

    def _optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    @contextmanager
    def _disable_lora(self):
        old_scalings = {}
        for name, module in self.injected.items():
            old_scalings[name] = module.scaling
            module.scaling = 0.0
        try:
            yield
        finally:
            for name, module in self.injected.items():
                module.scaling = old_scalings[name]

    def _load_volume_windows(self, path: Path) -> Tuple[torch.Tensor, List[Tuple]]:
        image, _ = self.reader.read_images([str(path)])
        image, _, _ = self.predictor.preprocess(image)
        patch_size = tuple(int(v) for v in self.predictor.patch_size)
        image, _ = self.pad_nd_image(image, patch_size, "constant", {"value": 0}, True, None)
        slicers = self.predictor._internal_get_sliding_window_slicers(image.shape[1:])
        if not self._printed_padding_info:
            print(
                f"Training sliding window: volume_shape={tuple(image.shape)}, "
                f"patch_size={patch_size}, tile_step_size={self.predictor.tile_step_size}, "
                f"num_patches={len(slicers)}"
            )
            self._printed_padding_info = True
        return image, slicers

    def _training_target_shape(self, image_shape: Sequence[int]) -> Tuple[int, ...]:
        patch_size = tuple(int(v) for v in self.predictor.patch_size)
        stride_multiple = self._encoder_stride_multiple(len(image_shape))
        target = []
        for size, min_size, multiple in zip(image_shape, patch_size, stride_multiple):
            padded = max(int(size), int(min_size))
            padded = ((padded + multiple - 1) // multiple) * multiple
            target.append(padded)
        return tuple(target)

    def _encoder_stride_multiple(self, ndim: int) -> Tuple[int, ...]:
        strides = getattr(self.network.encoder, "strides", None)
        if strides is None:
            return tuple([32] * ndim)

        multiple = [1] * ndim
        for stride in strides:
            if isinstance(stride, int):
                stride_values = [stride] * ndim
            else:
                stride_values = list(stride)
            for dim_idx in range(ndim):
                multiple[dim_idx] *= int(stride_values[dim_idx])
        return tuple(max(value, 1) for value in multiple)

    def _forward_outputs(self, patch: torch.Tensor) -> List[torch.Tensor]:
        outputs, _ = self._forward_outputs_and_features(patch)
        if isinstance(outputs, torch.Tensor):
            return [outputs]
        return list(outputs)

    def _selected_decoder_stages(self) -> List[int]:
        if self.config.train_decoder_output_stages:
            return list(self.config.train_decoder_output_stages)
        return []

    def _encoder_selected_skips(
        self,
        patch: torch.Tensor,
        selected_decoder_stages: List[int],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        encoder = self.network.encoder
        num_encoder_stages = len(encoder.stages)
        selected_feature_idx = int(self.network.selected_decoder_layer)
        if selected_feature_idx < 0:
            selected_feature_idx += num_encoder_stages

        if selected_decoder_stages:
            max_decoder_stage = max(int(stage) for stage in selected_decoder_stages)
            keep_from_stage = num_encoder_stages - (max_decoder_stage + 2)
        else:
            keep_from_stage = 0
        keep_from_stage = max(0, keep_from_stage)
        print_encoder_plan = not self._printed_encoder_plan
        if print_encoder_plan:
            print(
                f"Training encoder forward: input_shape={tuple(patch.shape)}, "
                f"selected_decoder_stages={selected_decoder_stages or 'full'}, "
                f"keep_encoder_stages={list(range(keep_from_stage, num_encoder_stages))}, "
                f"selected_feature_stage={selected_feature_idx}, "
                f"cudnn_benchmark={torch.backends.cudnn.benchmark}, "
                f"cudnn_deterministic={torch.backends.cudnn.deterministic}, "
                f"cudnn_enabled={torch.backends.cudnn.enabled}"
            )

        def run_encoder_stages() -> Tuple[List[torch.Tensor], torch.Tensor]:
            x = patch
            if encoder.stem is not None:
                if print_encoder_plan:
                    print(f"  encoder stem: input_shape={tuple(x.shape)}")
                x = encoder.stem(x)

            decoder_skips = []
            selected_feature = None
            for stage_idx, stage in enumerate(encoder.stages):
                if print_encoder_plan:
                    print(f"  encoder stage {stage_idx}: input_shape={tuple(x.shape)}")
                x = stage(x)
                if stage_idx == selected_feature_idx:
                    selected_feature = x
                if stage_idx >= keep_from_stage:
                    decoder_skips.append(x)
            return decoder_skips, selected_feature

        decoder_skips, selected_feature = run_encoder_stages()
        self._printed_encoder_plan = True

        if selected_feature is None:
            raise RuntimeError(f"selected_decoder_layer={self.network.selected_decoder_layer} was not produced")
        return decoder_skips, selected_feature

    def _decoder_selected_outputs(
        self,
        skips: List[torch.Tensor],
        mask_embeddings: List[torch.Tensor],
        selected_stages: List[int],
    ) -> List[torch.Tensor]:
        if not selected_stages:
            return self.network.decoder(skips, mask_embeddings)

        decoder = self.network.decoder
        selected = set(int(stage) for stage in selected_stages)
        max_stage = max(selected)
        if min(selected) < 0 or max_stage >= len(decoder.stages):
            raise ValueError(
                f"train_decoder_output_stages must be in [0, {len(decoder.stages) - 1}], "
                f"got {sorted(selected)}"
            )

        lres_input = skips[-1]
        outputs = []
        mask_embeddings = list(mask_embeddings)[::-1]

        for stage_idx in range(len(decoder.stages)):
            x = decoder.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = decoder.stages[stage_idx](x)

            seg_pred = None
            if stage_idx == (len(decoder.stages) - 1):
                seg_pred = torch.einsum("b c h w d, b n c -> b n h w d", x, mask_embeddings[-1])
            elif stage_idx >= len(decoder.stages) - len(mask_embeddings):
                mask_embedding = mask_embeddings.pop(0)
                batch_size, _, _ = mask_embedding.shape
                mask_embedding_reshaped = mask_embedding.view(batch_size, decoder.num_heads, -1)
                fusion_features = torch.einsum("b c h w d, b n c -> b n h w d", x, mask_embedding_reshaped)
                x = torch.cat((x, fusion_features), dim=1)
                seg_pred = decoder.seg_layers[stage_idx](x)

            if stage_idx in selected and seg_pred is not None:
                outputs.append(seg_pred)
            if stage_idx >= max_stage:
                break
            lres_input = x

        return outputs

    def _forward_outputs_and_features(self, patch: torch.Tensor) -> Tuple[List[torch.Tensor], dict]:
        with torch.no_grad():
            selected_stages = self._selected_decoder_stages()
            skips, selected_feature = self._encoder_selected_skips(patch, selected_stages)

            bottleneck_embed = rearrange(selected_feature, "b c d h w -> b h w d c")
            bottleneck_embed = self.network.project_bottleneck_embed(bottleneck_embed)
            _, h_dim, w_dim, d_dim, c_dim = bottleneck_embed.shape
            selected_pseudo_cls = bottleneck_embed.mean(dim=(1, 2, 3)) # 用selected visual feature的全局池化作为pseudo-CLS
            bottleneck_embed = rearrange(bottleneck_embed, "b h w d c -> (h w d) b c") # VoxTell prompt decoder的image memory
            pos_embed = self._pos_embed_for_shape(h_dim, w_dim, d_dim, c_dim, bottleneck_embed)

            text_embedding = self.text_embeddings
            if text_embedding.ndim == 4:
                text_embedding = text_embedding.squeeze(2)
            if text_embedding.shape[0] == 1 and patch.shape[0] != 1:
                text_embedding = text_embedding.expand(patch.shape[0], -1, -1)

            text_embed = text_embedding.permute(1, 0, 2)
            text_project_dtype = next(self.network.project_text_embed.parameters()).dtype
            text_embed = text_embed.to(dtype=text_project_dtype)
            text_embed = self.network.project_text_embed(text_embed)

        mask_embedding, attentions = self.network.transformer_decoder(
            tgt=text_embed,
            memory=bottleneck_embed,
            pos=pos_embed,
            memory_key_padding_mask=None,
        )
        mask_embedding = mask_embedding.permute(1, 0, 2)
        mask_embeddings = [
            projection(mask_embedding)
            for projection in self.network.project_to_decoder_channels
        ]

        # outs = []
        # for prompt_idx in range(text_embedding.shape[1]):
        #     prompt_embeds = [m[:, prompt_idx:prompt_idx + 1] for m in mask_embeddings]
        #     # outs.append(self.network.decoder(skips, prompt_embeds))
        # outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]

        outs = []
        for prompt_idx in range(text_embedding.shape[1]):
            prompt_embeds = [m[:, prompt_idx:prompt_idx + 1] for m in mask_embeddings]
            num_skips = len(skips)

            def run_decoder(*args):
                decoder_skips = list(args[:num_skips])
                decoder_prompts = list(args[num_skips:])
                decoder_outputs = self._decoder_selected_outputs(
                    decoder_skips,
                    decoder_prompts,
                    self.config.train_decoder_output_stages or [],
                )
                return tuple(decoder_outputs)

            if torch.is_grad_enabled():
                scale_outs = checkpoint(run_decoder, *skips, *prompt_embeds, use_reentrant=False)
            else:
                scale_outs = run_decoder(*skips, *prompt_embeds)
            outs.append(scale_outs)
        outs = [torch.cat(scale_outs, dim=1) for scale_outs in zip(*outs)]

        features = {
            "encoder_stages": skips,
            "pseudo_cls_tokens": [stage.mean(dim=(2, 3, 4)) for stage in skips],
            "selected_pseudo_cls": selected_pseudo_cls,
            "mask_embedding": mask_embedding,
            "attentions": attentions,
        }
        return list(outs), features

    def _pos_embed_for_shape(self, h_dim: int, w_dim: int, d_dim: int, c_dim: int, reference: torch.Tensor) -> torch.Tensor:
        expected_tokens = h_dim * w_dim * d_dim
        if self.network.pos_embed.shape[0] == expected_tokens and self.network.pos_embed.shape[-1] == c_dim:
            return self.network.pos_embed.to(device=reference.device, dtype=reference.dtype)

        from positional_encodings.torch_encodings import PositionalEncoding3D

        pos_encoder = PositionalEncoding3D(c_dim).to(reference.device)
        pos = pos_encoder(
            torch.zeros(1, h_dim, w_dim, d_dim, c_dim, device=reference.device, dtype=torch.float32)
        )
        pos = rearrange(pos, "b h w d c -> (h w d) b c")
        return pos.to(dtype=reference.dtype)

    def _reshape_prompt_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        return tensor.view(shape[0], self.num_classes, self.num_templates, *shape[2:])

    def _local_mlmp_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        grouped = self._reshape_prompt_axis(logits)
        if self.config.local_entropy_mode == "class_softmax" and self.num_classes > 1:
            return softmax_entropy(grouped, dim=1).mean()
        if self.config.local_entropy_mode == "class_softmax" and self.num_classes <= 1:
            return sigmoid_entropy(grouped).mean()
        if self.config.local_entropy_mode != "binary":
            raise ValueError("local_entropy_mode must be 'class_softmax' or 'binary'")
        return sigmoid_entropy(grouped).mean()

    def _stage_entropy_score(self, logits: torch.Tensor) -> torch.Tensor:
        grouped = self._reshape_prompt_axis(logits)
        if self.config.local_entropy_mode == "class_softmax" and self.num_classes > 1:
            return softmax_entropy(grouped, dim=1).mean()
        return sigmoid_entropy(grouped).mean()

    def _foreground_prior_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self._reshape_prompt_axis(logits).sigmoid()
        mean_prob = probs.mean()
        target = logits.new_tensor(float(self.config.foreground_prior))
        return F.relu(target - mean_prob).pow(2)

    def _entropy_weighted_outputs(self, outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(outputs) == 1:
            return outputs[0], outputs[0].new_ones(1)

        entropies = torch.stack([self._stage_entropy_score(logits) for logits in outputs])
        weights = F.softmax(-entropies, dim=0)
        target_shape = outputs[-1].shape[-3:]

        fused = outputs[-1].new_zeros(outputs[-1].shape)
        for weight, logits in zip(weights, outputs):
            if logits.shape[-3:] != target_shape:
                logits = F.interpolate(logits, size=target_shape, mode="trilinear", align_corners=False)
            fused = fused + weight.to(dtype=logits.dtype) * logits
        return fused, weights.detach()

    def _global_mlmp_entropy(self, features: dict) -> torch.Tensor:
        cls_token = features["selected_pseudo_cls"]
        mask_embedding = features["mask_embedding"]
        cls_token = F.normalize(cls_token.float(), dim=-1)
        mask_embedding = F.normalize(mask_embedding.float(), dim=-1)
        logits = torch.einsum("bd,bnd->bn", cls_token, mask_embedding)
        grouped = logits.view(logits.shape[0], self.num_classes, self.num_templates)
        if self.num_classes > 1:
            return softmax_entropy(grouped, dim=1).mean()
        return sigmoid_entropy(grouped).mean()

    def _loss(self, patch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        outputs, features = self._forward_outputs_and_features(patch)
        entropy = patch.new_zeros(())
        pseudo = patch.new_zeros(())
        foreground_prior = patch.new_zeros(())
        source_consistency = patch.new_zeros(())

        # LUAML(T)
        for idx, logits in enumerate(outputs):
            weight = 1.0 / (2 ** idx)
            entropy = entropy + weight * self._local_mlmp_entropy(logits)
            if self.config.pseudo_weight > 0:
                pseudo = pseudo + weight * pseudo_label_loss(logits, self.config.pseudo_threshold)
            if self.config.foreground_prior_weight > 0:
                foreground_prior = foreground_prior + weight * self._foreground_prior_loss(logits)

        global_entropy = self._global_mlmp_entropy(features)

        if self.config.source_consistency_weight > 0:
            with torch.no_grad(), self._disable_lora():
                teacher_outputs = self._forward_outputs(patch)
                teacher_logits, _ = self._entropy_weighted_outputs(teacher_outputs)
                teacher_probs = teacher_logits.float().sigmoid()
            student_logits, _ = self._entropy_weighted_outputs(outputs)
            student_probs = student_logits.float().sigmoid()
            if teacher_probs.shape[-3:] != student_probs.shape[-3:]:
                teacher_probs = F.interpolate(
                    teacher_probs,
                    size=student_probs.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            source_consistency = F.mse_loss(student_probs, teacher_probs)

        consistency = patch.new_zeros(())
        if self.config.consistency_weight > 0:
            with torch.no_grad():
                teacher = [out.detach().float().sigmoid() for out in outputs]
            aug_patch = random_intensity_augment(patch, self.config.noise_std)  # 对patch加噪
            aug_outputs = self._forward_outputs(aug_patch) # 再次前向传播
            for idx, (student_logits, teacher_probs) in enumerate(zip(aug_outputs, teacher)):
                student = student_logits.float().sigmoid()
                if student.shape[-3:] != teacher_probs.shape[-3:]:
                    teacher_probs = F.interpolate(
                        teacher_probs,
                        size=student.shape[-3:],
                        mode="trilinear",
                        align_corners=False,
                    )
                consistency = consistency + (1.0 / (2 ** idx)) * F.mse_loss(student, teacher_probs)

        loss = (
            self.config.entropy_weight * entropy
            + self.config.alpha_global * global_entropy
            + self.config.pseudo_weight * pseudo
            + self.config.foreground_prior_weight * foreground_prior
            + self.config.source_consistency_weight * source_consistency
            + self.config.consistency_weight * consistency
        )
        metrics = {
            "loss": float(loss.detach().cpu()),
            "local_entropy": float(entropy.detach().cpu()),
            "global_entropy": float(global_entropy.detach().cpu()),
            "pseudo": float(pseudo.detach().cpu()),
            "foreground_prior": float(foreground_prior.detach().cpu()),
            "source_consistency": float(source_consistency.detach().cpu()),
            "consistency": float(consistency.detach().cpu()),
        }
        return loss, metrics

    def adapt(self) -> None:
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2))

        train_paths = load_split_manifest(
            Path(self.config.split_manifest),
            "train",
            Path(self.config.data_dir),
            self.config.sequences,
        )
        val_paths = load_split_manifest(
            Path(self.config.split_manifest),
            "test",
            Path(self.config.data_dir),
            self.config.sequences,
        )
        if self.config.max_cases > 0:
            train_paths = train_paths[: self.config.max_cases]
        if self.config.val_max_cases > 0:
            val_paths = val_paths[: self.config.val_max_cases]

        history = []
        print(f"Found {len(train_paths)} train volumes and {len(val_paths)} validation volumes")
        print(f"Injected LoRA modules: {len(self.injected)}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.network.parameters() if p.requires_grad):,}")
        print(f"Base prompts: {self.num_classes}; prompt templates: {self.num_templates}; variants: {len(self.prompt_variants)}")

        pbar = tqdm(range(self.config.epochs), desc=f"iterations 0/{self.config.epochs}", dynamic_ncols=True)
        running = {}
        since_last_save = 0
        total_updates = 0
        for iteration in pbar:
            path = random.choice(train_paths)
            volume, slicers = self._load_volume_windows(path)
            random.shuffle(slicers)
            if self.config.steps_per_volume > 0:
                slicers = slicers[: self.config.steps_per_volume]
            case_index = train_paths.index(path)
            last_metrics = None

            for patch_idx, slicer in enumerate(slicers):
                patch = torch.clone(volume[slicer][None], memory_format=torch.contiguous_format).to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                amp_enabled = self.config.amp and self.device.type == "cuda"
                with torch.autocast(self.device.type, enabled=amp_enabled):
                    loss, metrics = self._loss(patch)
                loss.backward()
                nn.utils.clip_grad_norm_(list(lora_parameters(self.network)), max_norm=1.0)
                self._optimizer_state_to_device()
                self.optimizer.step()

                total_updates += 1
                since_last_save += 1
                last_metrics = metrics
                record = {
                    "iteration": iteration + 1,
                    "update": total_updates,
                    "case": str(path),
                    "case_index": case_index,
                    "patch_index": patch_idx,
                    "num_patches": len(slicers),
                    "slicer": [
                        [sl.start, sl.stop, sl.step]
                        for sl in slicer
                    ],
                    **metrics,
                }
                history.append(record)
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value

                pbar.set_description(f"iterations {iteration + 1}/{self.config.epochs}")
                pbar.set_postfix({
                    "case": case_index,
                    "patch": f"{patch_idx + 1}/{len(slicers)}",
                    "updates": total_updates,
                    "loss": f"{metrics['loss']:.4f}",
                    "local": f"{metrics['local_entropy']:.4f}",
                    "global": f"{metrics['global_entropy']:.4f}",
                    "fg": f"{metrics['foreground_prior']:.4f}",
                    "src": f"{metrics['source_consistency']:.4f}",
                })
                del patch, loss

            if last_metrics is None:
                continue

            should_validate = (
                self.config.val_label_dir
                and self.config.val_interval > 0
                and (iteration + 1) % self.config.val_interval == 0
            )
            if should_validate:
                interval_metrics = {key: value / max(since_last_save, 1) for key, value in running.items()}
                self.save_checkpoint(
                    save_dir / f"checkpoint_iter_{iteration + 1}.pt",
                    iteration + 1,
                    history,
                    interval_metrics,
                )
                val_metrics = self.validate(val_paths, save_dir / f"val_iter_{iteration + 1}")
                print(
                    f"validation iteration={iteration + 1} "
                    f"mean_dice={val_metrics['mean_dice']:.4f} mean_miou={val_metrics['mean_miou']:.4f}"
                )
                for item in val_metrics["per_prompt"]:
                    print(
                        f"  prompt={item['prompt']} label={item['label']} "
                        f"dice={item['mean_dice']:.4f} miou={item['mean_miou']:.4f}"
                    )
                running = {key: 0.0 for key in running}
                since_last_save = 0

        self.save_checkpoint(save_dir / "checkpoint_final.pt", self.config.epochs, history, {})

    def save(self, path: Path, history: list) -> None:
        torch.save(
            {
                "lora_state_dict": lora_state_dict(self.network),
                "config": asdict(self.config),
                "history": history,
                "injected_modules": sorted(self.injected.keys()),
            },
            path,
        )
        Path(path).with_suffix(".history.json").write_text(json.dumps(history, indent=2))

    def save_checkpoint(self, path: Path, iteration: int, history: list, interval_metrics: dict) -> None:
        torch.save(
            {
                "iteration": iteration,
                "lora_state_dict": lora_state_dict(self.network),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "history": history,
                "interval_metrics": interval_metrics,
                "injected_modules": sorted(self.injected.keys()),
            },
            path,
        )

    def _predict_volume_probabilities(self, image_path: Path) -> Tuple[np.ndarray, dict]:
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

        image, _ = self.reader.read_images([str(image_path)])
        data, bbox, orig_shape = self.predictor.preprocess(image)
        data, slicer_revert_padding = self.pad_nd_image(
            data,
            self.predictor.patch_size,
            "constant",
            {"value": 0},
            True,
            None,
        )
        slicers = self.predictor._internal_get_sliding_window_slicers(data.shape[1:])
        results_device = torch.device("cpu")
        logits = torch.zeros(
            (len(self.prompt_variants), *data.shape[1:]),
            dtype=torch.float16,
            device=results_device,
        )
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.float16, device=results_device)
        gaussian = compute_gaussian(
            tuple(self.predictor.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=results_device,
        )
        stage_weight_sum = None
        stage_weight_count = 0

        self.network.eval()
        with torch.no_grad():
            amp_enabled = self.config.amp and self.device.type == "cuda"
            for slicer in tqdm(slicers, desc=f"predict {image_path.name}", dynamic_ncols=True):
                patch = torch.clone(data[slicer][None], memory_format=torch.contiguous_format).to(self.device)
                with torch.autocast(self.device.type, enabled=amp_enabled):
                    outputs, _ = self._forward_outputs_and_features(patch)
                    patch_logits, stage_weights = self._entropy_weighted_outputs(outputs)
                    if stage_weight_sum is None:
                        stage_weight_sum = stage_weights.float().cpu()
                    else:
                        stage_weight_sum += stage_weights.float().cpu()
                    stage_weight_count += 1
                    if patch_logits.shape[-3:] != tuple(self.predictor.patch_size):
                        patch_logits = F.interpolate(
                            patch_logits,
                            size=tuple(self.predictor.patch_size),
                            mode="trilinear",
                            align_corners=False,
                        )
                prediction = patch_logits[0].to(results_device, dtype=torch.float16)
                prediction *= gaussian
                logits[slicer] += prediction
                n_predictions[slicer[1:]] += gaussian
                del patch, patch_logits, prediction

        torch.div(logits, n_predictions, out=logits)
        logits = logits[(slice(None), *slicer_revert_padding[1:])].to("cpu").float()
        probs = logits.sigmoid().view(self.num_classes, self.num_templates, *logits.shape[1:]).mean(dim=1)
        probs_np = probs.numpy()
        reverted = np.zeros([self.num_classes, *orig_shape], dtype=np.float32)

        reverted = insert_crop_into_image(reverted, probs_np, bbox)
        diagnostics = {
            "prob_min": float(reverted.min()),
            "prob_max": float(reverted.max()),
            "prob_mean": float(reverted.mean()),
            "prob_p50": float(np.percentile(reverted, 50)),
            "prob_p95": float(np.percentile(reverted, 95)),
            "prob_p99": float(np.percentile(reverted, 99)),
        }
        if stage_weight_sum is not None and stage_weight_count > 0:
            diagnostics["stage_weights"] = (stage_weight_sum / stage_weight_count).tolist()
        return reverted, diagnostics

    def validate(self, image_paths: List[Path], output_dir: Path) -> dict:
        self.network.eval()
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = output_dir / "masks"
        if self.config.save_val_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)

        val_paths = list(image_paths)
        if self.config.val_max_cases > 0:
            val_paths = val_paths[: self.config.val_max_cases]

        rows = []
        for image_path in tqdm(val_paths, desc="validation", dynamic_ncols=True):
            label_path = match_label_path(image_path, self.config.val_label_dir)
            label, _ = self.reader.read_seg(str(label_path))
            label = np.asarray(label)
            if label.ndim == 4:
                label = label[0]

            probs, diagnostics = self._predict_volume_probabilities(image_path)
            image_props = None
            if self.config.save_val_masks:
                _, image_props = self.reader.read_images([str(image_path)])
            for class_idx, (prompt, label_value) in enumerate(zip(self.base_prompts, self.label_values)):
                pred = probs[class_idx] >= self.config.pred_threshold
                pred_voxels_raw = int(pred.sum(dtype=np.int64))
                if self.config.largest_component:
                    pred = keep_largest_connected_component(pred)
                gt = label == label_value
                dice, iou = foreground_dice_iou(pred, gt)
                rows.append({
                    "case": image_path.name,
                    "prompt": prompt,
                    "label": int(label_value),
                    "dice": dice,
                    "iou": iou,
                    "pred_voxels": int(pred.sum(dtype=np.int64)),
                    "pred_voxels_raw": pred_voxels_raw,
                    "gt_voxels": int(gt.sum(dtype=np.int64)),
                    **diagnostics,
                })
                if self.config.save_val_masks:
                    mask = pred.astype(np.uint8) * np.uint8(label_value)
                    out_name = f"{image_path.name.replace('.nii.gz', '').replace('.nii', '')}_{safe_name(prompt)}.nii.gz"
                    self.reader.write_seg(mask, str(mask_dir / out_name), image_props)

        per_prompt = []
        for prompt, label_value in zip(self.base_prompts, self.label_values):
            prompt_rows = [row for row in rows if row["prompt"] == prompt and int(row["label"]) == int(label_value)]
            per_prompt.append({
                "prompt": prompt,
                "label": int(label_value),
                "mean_dice": float(np.mean([row["dice"] for row in prompt_rows])) if prompt_rows else 0.0,
                "mean_miou": float(np.mean([row["iou"] for row in prompt_rows])) if prompt_rows else 0.0,
                "num_cases": len(prompt_rows),
            })

        mean_dice = float(np.mean([row["dice"] for row in rows])) if rows else 0.0
        mean_miou = float(np.mean([row["iou"] for row in rows])) if rows else 0.0
        metrics = {
            "mean_dice": mean_dice,
            "mean_miou": mean_miou,
            "per_prompt": per_prompt,
            "rows": rows,
        }
        print("Val: mean_dice {}, mean_miou {}".format(mean_dice, mean_miou))
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics
