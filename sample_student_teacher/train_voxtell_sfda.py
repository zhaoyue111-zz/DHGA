from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import DHGADataStrategy, binary_case_metrics, load_split_manifest, match_label_path, set_seed
from .model import StudentTeacherLoRA
from .train_step import train_one_step
from .voxtell import SimpleVoxTellSegmenter


@dataclass
class TrainConfig:
    voxtell_repo: str = "/data/zy/VoxTell_from_disk"
    model_dir: str = "/data/zy/VoxTell_from_disk/model"
    data_dir: str = "/data/zy/CT_MRI_DATA_3D/images/P0"
    split_manifest: str = "/data/zy/DHGA/worst_zeroshot_split_p0/worst_zeroshot_split.json"
    sequences: str = "P0"
    val_label_dir: str = "/data/zy/CT_MRI_DATA_3D/labels/P0"
    label_values: list[int] = field(default_factory=lambda: [5])
    prompts: list[str] = field(default_factory=lambda: ["liver"])
    prompt_templates: list[str] = field(default_factory=lambda: ["{}"])
    text_encoding_model: str = "Qwen/Qwen3-Embedding-4B"
    device: str = "cuda"
    save_dir: str = ".save/simple_student_teacher"
    epochs: int = 1
    steps_per_volume: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    ema_decay: float = 0.999
    amp: bool = True
    seed: int = 0
    max_cases: int = 0
    eval_interval: int = 1
    eval_split: str = "test"
    eval_max_cases: int = 0
    pred_threshold: float = 0.5
    save_masks: bool = False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser("Minimal VoxTell student-teacher LoRA SFDA")
    defaults = TrainConfig()
    parser.add_argument("--voxtell_repo", default=defaults.voxtell_repo)
    parser.add_argument("--model_dir", default=defaults.model_dir)
    parser.add_argument("--data_dir", default=defaults.data_dir)
    parser.add_argument("--split_manifest", default=defaults.split_manifest)
    parser.add_argument("--sequences", default=defaults.sequences)
    parser.add_argument("--val_label_dir", default=defaults.val_label_dir)
    parser.add_argument("--label_values", nargs="*", type=int, default=defaults.label_values)
    parser.add_argument("--prompts", nargs="*", default=defaults.prompts)
    parser.add_argument("--prompt_templates", nargs="*", default=defaults.prompt_templates)
    parser.add_argument("--text_encoding_model", default=defaults.text_encoding_model)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--save_dir", default=defaults.save_dir)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--steps_per_volume", type=int, default=defaults.steps_per_volume)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--lora_rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--lora_alpha", type=float, default=defaults.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=defaults.lora_dropout)
    parser.add_argument("--ema_decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=defaults.amp)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max_cases", type=int, default=defaults.max_cases)
    parser.add_argument("--eval_interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--eval_split", choices=("train", "val", "test"), default=defaults.eval_split)
    parser.add_argument("--eval_max_cases", type=int, default=defaults.eval_max_cases)
    parser.add_argument("--pred_threshold", type=float, default=defaults.pred_threshold)
    parser.add_argument("--save_masks", action="store_true")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def build_model_and_data(config: TrainConfig) -> tuple[StudentTeacherLoRA, object, DHGADataStrategy, torch.device]:
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    segmenter, predictor, _prompts = SimpleVoxTellSegmenter.from_voxtell(
        config.voxtell_repo,
        config.model_dir,
        config.prompts,
        config.prompt_templates,
        config.text_encoding_model,
        device,
    )
    model = StudentTeacherLoRA(
        segmenter,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        ema_decay=config.ema_decay,
    ).to(device)
    data = DHGADataStrategy(predictor)
    return model, predictor, data, device


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    model, predictor, data, device = build_model_and_data(config)
    train_paths = load_split_manifest(Path(config.split_manifest), "train", Path(config.data_dir), config.sequences)
    if config.max_cases > 0:
        train_paths = train_paths[: config.max_cases]
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    history: list[dict] = []
    for epoch in range(config.epochs):
        random.shuffle(train_paths)
        model.train()
        for path in tqdm(train_paths, desc=f"epoch {epoch + 1}/{config.epochs}", dynamic_ncols=True):
            volume, _props, _spacing = data.load_volume(path)
            for slicer in data.training_slicers(volume, config.steps_per_volume):
                patch = data.tensor_patch(volume, slicer, device)
                metrics = train_one_step(model, optimizer, patch, amp=config.amp)
                record = {"epoch": epoch + 1, "case": path.name, **metrics}
                history.append(record)
        torch.save(
            {
                "format": "simple_voxtell_lora_sfda_v1",
                "config": asdict(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "history": history[-256:],
            },
            save_dir / "checkpoint_latest.pt",
        )
        if config.eval_interval > 0 and (epoch + 1) % config.eval_interval == 0:
            metrics = evaluate(config, model=model, predictor=predictor, data=data, device=device, split=config.eval_split)
            (save_dir / f"metrics_epoch_{epoch + 1:04d}.json").write_text(json.dumps(metrics, indent=2))
    (save_dir / "history.json").write_text(json.dumps(history, indent=2))


def predict_case(
    model: StudentTeacherLoRA,
    predictor,
    data: DHGADataStrategy,
    image_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

    image, props = data.reader.read_images([str(image_path)])
    preprocessed, bbox, orig_shape = predictor.preprocess(image)
    padded, slicer_revert_padding = data.pad_nd_image(preprocessed, data.patch_size, "constant", {"value": 0}, True, None)
    slicers = predictor._internal_get_sliding_window_slicers(padded.shape[1:])
    volume = torch.as_tensor(padded[None], device=device, dtype=torch.float32)
    prob_sum = torch.zeros((1, 1, *padded.shape[1:]), device=device)
    count = torch.zeros_like(prob_sum)
    model.eval()
    with torch.inference_mode():
        for slicer in slicers:
            patch = torch.clone(volume[(slice(None), *slicer)], memory_format=torch.contiguous_format)
            logits = model(patch)
            prob = logits.float().sigmoid()
            if tuple(prob.shape[-3:]) != tuple(patch.shape[-3:]):
                prob = F.interpolate(prob, size=patch.shape[-3:], mode="trilinear", align_corners=False)
            prob_sum[(slice(None), slice(None), *slicer[1:])] += prob
            count[(slice(None), slice(None), *slicer[1:])] += 1
    prob = prob_sum / count.clamp_min(1)
    crop = prob[(slice(None), slice(None), *slicer_revert_padding[1:])].detach().cpu().numpy()[0]
    restored = np.zeros((crop.shape[0], *orig_shape), dtype=np.float32)
    restored = insert_crop_into_image(restored, crop, bbox)
    return restored, props


def evaluate(
    config: TrainConfig,
    model: StudentTeacherLoRA | None = None,
    predictor=None,
    data: DHGADataStrategy | None = None,
    device: torch.device | None = None,
    split: str = "test",
) -> dict:
    if model is None or predictor is None or data is None or device is None:
        model, predictor, data, device = build_model_and_data(config)
        checkpoint = Path(config.save_dir) / "checkpoint_latest.pt"
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
    paths = load_split_manifest(Path(config.split_manifest), split, Path(config.data_dir), config.sequences)
    if config.eval_max_cases > 0:
        paths = paths[: config.eval_max_cases]
    save_dir = Path(config.save_dir) / f"eval_{split}"
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in tqdm(paths, desc=f"eval {split}", dynamic_ncols=True):
        prob, props = predict_case(model, predictor, data, image_path, device)
        pred = prob[0] >= config.pred_threshold
        row = {"case": image_path.name, "pred_voxels": int(pred.sum())}
        if config.val_label_dir:
            label_path = match_label_path(image_path, config.val_label_dir)
            label, _ = data.reader.read_seg(str(label_path))
            label = np.asarray(label[0] if label.ndim == 4 else label)
            gt = label == int(config.label_values[0])
            row.update(binary_case_metrics(pred, gt))
        rows.append(row)
        if config.save_masks:
            out_name = image_path.name.replace(".nii.gz", "").replace(".nii", "") + "_simple.nii.gz"
            data.reader.write_seg(pred.astype(np.uint8), str(save_dir / out_name), props)
    summary = {"rows": rows}
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.integer, np.floating))})
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    (save_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
