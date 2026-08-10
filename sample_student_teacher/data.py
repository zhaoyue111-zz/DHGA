from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_sequences(sequences: str | Iterable[str] | None) -> list[str] | None:
    if sequences is None:
        return None
    if isinstance(sequences, str):
        parts = sequences.replace(",", " ").split()
        return parts or None
    return list(sequences)


def discover_image_paths(data_root: Path, sequences: str | Iterable[str] | None = None) -> list[Path]:
    data_root = data_root.expanduser()
    selected = parse_sequences(sequences)
    if any(data_root.glob("*.nii.gz")) or any(data_root.glob("*.nii")):
        return sorted(data_root.glob("*.nii.gz")) + sorted(data_root.glob("*.nii"))
    image_root = data_root / "images" if (data_root / "images").is_dir() else data_root
    sequence_names = selected or sorted(path.name for path in image_root.iterdir() if path.is_dir())
    paths: list[Path] = []
    for sequence in sequence_names:
        sequence_dir = image_root / sequence
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"Missing image sequence directory: {sequence_dir}")
        paths.extend(sorted(sequence_dir.glob("*.nii.gz")))
        paths.extend(sorted(sequence_dir.glob("*.nii")))
    if not paths:
        raise RuntimeError(f"No NIfTI images found under {image_root}")
    return paths


def load_split_manifest(
    manifest_path: Path,
    split: str,
    data_root: Path,
    sequences: str | Iterable[str] | None = None,
) -> list[Path]:
    with manifest_path.open() as f:
        manifest = json.load(f)
    split_key = split if split in manifest else f"{split}_cases"
    if split_key not in manifest:
        available = ", ".join(sorted(key for key, value in manifest.items() if isinstance(value, list)))
        raise KeyError(f"Split '{split}' not found in {manifest_path}. Available list fields: {available}")
    raw_paths = [Path(value) for value in manifest[split_key]]
    if any(not path.is_absolute() for path in raw_paths):
        case_to_path = {path.name: path for path in discover_image_paths(data_root, sequences)}
        missing = [str(path) for path in raw_paths if path.name not in case_to_path]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
            raise FileNotFoundError(f"Missing cases from split '{split_key}' in {manifest_path}: {preview}{suffix}")
        paths = [case_to_path[path.name] for path in raw_paths]
    else:
        paths = raw_paths
    if not paths:
        raise RuntimeError(f"Split '{split_key}' in {manifest_path} is empty")
    missing_files = [str(path) for path in paths if not path.exists()]
    if missing_files:
        preview = ", ".join(missing_files[:5])
        suffix = "" if len(missing_files) <= 5 else f", ... ({len(missing_files)} missing total)"
        raise FileNotFoundError(f"Missing image files from split '{split_key}' in {manifest_path}: {preview}{suffix}")
    return paths


def match_label_path(image_path: Path, label_dir: str | Path) -> Path:
    label_root = Path(label_dir)
    label_path = label_root / image_path.name
    if label_path.exists():
        return label_path
    if image_path.name.endswith(".nii.gz"):
        alt = label_root / image_path.name[:-3]
    else:
        alt = label_root / f"{image_path.name}.gz"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"No matching label for {image_path.name} under {label_root}")


def spacing_from_reader_properties(props: dict) -> tuple[float, float, float]:
    spacing = props.get("spacing") or props.get("sitk_stuff", {}).get("spacing")
    if spacing is None:
        return (1.0, 1.0, 1.0)
    values = tuple(float(value) for value in spacing)
    return values if len(values) == 3 else (1.0, 1.0, 1.0)


def random_slicer(image_shape: Sequence[int], patch_size: Sequence[int]) -> tuple[slice, ...]:
    starts = []
    for dim, size in zip(image_shape, patch_size):
        starts.append(0 if dim <= size else random.randint(0, dim - size))
    return tuple([slice(None), *[slice(start, start + size) for start, size in zip(starts, patch_size)]])


class DHGADataStrategy:
    """DHGA-compatible image loading, VoxTell preprocessing, padding and patching."""

    def __init__(self, predictor) -> None:
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

        self.predictor = predictor
        self.reader = NibabelIOWithReorient()
        self.pad_nd_image = pad_nd_image

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.predictor.patch_size)

    def load_volume(self, image_path: Path) -> tuple[np.ndarray, dict, tuple[float, float, float]]:
        image, props = self.reader.read_images([str(image_path)])
        spacing = spacing_from_reader_properties(props)
        data, _bbox, _orig_shape = self.predictor.preprocess(image)
        data, _ = self.pad_nd_image(data, self.patch_size, "constant", {"value": 0}, True, None)
        return data, props, spacing

    def sliding_window_slicers(self, data: np.ndarray) -> list[tuple[slice, ...]]:
        return list(self.predictor._internal_get_sliding_window_slicers(data.shape[1:]))

    def training_slicers(self, data: np.ndarray, steps_per_volume: int) -> list[tuple[slice, ...]]:
        if steps_per_volume <= 0:
            slicers = self.sliding_window_slicers(data)
            random.shuffle(slicers)
            return slicers
        return [random_slicer(data.shape[1:], self.patch_size) for _ in range(int(steps_per_volume))]

    def tensor_patch(self, data: np.ndarray, slicer: tuple[slice, ...], device: torch.device) -> Tensor:
        patch = torch.as_tensor(data[slicer][None], device=device, dtype=torch.float32)
        return torch.clone(patch, memory_format=torch.contiguous_format)


def binary_case_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    dice = 2.0 * tp / max(2 * tp + fp + fn, 1)
    fg_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    return {
        "dice": float(dice),
        "miou": float(0.5 * (fg_iou + bg_iou)),
        "iou": float(fg_iou),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "gt_voxels": int(gt.sum()),
        "pred_voxels": int(pred.sum()),
    }
