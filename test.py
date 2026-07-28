from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dhga.config import DHGAConfig
from dhga.evaluation import DHGAEvaluator, compute_binary_case_metrics
from voxtell_sfda.adapter import load_split_manifest, match_label_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DHGA checkpoint test with largest-connected-component postprocess")
    parser.add_argument("--checkpoint", required=True, help="DHGA training checkpoint, e.g. checkpoint_final.pt")
    parser.add_argument("--save_dir", default=".save/dhga/test_lcc", help="Directory for metrics and optional masks")
    parser.add_argument("--voxtell_repo", default="/data/zy/VoxTell_from_disk")
    parser.add_argument("--model_dir", default="/data/zy/VoxTell_from_disk/model")
    parser.add_argument("--data_dir", default="/data/zy/CT_MRI_DATA_3D/images/P0")
    parser.add_argument("--split_manifest", default="/data/zy/DHGA/worst_zeroshot_split_p0/worst_zeroshot_split.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--sequences", default="P0")
    parser.add_argument("--val_label_dir", default="/data/zy/CT_MRI_DATA_3D/labels/P0")
    parser.add_argument("--label_values", nargs="*", type=int, default=[5])
    parser.add_argument("--prompts", nargs="*", default=["liver"])
    parser.add_argument("--prompt_templates", nargs="*", default=["{}"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save_masks", action="store_true", help="Save LCC masks as NIfTI")
    return parser.parse_args()


def config_from_checkpoint(args: argparse.Namespace) -> DHGAConfig:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    values = dict(payload.get("config", {}))
    if not values:
        values = DHGAConfig().to_dict()
    values.update(
        {
            "voxtell_repo": args.voxtell_repo,
            "model_dir": args.model_dir,
            "data_dir": args.data_dir,
            "split_manifest": args.split_manifest,
            "sequences": args.sequences,
            "prompt_templates": list(args.prompt_templates),
            "device": args.device,
            "max_cases": args.max_cases,
            "val_label_dir": args.val_label_dir,
            "label_values": list(args.label_values or [1]),
            "init_checkpoint": str(args.checkpoint),
            "resume_checkpoint": "",
            "tensorboard_enabled": False,
        }
    )
    if args.threshold is not None:
        values["pred_threshold"] = float(args.threshold)
    return DHGAConfig.from_mapping(values)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    try:
        from scipy import ndimage

        labeled, count = ndimage.label(mask)
        if count <= 1:
            return mask
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        return labeled == int(sizes.argmax())
    except Exception:
        return _largest_connected_component_fallback(mask)


def _largest_connected_component_fallback(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int, int]] = []
    for start in np.argwhere(mask):
        z, y, x = (int(v) for v in start)
        if visited[z, y, x]:
            continue
        component: list[tuple[int, int, int]] = []
        queue: deque[tuple[int, int, int]] = deque([(z, y, x)])
        visited[z, y, x] = True
        while queue:
            cz, cy, cx = queue.popleft()
            component.append((cz, cy, cx))
            for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if (
                    0 <= nz < mask.shape[0]
                    and 0 <= ny < mask.shape[1]
                    and 0 <= nx < mask.shape[2]
                    and mask[nz, ny, nx]
                    and not visited[nz, ny, nx]
                ):
                    visited[nz, ny, nx] = True
                    queue.append((nz, ny, nx))
        if len(component) > len(best):
            best = component
    out = np.zeros(mask.shape, dtype=bool)
    for z, y, x in best:
        out[z, y, x] = True
    return out


def main() -> None:
    args = parse_args()
    config = config_from_checkpoint(args)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    evaluator = DHGAEvaluator(config, args.prompts, save_dir, args.val_label_dir, args.label_values)
    paths = load_split_manifest(Path(config.split_manifest), args.split, Path(config.data_dir), config.sequences)
    if args.max_cases > 0:
        paths = paths[: args.max_cases]

    rows = []
    for path in tqdm(paths, desc=f"DHGA {args.split} LCC test", dynamic_ncols=True):
        probs, image_props = evaluator.predict_case(path)
        raw_pred = probs[0] >= config.pred_threshold
        pred = largest_connected_component(raw_pred)

        label_path = match_label_path(path, args.val_label_dir)
        label, _ = evaluator.reader.read_seg(str(label_path))
        label = np.asarray(label[0] if label.ndim == 4 else label)
        gt = label == int(args.label_values[0])

        metrics = compute_binary_case_metrics(pred, gt)
        row = {
            "case": path.name,
            "dice": metrics["dice"],
            "miou": metrics["iou"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "raw_pred_voxels": int(raw_pred.sum()),
            "lcc_pred_voxels": int(pred.sum()),
            "gt_voxels": int(gt.sum()),
        }
        rows.append(row)
        if args.save_masks:
            out_name = path.name.replace(".nii.gz", "").replace(".nii", "") + "_dhga_lcc.nii.gz"
            evaluator.reader.write_seg(pred.astype(np.uint8), str(save_dir / out_name), image_props)

    summary = {"num_cases": len(rows), "rows": rows}
    for key in ("dice", "miou", "recall", "precision"):
        values = [float(row[key]) for row in rows]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
        summary[f"median_{key}"] = float(np.median(values)) if values else None

    (save_dir / "metrics_lcc.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
