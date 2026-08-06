#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
from typing import Any

import numpy as np

from voxtell_sfda.adapter import (
    foreground_dice_iou,
    keep_largest_connected_component,
    match_label_path,
)

METRIC_NAMES = ("dice", "miou", "precision", "recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only the largest 3D connected component for every *_dhga.nii.gz "
            "prediction in an evaluation directory, recompute binary segmentation "
            "metrics, and compare them with the original metrics.json."
        )
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        required=True,
        help="Evaluation directory containing metrics.json and *_dhga.nii.gz files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "DHGA config JSON. Default: <eval_dir>/../dhga_config.json. "
            "Used to obtain val_label_dir and label_values."
        ),
    )
    parser.add_argument(
        "--label_dir",
        type=Path,
        default=None,
        help="Override the label directory from dhga_config.json.",
    )
    parser.add_argument(
        "--label_value",
        type=int,
        default=None,
        help="Override the foreground label value from dhga_config.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<eval_dir>/largest_component_postprocess."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*_dhga.nii.gz",
        help="Prediction glob pattern. Default: *_dhga.nii.gz",
    )
    parser.add_argument(
        "--consistency_tol",
        type=float,
        default=1e-5,
        help=(
            "Maximum tolerated difference between metrics.json and metrics "
            "recomputed from the saved original mask. Default: 1e-5."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def case_name_from_prediction(prediction_name: str) -> str:
    suffix = "_dhga.nii.gz"
    if not prediction_name.endswith(suffix):
        raise ValueError(
            f"Prediction name does not end with {suffix!r}: {prediction_name}"
        )
    return prediction_name[: -len(suffix)] + ".nii.gz"


def as_3d(array: np.ndarray, path: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D mask or [1,D,H,W] array for {path}, got {array.shape}"
        )
    return array


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: pred={pred.shape}, gt={gt.shape}"
        )

    dice, iou = foreground_dice_iou(pred, gt)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    return {
        "dice": float(dice),
        "miou": float(iou),  # single foreground class: project iou == mIoU
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "pred_voxels": int(pred.sum()),
        "gt_voxels": int(gt.sum()),
    }


def original_row_for_case(
    rows_by_case: dict[str, dict[str, Any]], case_name: str
) -> dict[str, Any]:
    candidates = (
        case_name,
        strip_nii_suffix(case_name),
        strip_nii_suffix(case_name) + ".nii",
    )
    for candidate in candidates:
        row = rows_by_case.get(candidate)
        if row is not None:
            return row
    raise KeyError(
        f"No row for {case_name} in metrics.json. Tried: {candidates}"
    )


def original_metric_from_row(
    row: dict[str, Any],
    metric: str,
    recomputed: dict[str, float | int],
) -> float:
    source_key = "iou" if metric == "miou" else metric
    value = row.get(source_key)
    if isinstance(value, (int, float)):
        return float(value)
    return float(recomputed[metric])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_xlsx(
    path: Path,
    per_case_rows: list[dict[str, Any]],
    mean_rows: list[dict[str, Any]],
) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False

    try:
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame(per_case_rows).to_excel(
                writer, sheet_name="per_case", index=False
            )
            pd.DataFrame(mean_rows).to_excel(
                writer, sheet_name="mean", index=False
            )
    except ImportError:
        return False
    return True


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    metrics_path = eval_dir / "metrics.json"
    config_path = (
        args.config.resolve()
        if args.config is not None
        else (eval_dir.parent / "dhga_config.json").resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (eval_dir / "largest_component_postprocess").resolve()
    )
    mask_output_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = load_json(metrics_path)
    config = load_json(config_path)

    original_rows = metrics_json.get("rows")
    if not isinstance(original_rows, list):
        raise ValueError(f"{metrics_path} does not contain a list field named 'rows'")

    rows_by_case: dict[str, dict[str, Any]] = {}
    for row in original_rows:
        if not isinstance(row, dict) or "case" not in row:
            continue
        case = str(row["case"])
        rows_by_case[case] = row
        rows_by_case[strip_nii_suffix(case)] = row

    label_dir_value = args.label_dir or config.get("val_label_dir")
    if not label_dir_value:
        raise ValueError(
            "No label directory supplied and val_label_dir is absent from the config."
        )
    label_dir = Path(label_dir_value).resolve()
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    if args.label_value is not None:
        label_value = int(args.label_value)
    else:
        label_values = config.get("label_values", [1])
        if not isinstance(label_values, list) or not label_values:
            raise ValueError("label_values in the config must be a non-empty list")
        label_value = int(label_values[0])

    prediction_paths = sorted(eval_dir.glob(args.pattern))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No prediction files matching {args.pattern!r} under {eval_dir}"
        )

    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

    reader = NibabelIOWithReorient()
    comparison_rows: list[dict[str, Any]] = []
    consistency_failures: list[str] = []

    for prediction_path in prediction_paths:
        case_name = case_name_from_prediction(prediction_path.name)
        original_row = original_row_for_case(rows_by_case, case_name)

        prediction_array, prediction_props = reader.read_seg(str(prediction_path))
        pred = as_3d(prediction_array, prediction_path) > 0

        image_case_path = Path(case_name)
        label_path = match_label_path(image_case_path, label_dir)
        label_array, _ = reader.read_seg(str(label_path))
        gt_labels = as_3d(label_array, label_path)
        gt = gt_labels == label_value

        if pred.shape != gt.shape:
            raise ValueError(
                f"{case_name}: prediction shape {pred.shape} != GT shape {gt.shape}"
            )

        recomputed_original = binary_metrics(pred, gt)
        largest = keep_largest_connected_component(pred).astype(bool)
        postprocessed = binary_metrics(largest, gt)

        original_values = {
            metric: original_metric_from_row(
                original_row, metric, recomputed_original
            )
            for metric in METRIC_NAMES
        }

        metric_diffs = {
            metric: abs(
                original_values[metric] - float(recomputed_original[metric])
            )
            for metric in METRIC_NAMES
        }
        max_consistency_diff = max(metric_diffs.values())
        consistent = max_consistency_diff <= float(args.consistency_tol)
        if not consistent:
            consistency_failures.append(
                f"{case_name}: max metric difference={max_consistency_diff:.8g}"
            )

        original_components = original_row.get("connected_components")
        if not isinstance(original_components, (int, float)):
            from scipy import ndimage

            _, original_components = ndimage.label(
                pred,
                structure=np.ones((3, 3, 3), dtype=np.uint8),
            )

        kept_voxels = int(largest.sum())
        removed_voxels = int(pred.sum()) - kept_voxels

        row: dict[str, Any] = {
            "case": case_name,
            "prediction_file": str(prediction_path),
            "label_file": str(label_path),
            "original_components": int(original_components),
            "lcc_components": int(1 if largest.any() else 0),
            "original_pred_voxels": int(pred.sum()),
            "lcc_pred_voxels": kept_voxels,
            "removed_voxels": removed_voxels,
            "removed_ratio": float(removed_voxels / max(int(pred.sum()), 1)),
            "eval_mask_consistent": bool(consistent),
            "eval_vs_recomputed_max_abs_diff": float(max_consistency_diff),
        }

        for metric in METRIC_NAMES:
            original_value = original_values[metric]
            lcc_value = float(postprocessed[metric])
            row[f"original_{metric}"] = original_value
            row[f"lcc_{metric}"] = lcc_value
            row[f"delta_{metric}"] = lcc_value - original_value

        for key in ("tp_voxels", "fp_voxels", "fn_voxels"):
            row[f"original_{key}"] = int(recomputed_original[key])
            row[f"lcc_{key}"] = int(postprocessed[key])
            row[f"delta_{key}"] = int(postprocessed[key]) - int(
                recomputed_original[key]
            )

        output_mask_path = mask_output_dir / prediction_path.name
        reader.write_seg(
            largest.astype(np.uint8),
            str(output_mask_path),
            prediction_props,
        )
        row["lcc_mask_file"] = str(output_mask_path)
        comparison_rows.append(row)

    # Match the evaluator's aggregation: arithmetic mean across cases (macro average).
    mean_rows: list[dict[str, Any]] = []
    for metric in METRIC_NAMES:
        original_mean = float(
            np.mean([float(row[f"original_{metric}"]) for row in comparison_rows])
        )
        lcc_mean = float(
            np.mean([float(row[f"lcc_{metric}"]) for row in comparison_rows])
        )
        json_key = "mean_iou" if metric == "miou" else f"mean_{metric}"
        reported_mean = metrics_json.get(json_key)
        mean_rows.append(
            {
                "metric": metric,
                "original_macro_mean": original_mean,
                "metrics_json_reported_mean": (
                    float(reported_mean)
                    if isinstance(reported_mean, (int, float))
                    else None
                ),
                "lcc_macro_mean": lcc_mean,
                "delta": lcc_mean - original_mean,
            }
        )

    per_case_fields = [
        "case",
        "original_components",
        "lcc_components",
        "original_pred_voxels",
        "lcc_pred_voxels",
        "removed_voxels",
        "removed_ratio",
        "original_dice",
        "lcc_dice",
        "delta_dice",
        "original_miou",
        "lcc_miou",
        "delta_miou",
        "original_precision",
        "lcc_precision",
        "delta_precision",
        "original_recall",
        "lcc_recall",
        "delta_recall",
        "original_tp_voxels",
        "lcc_tp_voxels",
        "delta_tp_voxels",
        "original_fp_voxels",
        "lcc_fp_voxels",
        "delta_fp_voxels",
        "original_fn_voxels",
        "lcc_fn_voxels",
        "delta_fn_voxels",
        "eval_mask_consistent",
        "eval_vs_recomputed_max_abs_diff",
        "prediction_file",
        "label_file",
        "lcc_mask_file",
    ]
    mean_fields = [
        "metric",
        "original_macro_mean",
        "metrics_json_reported_mean",
        "lcc_macro_mean",
        "delta",
    ]

    per_case_csv = output_dir / "largest_component_per_case.csv"
    mean_csv = output_dir / "largest_component_mean.csv"
    write_csv(per_case_csv, comparison_rows, per_case_fields)
    write_csv(mean_csv, mean_rows, mean_fields)

    result_json = {
        "eval_dir": str(eval_dir),
        "metrics_json": str(metrics_path),
        "config": str(config_path),
        "label_dir": str(label_dir),
        "label_value": label_value,
        "connectivity": (
            "26-neighbour 3D connectivity, inherited from "
            "voxtell_sfda.adapter.keep_largest_connected_component"
        ),
        "num_cases": len(comparison_rows),
        "rows": comparison_rows,
        "mean": mean_rows,
        "consistency_failures": consistency_failures,
    }
    result_json_path = output_dir / "largest_component_comparison.json"
    result_json_path.write_text(
        json.dumps(result_json, indent=2, ensure_ascii=False)
    )

    xlsx_path = output_dir / "largest_component_comparison.xlsx"
    xlsx_written = maybe_write_xlsx(xlsx_path, comparison_rows, mean_rows)

    print(f"Processed {len(comparison_rows)} cases.")
    print(f"LCC masks: {mask_output_dir}")
    print(f"Per-case CSV: {per_case_csv}")
    print(f"Mean CSV: {mean_csv}")
    print(f"JSON: {result_json_path}")
    if xlsx_written:
        print(f"Excel workbook: {xlsx_path}")
    else:
        print(
            "Excel workbook was not written because pandas/openpyxl is unavailable; "
            "the CSV files contain the full results."
        )
    if consistency_failures:
        print(
            f"WARNING: {len(consistency_failures)} case(s) did not reproduce "
            f"metrics.json within tolerance {args.consistency_tol}:"
        )
        for message in consistency_failures:
            print(f"  - {message}")


if __name__ == "__main__":
    main()