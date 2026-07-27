from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dhga.config import DHGAConfig
from dhga.voxtell_model import build_dhga_voxtell_model
from voxtell_sfda.adapter import foreground_dice_iou, load_split_manifest, match_label_path


def spacing_from_reader_properties(props: dict) -> tuple[float, float, float]:
    spacing = props.get("spacing") or props.get("sitk_stuff", {}).get("spacing")
    if spacing is None:
        return (1.0, 1.0, 1.0)
    values = tuple(float(v) for v in spacing)
    return values if len(values) == 3 else (1.0, 1.0, 1.0)


class DHGAEvaluator:
    def __init__(self, config: DHGAConfig, prompts: list[str], save_dir: str | Path, label_dir: str = "", label_values: list[int] | None = None) -> None:
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.predictor, self.prompts = build_dhga_voxtell_model(config, prompts)
        if config.init_checkpoint or config.resume_checkpoint:
            from dhga.checkpoint import load_training_checkpoint

            load_training_checkpoint(config.resume_checkpoint or config.init_checkpoint, self.model, load_training_state=False)
        self.device = next(self.model.parameters()).device
        self.label_dir = label_dir
        self.label_values = label_values or [1]
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

        self.reader = NibabelIOWithReorient()
        self.pad_nd_image = pad_nd_image
        self.insert_crop_into_image = insert_crop_into_image

    def evaluate_split(self, split: str = "test", max_cases: int = 0) -> dict:
        paths = load_split_manifest(Path(self.config.split_manifest), split, Path(self.config.data_dir), self.config.sequences)
        if max_cases > 0:
            paths = paths[:max_cases]
        rows = []
        for path in tqdm(paths, desc="DHGA evaluation-only", dynamic_ncols=True):
            probs, image_props = self.predict_case(path)
            pred = probs[0] >= self.config.pred_threshold
            row = {"case": path.name, "pred_voxels": int(pred.sum()), **getattr(self, "last_case_diagnostics", {})}
            if self.label_dir:
                label_path = match_label_path(path, self.label_dir)
                label, _ = self.reader.read_seg(str(label_path))
                label = np.asarray(label[0] if label.ndim == 4 else label)
                gt = label == int(self.label_values[0])
                dice, iou = foreground_dice_iou(pred, gt)
                sem_pred = getattr(self, "last_semantic_pred", pred)
                app_pred = getattr(self, "last_appearance_pred", pred)
                oracle_union = np.logical_or(sem_pred, app_pred)
                oracle_intersection = np.logical_and(sem_pred, app_pred)
                oracle_union_dice, oracle_union_iou = foreground_dice_iou(oracle_union, gt)
                oracle_intersection_dice, oracle_intersection_iou = foreground_dice_iou(oracle_intersection, gt)
                tp = int(np.logical_and(pred, gt).sum())
                fp = int(np.logical_and(pred, ~gt).sum())
                fn = int(np.logical_and(~pred, gt).sum())
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                gt_voxels = int(gt.sum())
                pred_voxels = int(pred.sum())
                row.update({
                    "dice": dice,
                    "iou": iou,
                    "precision": float(precision),
                    "recall": float(recall),
                    "fp_voxels": fp,
                    "fn_voxels": fn,
                    "gt_voxels": gt_voxels,
                    "pred_gt_volume_ratio": float(pred_voxels / max(gt_voxels, 1)),
                    "connected_components": connected_components_3d(pred),
                    "oracle_union_dice": oracle_union_dice,
                    "oracle_union_iou": oracle_union_iou,
                    "oracle_intersection_dice": oracle_intersection_dice,
                    "oracle_intersection_iou": oracle_intersection_iou,
                })
            rows.append(row)
            out_name = path.name.replace(".nii.gz", "").replace(".nii", "") + "_dhga.nii.gz"
            self.reader.write_seg(pred.astype(np.uint8), str(self.save_dir / out_name), image_props)
        metrics = {
            "rows": rows,
            "mean_dice": float(np.mean([r["dice"] for r in rows if "dice" in r])) if any("dice" in r for r in rows) else None,
        }
        (self.save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    def predict_case(self, image_path: Path) -> tuple[np.ndarray, dict]:
        image, props = self.reader.read_images([str(image_path)])
        spacing = spacing_from_reader_properties(props)
        data, bbox, orig_shape = self.predictor.preprocess(image)
        data, slicer_revert_padding = self.pad_nd_image(data, tuple(self.predictor.patch_size), "constant", {"value": 0}, True, None)
        slicers = self.predictor._internal_get_sliding_window_slicers(data.shape[1:])
        data_t = torch.as_tensor(data[None], device=self.device, dtype=torch.float32)
        sem_sum = torch.zeros((1, 1, *data.shape[1:]), device=self.device)
        app_sum = torch.zeros_like(sem_sum)
        count = torch.zeros_like(sem_sum)
        visual_sum = None
        self.model.eval()
        with torch.no_grad():
            for slicer in slicers:
                patch = torch.clone(data_t[(slice(None), *slicer)], memory_format=torch.contiguous_format)
                out = self.model(patch, spacing, run_geometry=False)
                target_shape = patch.shape[-3:]
                sem = F.interpolate(out.semantic_prob, size=target_shape, mode="trilinear", align_corners=False)
                app = F.interpolate(out.appearance_prob, size=target_shape, mode="trilinear", align_corners=False)
                sem_sum[(slice(None), slice(None), *slicer[1:])] += sem
                app_sum[(slice(None), slice(None), *slicer[1:])] += app
                count[(slice(None), slice(None), *slicer[1:])] += 1
                visual = out.features.encoder_stages[self.model.geometry_feature_idx]
                visual = F.interpolate(visual, size=target_shape, mode="trilinear", align_corners=False)
                if visual_sum is None:
                    visual_sum = torch.zeros((1, visual.shape[1], *data.shape[1:]), device=self.device)
                visual_sum[(slice(None), slice(None), *slicer[1:])] += visual
            sem_prob = sem_sum / count.clamp_min(1)
            app_prob = app_sum / count.clamp_min(1)
            visual_feature = visual_sum / count.clamp_min(1)
            router = self.model.router(sem_prob, app_prob)
            self.last_semantic_pred = (sem_prob >= self.config.pred_threshold).cpu().numpy()[0, 0]
            self.last_appearance_pred = (app_prob >= self.config.pred_threshold).cpu().numpy()[0, 0]
            sem_mask = sem_prob >= self.config.pred_threshold
            app_mask = app_prob >= self.config.pred_threshold
            sem_only = sem_mask & ~app_mask
            app_only = app_mask & ~sem_mask
            union = sem_mask | app_mask
            self.last_case_diagnostics = {
                "semantic_appearance_complement_rate": float((sem_only | app_only).float().sum().cpu() / union.float().sum().clamp_min(1.0).cpu()),
                "semantic_only_voxels": int(sem_only.sum().cpu()),
                "appearance_only_voxels": int(app_only.sum().cpu()),
                "disagreement_mean": float(router.disagreement.mean().cpu()),
                "disagreement_voxel_rate": float((router.disagreement > 0.5).float().mean().cpu()),
                "geometry_gate_mean": float(router.w_geo.mean().cpu()),
            }
            geometry = self.model.run_geometry(
                data_t,
                sem_prob,
                app_prob,
                router,
                self.model.text_embeddings,
                spacing,
                visual_feature=visual_feature,
            )
            from dhga.geometry import mask_to_sdf

            phi = mask_to_sdf(router.fused_prob >= self.config.pred_threshold, spacing)
            geo_prob = ((phi - geometry["dense_displacement_mm"]) / max(self.config.dhga_ray_step_mm, 1e-6)).neg().sigmoid()
            gate = ((phi.abs() <= self.config.dhga_search_radius_mm) * router.w_geo).clamp(0, 1)
            final = router.fused_prob * (1.0 - gate) + geo_prob * gate
        final = final[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        reverted = np.zeros((final.shape[0], *orig_shape), dtype=np.float32)
        reverted = self.insert_crop_into_image(reverted, final, bbox)
        return reverted, props


def connected_components_3d(mask: np.ndarray) -> int:
    try:
        from scipy import ndimage

        _, count = ndimage.label(mask.astype(bool))
        return int(count)
    except Exception:
        mask = mask.astype(bool)
        visited = np.zeros(mask.shape, dtype=bool)
        count = 0
        for start in np.argwhere(mask):
            z, y, x = (int(v) for v in start)
            if visited[z, y, x]:
                continue
            count += 1
            stack = [(z, y, x)]
            visited[z, y, x] = True
            while stack:
                cz, cy, cx = stack.pop()
                for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    nz, ny, nx = cz + dz, cy + dy, cx + dx
                    if 0 <= nz < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nx < mask.shape[2] and mask[nz, ny, nx] and not visited[nz, ny, nx]:
                        visited[nz, ny, nx] = True
                        stack.append((nz, ny, nx))
        return count
