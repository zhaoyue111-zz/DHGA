from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dhga.config import DHGAConfig
from dhga.geometry import mask_to_sdf
from dhga.inference import finalize_probability
from dhga.voxtell_model import build_dhga_voxtell_model
from voxtell_sfda.adapter import foreground_dice_iou, load_split_manifest, match_label_path


def spacing_from_reader_properties(props: dict) -> tuple[float, float, float]:
    spacing = props.get("spacing") or props.get("sitk_stuff", {}).get("spacing")
    if spacing is None:
        return (1.0, 1.0, 1.0)
    values = tuple(float(v) for v in spacing)
    return values if len(values) == 3 else (1.0, 1.0, 1.0)


class DHGAEvaluator:
    def __init__(
        self,
        config: DHGAConfig,
        prompts: list[str],
        save_dir: str | Path,
        label_dir: str = "",
        label_values: list[int] | None = None,
        model=None,
        predictor=None,
    ) -> None:
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if model is None or predictor is None:
            self.model, self.predictor, self.prompts = build_dhga_voxtell_model(config, prompts)
        else:
            self.model, self.predictor, self.prompts = model, predictor, prompts
        if model is None and (config.init_checkpoint or config.resume_checkpoint):
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
                sem_pred = getattr(self, "last_semantic_pred", pred)
                app_pred = getattr(self, "last_appearance_pred", pred)
                row.update(compute_binary_case_metrics(pred, gt, sem_pred, app_pred))
            rows.append(row)
            out_name = path.name.replace(".nii.gz", "").replace(".nii", "") + "_dhga.nii.gz"
            self.reader.write_seg(pred.astype(np.uint8), str(self.save_dir / out_name), image_props)
            raw_disagreement = getattr(self, "last_raw_disagreement", None)
            if raw_disagreement is not None:
                raw_base = path.name.replace(".nii.gz", "").replace(".nii", "") + "_raw_disagreement"
                np.save(self.save_dir / f"{raw_base}.npy", raw_disagreement.astype(np.float32, copy=False))
                write_float_volume_like_reader(raw_disagreement.astype(np.float32, copy=False), str(self.save_dir / f"{raw_base}.nii.gz"), image_props)
        metrics = {
            "rows": rows,
            "mean_dice": float(np.mean([r["dice"] for r in rows if "dice" in r])) if any("dice" in r for r in rows) else None,
            "mean_iou": float(np.mean([r["iou"] for r in rows if "iou" in r])) if any("iou" in r for r in rows) else None,
            "mean_precision": float(np.mean([r["precision"] for r in rows if "precision" in r])) if any("precision" in r for r in rows) else None,
            "mean_recall": float(np.mean([r["recall"] for r in rows if "recall" in r])) if any("recall" in r for r in rows) else None,
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
                visual = self.model.geometry_visual_proj(out.features.encoder_stages[self.model.geometry_feature_idx])
                visual = F.interpolate(visual, size=target_shape, mode="trilinear", align_corners=False)
                if visual_sum is None:
                    visual_sum = torch.zeros((1, visual.shape[1], *data.shape[1:]), device=self.device)
                visual_sum[(slice(None), slice(None), *slicer[1:])] += visual
            sem_prob = sem_sum / count.clamp_min(1)
            app_prob = app_sum / count.clamp_min(1)
            visual_feature = visual_sum / count.clamp_min(1)
            router = self.model.router(sem_prob, app_prob, visual_context=visual_feature.mean(dim=1, keepdim=True))
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
            if self.config.dhga_geometry_enabled:
                geometry = self.model.run_geometry(data_t,sem_prob,app_prob,router, self.model.text_embeddings, spacing, visual_feature=visual_feature,visual_feature_is_projected=True,)
                phi = mask_to_sdf(router.fused_prob >= self.config.pred_threshold, spacing)
                final = finalize_probability(router.fused_prob,phi,geometry["dense_displacement_mm"],router.w_geo,self.config,geometry.get("dense_valid_weight"),)
            else:
                geometry = None
                final=router.fused_prob
        final = final[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        sem_crop = sem_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        app_crop = app_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        reverted = np.zeros((final.shape[0], *orig_shape), dtype=np.float32)
        reverted = self.insert_crop_into_image(reverted, final, bbox)
        sem_reverted = np.zeros((sem_crop.shape[0], *orig_shape), dtype=np.float32)
        app_reverted = np.zeros((app_crop.shape[0], *orig_shape), dtype=np.float32)
        sem_reverted = self.insert_crop_into_image(sem_reverted, sem_crop, bbox)
        app_reverted = self.insert_crop_into_image(app_reverted, app_crop, bbox)
        self.last_raw_disagreement = np.abs(sem_reverted[0].astype(np.float32) - app_reverted[0].astype(np.float32))
        self.last_semantic_pred = sem_reverted[0] >= self.config.pred_threshold
        self.last_appearance_pred = app_reverted[0] >= self.config.pred_threshold
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


def write_float_volume_like_reader(volume: np.ndarray, output_fname: str, properties: dict) -> None:
    import nibabel
    from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("raw disagreement volume must have shape [D, H, W]")
    restored = volume.transpose((2, 1, 0))
    img = nibabel.Nifti1Image(restored, affine=properties["nibabel_stuff"]["reoriented_affine"])
    img_ornt = io_orientation(properties["nibabel_stuff"]["original_affine"])
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    img_reoriented = img.as_reoriented(from_canonical)
    nibabel.save(img_reoriented, output_fname)


def compute_binary_case_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    semantic_pred: np.ndarray | None = None,
    appearance_pred: np.ndarray | None = None,
) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    dice, iou = foreground_dice_iou(pred, gt)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    metrics: dict[str, float | int] = {
        "dice": dice,
        "iou": iou,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "fp_voxels": fp,
        "fn_voxels": fn,
        "gt_voxels": gt_voxels,
        "pred_gt_volume_ratio": float(pred_voxels / max(gt_voxels, 1)),
        "connected_components": connected_components_3d(pred),
    }
    if semantic_pred is not None and appearance_pred is not None:
        semantic_pred = semantic_pred.astype(bool)
        appearance_pred = appearance_pred.astype(bool)
        oracle_union = np.logical_or(semantic_pred, appearance_pred)
        oracle_intersection = np.logical_and(semantic_pred, appearance_pred)
        union_dice, union_iou = foreground_dice_iou(oracle_union, gt)
        intersection_dice, intersection_iou = foreground_dice_iou(oracle_intersection, gt)
        metrics.update({
            "oracle_union_dice": union_dice,
            "oracle_union_iou": union_iou,
            "oracle_intersection_dice": intersection_dice,
            "oracle_intersection_iou": intersection_iou,
        })
    return metrics
