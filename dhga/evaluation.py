from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dhga.config import DHGAConfig
from dhga.geometry import mask_to_sdf
from dhga.inference import finalize_probability, geometry_effective_gate
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
                fused_pred = getattr(self, "last_fused_pred", pred)
                row.update(compute_binary_case_metrics(pred, gt, sem_pred, app_pred))
                simple_pred = getattr(self, "last_simple_average_pred", None)
                zero_pred = getattr(self, "last_zero_shot_pred", None)
                if simple_pred is not None:
                    simple_base = _binary_metric_values(simple_pred, gt)
                    row["simple_average_dice"] = simple_base["dice"]
                if zero_pred is not None:
                    zero_base = _binary_metric_values(zero_pred, gt)
                    row["zero_shot_dice"] = zero_base["dice"]
                row["prototype_fused_dice"] = row.get("dice", 0.0)
                row.update(compute_geometry_case_metrics(fused_pred, pred, gt, spacing_from_reader_properties(image_props), self.config.dhga_surface_tolerance_mm))
                raw_disagreement = getattr(self, "last_raw_disagreement", None)
                if raw_disagreement is not None:
                    row.update(
                        compute_raw_disagreement_metrics(
                            raw_disagreement,
                            semantic_pred=sem_pred,
                            appearance_pred=app_pred,
                            gt=gt,
                            spacing=spacing_from_reader_properties(image_props),
                        )
                    )
            rows.append(row)
            out_name = path.name.replace(".nii.gz", "").replace(".nii", "") + "_dhga.nii.gz"
            self.reader.write_seg(pred.astype(np.uint8), str(self.save_dir / out_name), image_props)
            raw_disagreement = getattr(self, "last_raw_disagreement", None)
            if raw_disagreement is not None:
                raw_base = path.name.replace(".nii.gz", "").replace(".nii", "") + "_raw_disagreement"
                np.save(self.save_dir / f"{raw_base}.npy", raw_disagreement.astype(np.float32, copy=False))
                write_float_volume_like_reader(raw_disagreement.astype(np.float32, copy=False), str(self.save_dir / f"{raw_base}.nii.gz"), image_props)
        metrics = {"rows": rows}
        numeric_keys = sorted({
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        })
        for key in numeric_keys:
            values = [float(row[key]) for row in rows if key in row]
            metrics[f"mean_{key}"] = float(np.mean(values)) if values else None
        (self.save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    def predict_case(self, image_path: Path) -> tuple[np.ndarray, dict]:
        image, props = self.reader.read_images([str(image_path)])
        spacing = spacing_from_reader_properties(props)
        data, bbox, orig_shape = self.predictor.preprocess(image)
        data, slicer_revert_padding = self.pad_nd_image(data, tuple(self.predictor.patch_size), "constant", {"value": 0}, True, None)
        slicers = self.predictor._internal_get_sliding_window_slicers(data.shape[1:])
        data_t = torch.as_tensor(data[None], device=self.device, dtype=torch.float32)
        if self.config.dhga_stage == "B" and self.config.dhga_stage_b_method == "prototype_v4":
            return self._predict_case_stage_b_v4(data_t, slicers, slicer_revert_padding, bbox, orig_shape, props, spacing)
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
            disagreement_region = router.disagreement > 0.5
            self.last_case_diagnostics = {
                "semantic_appearance_complement_rate": float((sem_only | app_only).float().sum().cpu() / union.float().sum().clamp_min(1.0).cpu()),
                "semantic_only_voxels": int(sem_only.sum().cpu()),
                "appearance_only_voxels": int(app_only.sum().cpu()),
                "disagreement_mean": float(router.disagreement.mean().cpu()),
                "disagreement_voxel_rate": float((router.disagreement > 0.5).float().mean().cpu()),
                "mean_w_sem": float(router.w_sem.mean().cpu()),
                "mean_w_app": float(router.w_app.mean().cpu()),
                "mean_w_geo": float(router.w_geo.mean().cpu()),
                "foreground_region_w_sem": _masked_tensor_mean(router.w_sem, union),
                "foreground_region_w_app": _masked_tensor_mean(router.w_app, union),
                "foreground_region_w_geo": _masked_tensor_mean(router.w_geo, union),
                "disagreement_region_w_sem": _masked_tensor_mean(router.w_sem, disagreement_region),
                "disagreement_region_w_app": _masked_tensor_mean(router.w_app, disagreement_region),
                "disagreement_region_w_geo": _masked_tensor_mean(router.w_geo, disagreement_region),
                "geometry_gate_mean": float(router.w_geo.mean().cpu()),
                "geometry_gate_disagreement_mean": float((router.w_geo * router.disagreement).sum().cpu() / router.disagreement.sum().clamp_min(1e-6).cpu()),
                "geometry_gate_high_disagreement_mean": _masked_tensor_mean(router.w_geo, disagreement_region),
            }
            geometry_stats = {}
            if self.config.dhga_geometry_enabled:
                geometry = self.model.run_geometry(data_t,sem_prob,app_prob,router, self.model.text_embeddings, spacing, visual_feature=visual_feature,visual_feature_is_projected=True,)
                phi = mask_to_sdf(router.fused_prob >= self.config.pred_threshold, spacing)
                effective_gate = geometry.get("effective_gate")
                if effective_gate is None:
                    effective_gate = geometry_effective_gate(
                        router.w_geo,
                        (sem_prob - app_prob).abs().clamp(0, 1),
                        phi,
                        self.config,
                        geometry.get("dense_valid_weight"),
                        geometry["dense_displacement_mm"],
                    )
                final = finalize_probability(
                    router.fused_prob,
                    phi,
                    geometry["dense_displacement_mm"],
                    router.w_geo,
                    self.config,
                    geometry.get("dense_valid_weight"),
                    expert_disagreement=(sem_prob - app_prob).abs().clamp(0, 1),
                )
                geometry_stats = summarize_geometry_tensors(geometry["dense_displacement_mm"], effective_gate, router.fused_prob, final, self.config.pred_threshold)
            else:
                geometry = None
                final=router.fused_prob
        final = final[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        fused_crop = router.fused_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        sem_crop = sem_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        app_crop = app_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        reverted = np.zeros((final.shape[0], *orig_shape), dtype=np.float32)
        reverted = self.insert_crop_into_image(reverted, final, bbox)
        fused_reverted = np.zeros((fused_crop.shape[0], *orig_shape), dtype=np.float32)
        sem_reverted = np.zeros((sem_crop.shape[0], *orig_shape), dtype=np.float32)
        app_reverted = np.zeros((app_crop.shape[0], *orig_shape), dtype=np.float32)
        fused_reverted = self.insert_crop_into_image(fused_reverted, fused_crop, bbox)
        sem_reverted = self.insert_crop_into_image(sem_reverted, sem_crop, bbox)
        app_reverted = self.insert_crop_into_image(app_reverted, app_crop, bbox)
        self.last_raw_disagreement = np.abs(sem_reverted[0].astype(np.float32) - app_reverted[0].astype(np.float32))
        self.last_fused_pred = fused_reverted[0] >= self.config.pred_threshold
        self.last_semantic_pred = sem_reverted[0] >= self.config.pred_threshold
        self.last_appearance_pred = app_reverted[0] >= self.config.pred_threshold
        self.last_case_diagnostics.update(geometry_stats)
        return reverted, props

    def _predict_case_stage_b_v4(self, data_t: torch.Tensor, slicers: list, slicer_revert_padding, bbox, orig_shape, props: dict, spacing: tuple[float, float, float]) -> tuple[np.ndarray, dict]:
        data_shape = tuple(data_t.shape[-3:])
        sem_sum = torch.zeros((1, 1, *data_shape), device=self.device)
        app_sum = torch.zeros_like(sem_sum)
        fused_sum = torch.zeros_like(sem_sum)
        anchor_sum = torch.zeros_like(sem_sum)
        count = torch.zeros_like(sem_sum)
        self.model.eval()
        with torch.no_grad():
            for slicer in slicers:
                patch = torch.clone(data_t[(slice(None), *slicer)], memory_format=torch.contiguous_format)
                out = self.model.forward_stage_b_v4(patch, spacing, compute_anchor=True)
                target_shape = patch.shape[-3:]
                sem = F.interpolate(out.semantic_prob, size=target_shape, mode="trilinear", align_corners=False)
                app = F.interpolate(out.appearance_prob, size=target_shape, mode="trilinear", align_corners=False)
                fused = F.interpolate(out.final_prob, size=target_shape, mode="trilinear", align_corners=False)
                anchor = F.interpolate(out.anchor_prob, size=target_shape, mode="trilinear", align_corners=False)
                region = (slice(None), slice(None), *slicer[1:])
                sem_sum[region] += sem
                app_sum[region] += app
                fused_sum[region] += fused
                anchor_sum[region] += anchor
                count[region] += 1
        sem_prob = sem_sum / count.clamp_min(1)
        app_prob = app_sum / count.clamp_min(1)
        final = fused_sum / count.clamp_min(1)
        anchor_prob = anchor_sum / count.clamp_min(1)
        simple = 0.5 * (sem_prob + app_prob)
        sem_mask = sem_prob >= self.config.pred_threshold
        app_mask = app_prob >= self.config.pred_threshold
        union = sem_mask | app_mask
        self.last_case_diagnostics = {
            "semantic_appearance_complement_rate": float(((sem_mask ^ app_mask).float().sum().cpu()) / union.float().sum().clamp_min(1.0).cpu()),
            "semantic_only_voxels": int((sem_mask & ~app_mask).sum().cpu()),
            "appearance_only_voxels": int((app_mask & ~sem_mask).sum().cpu()),
            "stage_b_v4_eval": 1.0,
            "semantic_eval_pred_volume": float(sem_mask.float().mean().cpu()),
            "appearance_eval_pred_volume": float(app_mask.float().mean().cpu()),
            "simple_average_eval_pred_volume": float((simple >= self.config.pred_threshold).float().mean().cpu()),
            "prototype_fused_eval_pred_volume": float((final >= self.config.pred_threshold).float().mean().cpu()),
            "zero_shot_eval_pred_volume": float((anchor_prob >= self.config.pred_threshold).float().mean().cpu()),
        }
        final = final[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        sem_crop = sem_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        app_crop = app_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        simple_crop = simple[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        anchor_crop = anchor_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        reverted = self.insert_crop_into_image(np.zeros((final.shape[0], *orig_shape), dtype=np.float32), final, bbox)
        sem_reverted = self.insert_crop_into_image(np.zeros((sem_crop.shape[0], *orig_shape), dtype=np.float32), sem_crop, bbox)
        app_reverted = self.insert_crop_into_image(np.zeros((app_crop.shape[0], *orig_shape), dtype=np.float32), app_crop, bbox)
        simple_reverted = self.insert_crop_into_image(np.zeros((simple_crop.shape[0], *orig_shape), dtype=np.float32), simple_crop, bbox)
        anchor_reverted = self.insert_crop_into_image(np.zeros((anchor_crop.shape[0], *orig_shape), dtype=np.float32), anchor_crop, bbox)
        self.last_raw_disagreement = np.abs(sem_reverted[0].astype(np.float32) - app_reverted[0].astype(np.float32))
        self.last_fused_pred = simple_reverted[0] >= self.config.pred_threshold
        self.last_semantic_pred = sem_reverted[0] >= self.config.pred_threshold
        self.last_appearance_pred = app_reverted[0] >= self.config.pred_threshold
        self.last_simple_average_pred = simple_reverted[0] >= self.config.pred_threshold
        self.last_zero_shot_pred = anchor_reverted[0] >= self.config.pred_threshold
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


def _masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    if not bool(mask.any().detach().cpu()):
        return 0.0
    return float(values[mask].detach().float().mean().cpu())


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
    base = _binary_metric_values(pred, gt)
    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    metrics: dict[str, float | int] = {
        "dice": base["dice"],
        "iou": base["iou"],
        "precision": base["precision"],
        "recall": base["recall"],
        "tp_voxels": int(base["tp_voxels"]),
        "fp_voxels": int(base["fp_voxels"]),
        "fn_voxels": int(base["fn_voxels"]),
        "fused_dice": base["dice"],
        "gt_voxels": gt_voxels,
        "pred_gt_volume_ratio": float(pred_voxels / max(gt_voxels, 1)),
        "connected_components": connected_components_3d(pred),
    }
    if semantic_pred is not None and appearance_pred is not None:
        semantic_pred = semantic_pred.astype(bool)
        appearance_pred = appearance_pred.astype(bool)
        sem = _binary_metric_values(semantic_pred, gt)
        app = _binary_metric_values(appearance_pred, gt)
        oracle_union = np.logical_or(semantic_pred, appearance_pred)
        oracle_intersection = np.logical_and(semantic_pred, appearance_pred)
        union_dice, union_iou = foreground_dice_iou(oracle_union, gt)
        intersection_dice, intersection_iou = foreground_dice_iou(oracle_intersection, gt)
        metrics.update({
            "semantic_dice": sem["dice"],
            "semantic_iou": sem["iou"],
            "semantic_precision": sem["precision"],
            "semantic_recall": sem["recall"],
            "appearance_dice": app["dice"],
            "appearance_iou": app["iou"],
            "appearance_precision": app["precision"],
            "appearance_recall": app["recall"],
            "oracle_union_dice": union_dice,
            "oracle_union_iou": union_iou,
            "oracle_intersection_dice": intersection_dice,
            "oracle_intersection_iou": intersection_iou,
        })
    return metrics


def compute_raw_disagreement_metrics(
    raw_disagreement: np.ndarray,
    semantic_pred: np.ndarray | None = None,
    appearance_pred: np.ndarray | None = None,
    gt: np.ndarray | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    boundary_band_mm: float = 2.0,
) -> dict[str, float]:
    raw = np.asarray(raw_disagreement, dtype=np.float32)
    metrics: dict[str, float] = {}
    metrics.update(_raw_disagreement_summary(raw, "raw_disagreement"))
    for threshold in (0.1, 0.2, 0.3):
        metrics[f"raw_disagreement_gt_{threshold:.1f}_rate"] = float((raw > threshold).mean()) if raw.size else 0.0

    if semantic_pred is not None and appearance_pred is not None:
        union = np.logical_or(np.asarray(semantic_pred, dtype=bool), np.asarray(appearance_pred, dtype=bool))
        metrics.update(_raw_disagreement_summary(raw[union], "raw_disagreement_union"))
        metrics["raw_disagreement_union_voxel_rate"] = float(union.mean()) if union.size else 0.0
        for threshold in (0.1, 0.2, 0.3):
            metrics[f"raw_disagreement_union_gt_{threshold:.1f}_rate"] = float((raw[union] > threshold).mean()) if union.any() else 0.0

    if gt is not None:
        boundary = gt_boundary_band(np.asarray(gt, dtype=bool), spacing, boundary_band_mm)
        metrics.update(_raw_disagreement_summary(raw[boundary], "raw_disagreement_gt_boundary"))
        for threshold in (0.1, 0.2, 0.3):
            metrics[f"raw_disagreement_gt_boundary_gt_{threshold:.1f}_rate"] = float((raw[boundary] > threshold).mean()) if boundary.any() else 0.0
        metrics["raw_disagreement_gt_boundary_voxel_rate"] = float(boundary.mean()) if boundary.size else 0.0
    return metrics


def summarize_geometry_tensors(displacement_mm: torch.Tensor, effective_gate: torch.Tensor, fused_prob: torch.Tensor, final_prob: torch.Tensor, threshold: float = 0.5) -> dict[str, float | int]:
    disp = displacement_mm.detach().float()
    gate = effective_gate.detach().float()
    modified = (final_prob.detach() >= float(threshold)) != (fused_prob.detach() >= float(threshold))
    active = gate > 0
    if bool(active.any().cpu()):
        active_disp = disp[active]
        disp_mean = float(active_disp.mean().cpu())
        disp_abs_mean = float(active_disp.abs().mean().cpu())
        disp_abs_max = float(active_disp.abs().max().cpu())
    else:
        disp_mean = 0.0
        disp_abs_mean = 0.0
        disp_abs_max = 0.0
    return {
        "geometry_effective_gate_mean": float(gate.mean().cpu()),
        "geometry_effective_gate_max": float(gate.max().cpu()) if gate.numel() else 0.0,
        "geometry_active_gate_voxel_ratio": float(active.float().mean().cpu()) if gate.numel() else 0.0,
        "geometry_displacement_active_mean_mm": disp_mean,
        "geometry_abs_displacement_active_mean_mm": disp_abs_mean,
        "geometry_abs_displacement_max_mm": disp_abs_max,
        "geometry_modified_voxel_ratio": float(modified.float().mean().cpu()) if modified.numel() else 0.0,
        "geometry_modified_voxels": int(modified.sum().cpu()),
    }


def compute_geometry_case_metrics(
    fused_pred: np.ndarray,
    final_pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float],
    surface_tolerance_mm: float,
) -> dict[str, float | int]:
    fused = np.asarray(fused_pred, dtype=bool)
    final = np.asarray(final_pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    fused_base = _binary_metric_values(fused, gt)
    final_base = _binary_metric_values(final, gt)
    fused_surface = surface_distance_metrics(fused, gt, spacing, surface_tolerance_mm)
    final_surface = surface_distance_metrics(final, gt, spacing, surface_tolerance_mm)
    modified = fused ^ final
    tp_gained = np.logical_and.reduce((gt, ~fused, final))
    tp_lost = np.logical_and.reduce((gt, fused, ~final))
    fp_added = np.logical_and.reduce((~gt, ~fused, final))
    fp_removed = np.logical_and.reduce((~gt, fused, ~final))
    return {
        "fused_before_geometry_dice": fused_base["dice"],
        "geometry_after_dice": final_base["dice"],
        "geometry_delta_dice": float(final_base["dice"] - fused_base["dice"]),
        "fused_before_geometry_surface_dice": fused_surface["surface_dice"],
        "geometry_after_surface_dice": final_surface["surface_dice"],
        "geometry_delta_surface_dice": float(final_surface["surface_dice"] - fused_surface["surface_dice"]),
        "fused_before_geometry_hd95": fused_surface["hd95"],
        "geometry_after_hd95": final_surface["hd95"],
        "fused_before_geometry_assd": fused_surface["assd"],
        "geometry_after_assd": final_surface["assd"],
        "geometry_modified_voxel_ratio_case": float(modified.mean()) if modified.size else 0.0,
        "geometry_tp_gained_voxels": int(tp_gained.sum()),
        "geometry_tp_lost_voxels": int(tp_lost.sum()),
        "geometry_fn_recovered_voxels": int(tp_gained.sum()),
        "geometry_fn_added_voxels": int(tp_lost.sum()),
        "geometry_fp_added_voxels": int(fp_added.sum()),
        "geometry_fp_removed_voxels": int(fp_removed.sum()),
        "geometry_tp_delta": int(final_base["tp_voxels"] - fused_base["tp_voxels"]),
        "geometry_fn_delta": int(final_base["fn_voxels"] - fused_base["fn_voxels"]),
        "geometry_fp_delta": int(final_base["fp_voxels"] - fused_base["fp_voxels"]),
    }


def surface_distance_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float], tolerance_mm: float) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if not pred.any() and not gt.any():
        return {"surface_dice": 1.0, "hd95": 0.0, "assd": 0.0}
    if not pred.any() or not gt.any():
        return {"surface_dice": 0.0, "hd95": float("inf"), "assd": float("inf")}
    pred_surface = _surface_voxels(pred)
    gt_surface = _surface_voxels(gt)
    if not pred_surface.any() or not gt_surface.any():
        return {"surface_dice": 0.0, "hd95": float("inf"), "assd": float("inf")}
    try:
        from scipy import ndimage

        dist_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
        dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
        pred_to_gt = dist_to_gt[pred_surface]
        gt_to_pred = dist_to_pred[gt_surface]
    except Exception:
        pred_to_gt = _surface_distances_numpy(pred_surface, gt_surface, spacing)
        gt_to_pred = _surface_distances_numpy(gt_surface, pred_surface, spacing)
    combined = np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float64)
    if combined.size == 0:
        return {"surface_dice": 0.0, "hd95": float("inf"), "assd": float("inf")}
    within = (pred_to_gt <= tolerance_mm).sum() + (gt_to_pred <= tolerance_mm).sum()
    denom = max(pred_to_gt.size + gt_to_pred.size, 1)
    return {
        "surface_dice": float(within / denom),
        "hd95": float(np.percentile(combined, 95)),
        "assd": float(combined.mean()),
    }


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return np.logical_and(mask, ~_binary_erode6(mask))


def _surface_distances_numpy(src_surface: np.ndarray, dst_surface: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    src = np.argwhere(src_surface).astype(np.float64)
    dst = np.argwhere(dst_surface).astype(np.float64)
    if src.size == 0 or dst.size == 0:
        return np.asarray([], dtype=np.float64)
    spacing_arr = np.asarray(spacing, dtype=np.float64)
    out = []
    for point in src:
        dist = np.sqrt((((dst - point) * spacing_arr) ** 2).sum(axis=1))
        out.append(float(dist.min()))
    return np.asarray(out, dtype=np.float64)


def _raw_disagreement_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def gt_boundary_band(mask: np.ndarray, spacing: tuple[float, float, float], band_mm: float = 2.0) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or bool(mask.all()):
        return np.zeros_like(mask, dtype=bool)
    try:
        from scipy import ndimage

        outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
        inside = ndimage.distance_transform_edt(mask, sampling=spacing)
        return np.minimum(outside, inside) <= float(band_mm)
    except Exception:
        min_spacing = max(float(min(spacing)), 1e-6)
        iterations = max(1, int(np.ceil(float(band_mm) / min_spacing)))
        dilated = mask.copy()
        eroded = mask.copy()
        for _ in range(iterations):
            dilated = _binary_dilate6(dilated)
            eroded = _binary_erode6(eroded)
        return np.logical_and(dilated, ~eroded)


def _binary_dilate6(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    out[1:, :, :] |= mask[:-1, :, :]
    out[:-1, :, :] |= mask[1:, :, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def _binary_erode6(mask: np.ndarray) -> np.ndarray:
    return ~_binary_dilate6(~np.asarray(mask, dtype=bool))


def _binary_metric_values(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    dice, iou = foreground_dice_iou(pred, gt)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return {
        "dice": dice,
        "iou": iou,
        "tp_voxels": tp,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "fp_voxels": fp,
        "fn_voxels": fn,
    }
