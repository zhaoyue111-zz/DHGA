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
from dhga.text_layer_ensemble import build_text_layer_geometry_gate
from dhga.voxtell_model import build_dhga_voxtell_model
from voxtell_sfda.adapter import foreground_dice_iou, load_split_manifest, match_label_path


def spacing_from_reader_properties(props: dict) -> tuple[float, float, float]:
    spacing = props.get("spacing") or props.get("sitk_stuff", {}).get("spacing")
    if spacing is None:
        return (1.0, 1.0, 1.0)
    values = tuple(float(v) for v in spacing)
    return values if len(values) == 3 else (1.0, 1.0, 1.0)


class DHGAEvaluator:
    def __init__(self, config: DHGAConfig, prompts: list[str], save_dir: str | Path, label_dir: str = "", label_values: list[int] | None = None, model=None, predictor=None) -> None:
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
            spacing = spacing_from_reader_properties(image_props)
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
                for metric_name, aux_pred in getattr(self, "last_text_layer_preds", {}).items():
                    aux = _binary_metric_values(aux_pred, gt)
                    row[f"{metric_name}_dice"] = aux["dice"]
                    row[f"{metric_name}_iou"] = aux["iou"]
                    row[f"{metric_name}_precision"] = aux["precision"]
                    row[f"{metric_name}_recall"] = aux["recall"]
                row.update(compute_geometry_case_metrics(fused_pred, pred, gt, spacing, self.config.dhga_surface_tolerance_mm))
                raw_disagreement = getattr(self, "last_raw_disagreement", None)
                if raw_disagreement is not None:
                    row.update(compute_raw_disagreement_metrics(raw_disagreement, semantic_pred=sem_pred, appearance_pred=app_pred, gt=gt, spacing=spacing))
                directional_primary_pred = getattr(self, "last_directional_primary_pred", None)
                expand_score = getattr(self, "last_candidate_expand_score", None)
                shrink_score = getattr(self, "last_candidate_shrink_score", None)
                if directional_primary_pred is not None and expand_score is not None and shrink_score is not None:
                    row.update(compute_directional_candidate_metrics(directional_primary_pred, gt, expand_score, shrink_score, spacing, self.config.dhga_geometry_boundary_band_mm))
            rows.append(row)
            case_base = path.name.replace(".nii.gz", "").replace(".nii", "")
            self.reader.write_seg(pred.astype(np.uint8), str(self.save_dir / f"{case_base}_dhga.nii.gz"), image_props)
            raw_disagreement = getattr(self, "last_raw_disagreement", None)
            if raw_disagreement is not None:
                np.save(self.save_dir / f"{case_base}_raw_disagreement.npy", raw_disagreement.astype(np.float32, copy=False))
                write_float_volume_like_reader(raw_disagreement.astype(np.float32, copy=False), str(self.save_dir / f"{case_base}_raw_disagreement.nii.gz"), image_props)
            if self.config.dhga_debug_outputs:
                directional_primary_pred = getattr(self, "last_directional_primary_pred", None)
                expand_score = getattr(self, "last_candidate_expand_score", None)
                shrink_score = getattr(self, "last_candidate_shrink_score", None)
                if directional_primary_pred is not None:
                    self.reader.write_seg(directional_primary_pred.astype(np.uint8), str(self.save_dir / f"{case_base}_directional_primary.nii.gz"), image_props)
                if expand_score is not None:
                    np.save(self.save_dir / f"{case_base}_candidate_expand_score.npy", expand_score.astype(np.float32, copy=False))
                    write_float_volume_like_reader(expand_score.astype(np.float32, copy=False), str(self.save_dir / f"{case_base}_candidate_expand_score.nii.gz"), image_props)
                if shrink_score is not None:
                    np.save(self.save_dir / f"{case_base}_candidate_shrink_score.npy", shrink_score.astype(np.float32, copy=False))
                    write_float_volume_like_reader(shrink_score.astype(np.float32, copy=False), str(self.save_dir / f"{case_base}_candidate_shrink_score.nii.gz"), image_props)
        metrics = {"rows": rows}
        numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.integer, np.floating))})
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
        sem_sum = torch.zeros((1, 1, *data.shape[1:]), device=self.device)
        app_sum = torch.zeros_like(sem_sum)
        count = torch.zeros_like(sem_sum)
        text_layer_sums: dict[str, torch.Tensor] = {}
        text_candidate_sum = torch.zeros_like(sem_sum)
        text_norm_disagreement_sum = torch.zeros_like(sem_sum)
        directional_primary_sum = torch.zeros_like(sem_sum)
        directional_expand_score_sum = torch.zeros_like(sem_sum)
        directional_shrink_score_sum = torch.zeros_like(sem_sum)
        directional_expand_raw_sum = torch.zeros_like(sem_sum)
        directional_shrink_raw_sum = torch.zeros_like(sem_sum)
        has_directional = False
        visual_sum = None
        self.model.eval()
        with torch.no_grad():
            for slicer in slicers:
                patch = torch.clone(data_t[(slice(None), *slicer)], memory_format=torch.contiguous_format)
                if self.config.dhga_stage == "B" and self.config.dhga_stage_b_method == "text_layer_ensemble" and hasattr(self.model, "forward_text_layer_ensemble"):
                    out = self.model.forward_text_layer_ensemble(patch, spacing)
                else:
                    out = self.model(patch, spacing, run_geometry=False)
                target_shape = patch.shape[-3:]
                sem = F.interpolate(out.semantic_prob, size=target_shape, mode="trilinear", align_corners=False)
                app = F.interpolate(out.appearance_prob, size=target_shape, mode="trilinear", align_corners=False)
                sem_sum[(slice(None), slice(None), *slicer[1:])] += sem
                app_sum[(slice(None), slice(None), *slicer[1:])] += app
                if out.layer_ensemble is not None:
                    for name, key in (("text_last_layer", "last_layer_prob"), ("text_layer_mean", "layer_mean_prob"), ("text_base_fusion", "p_base"), ("text_candidate_enhanced", "p_final")):
                        if key == "last_layer_prob":
                            prob = 0.5 * (out.layer_ensemble["semantic_p_last"] + out.layer_ensemble["appearance_p_last"])
                        elif key == "layer_mean_prob":
                            prob = 0.5 * (out.layer_ensemble["semantic_p_mean"] + out.layer_ensemble["appearance_p_mean"])
                        else:
                            prob = out.layer_ensemble[key]
                        prob = F.interpolate(prob, size=target_shape, mode="trilinear", align_corners=False)
                        if name not in text_layer_sums:
                            text_layer_sums[name] = torch.zeros_like(sem_sum)
                        text_layer_sums[name][(slice(None), slice(None), *slicer[1:])] += prob
                    candidate = F.interpolate(out.layer_ensemble["candidate_score"], size=target_shape, mode="trilinear", align_corners=False)
                    text_candidate_sum[(slice(None), slice(None), *slicer[1:])] += candidate
                    norm_dis = F.interpolate(out.layer_ensemble["normalized_disagreement"], size=target_shape, mode="trilinear", align_corners=False)
                    text_norm_disagreement_sum[(slice(None), slice(None), *slicer[1:])] += norm_dis
                    directional_keys = ("directional_primary_prob", "candidate_expand_score", "candidate_shrink_score", "candidate_expand_raw", "candidate_shrink_raw")
                    if all(key in out.layer_ensemble for key in directional_keys):
                        has_directional = True
                        directional_primary = F.interpolate(out.layer_ensemble["directional_primary_prob"], size=target_shape, mode="trilinear", align_corners=False)
                        expand_score = F.interpolate(out.layer_ensemble["candidate_expand_score"], size=target_shape, mode="trilinear", align_corners=False)
                        shrink_score = F.interpolate(out.layer_ensemble["candidate_shrink_score"], size=target_shape, mode="trilinear", align_corners=False)
                        expand_raw = F.interpolate(out.layer_ensemble["candidate_expand_raw"], size=target_shape, mode="nearest")
                        shrink_raw = F.interpolate(out.layer_ensemble["candidate_shrink_raw"], size=target_shape, mode="nearest")
                        directional_primary_sum[(slice(None), slice(None), *slicer[1:])] += directional_primary
                        directional_expand_score_sum[(slice(None), slice(None), *slicer[1:])] += expand_score
                        directional_shrink_score_sum[(slice(None), slice(None), *slicer[1:])] += shrink_score
                        directional_expand_raw_sum[(slice(None), slice(None), *slicer[1:])] += expand_raw
                        directional_shrink_raw_sum[(slice(None), slice(None), *slicer[1:])] += shrink_raw
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
            text_probs = {name: value / count.clamp_min(1) for name, value in text_layer_sums.items()}
            text_norm_disagreement = None
            directional_primary_prob = directional_primary_sum / count.clamp_min(1) if has_directional else None
            candidate_expand_score = directional_expand_score_sum / count.clamp_min(1) if has_directional else None
            candidate_shrink_score = directional_shrink_score_sum / count.clamp_min(1) if has_directional else None
            candidate_expand_raw = directional_expand_raw_sum / count.clamp_min(1) if has_directional else None
            candidate_shrink_raw = directional_shrink_raw_sum / count.clamp_min(1) if has_directional else None
            if text_probs:
                final_text_prob = text_probs["text_candidate_enhanced"]
                base_text_prob = text_probs["text_base_fusion"]
                candidate_prob = text_candidate_sum / count.clamp_min(1)
                text_norm_disagreement = text_norm_disagreement_sum / count.clamp_min(1)
                router.fused_prob = final_text_prob
                router.geometry_disagreement_weight = torch.maximum(router.geometry_disagreement_weight, candidate_prob.clamp(0, 1))
            else:
                final_text_prob = None
                base_text_prob = None
                candidate_prob = None
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
            if text_probs:
                self.last_case_diagnostics.update({"text_layer_candidate_ratio": float((candidate_prob > 0).float().mean().cpu()), "text_layer_base_pred_volume": float((base_text_prob >= self.config.pred_threshold).float().mean().cpu()), "text_layer_enhanced_pred_volume": float((final_text_prob >= self.config.pred_threshold).float().mean().cpu())})
            if has_directional:
                self.last_case_diagnostics.update({"directional_expand_raw_ratio_patch_space": float((candidate_expand_raw > 0.5).float().mean().cpu()), "directional_shrink_raw_ratio_patch_space": float((candidate_shrink_raw > 0.5).float().mean().cpu())})
            geometry_stats = {}
            if self.config.dhga_geometry_enabled:
                geometry = self.model.run_geometry(data_t, sem_prob, app_prob, router, self.model.text_embeddings, spacing, visual_feature=visual_feature, visual_feature_is_projected=True, candidate_score=candidate_prob, candidate_fg=(candidate_prob > 0).float() if candidate_prob is not None else None)
                phi = mask_to_sdf(router.fused_prob >= self.config.pred_threshold, spacing)
                if text_probs and candidate_prob is not None and text_norm_disagreement is not None:
                    sdf_boundary_band = (phi.abs() <= float(self.config.dhga_geometry_boundary_band_mm)).to(dtype=candidate_prob.dtype)
                    w_geo_eval = build_text_layer_geometry_gate(candidate_prob.clamp(0, 1), text_norm_disagreement.clamp(0, 1), sdf_boundary_band=sdf_boundary_band, config=self.config)
                    router.w_geo = w_geo_eval
                    geometry["w_geo_eval"] = w_geo_eval.detach()
                effective_gate = geometry.get("effective_gate")
                if effective_gate is None or (text_probs and candidate_prob is not None and text_norm_disagreement is not None):
                    effective_gate = geometry_effective_gate(router.w_geo, (sem_prob - app_prob).abs().clamp(0, 1), phi, self.config, geometry.get("dense_valid_weight"), geometry["dense_displacement_mm"])
                final = finalize_probability(router.fused_prob, phi, geometry["dense_displacement_mm"], router.w_geo, self.config, geometry.get("dense_valid_weight"), expert_disagreement=(sem_prob - app_prob).abs().clamp(0, 1))
                geometry_stats = summarize_geometry_tensors(geometry["dense_displacement_mm"], effective_gate, router.fused_prob, final, self.config.pred_threshold)
            else:
                final = final_text_prob if final_text_prob is not None else router.fused_prob
        final = final[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        fused_crop = router.fused_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        sem_crop = sem_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        app_crop = app_prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
        reverted = np.zeros((final.shape[0], *orig_shape), dtype=np.float32)
        fused_reverted = np.zeros((fused_crop.shape[0], *orig_shape), dtype=np.float32)
        sem_reverted = np.zeros((sem_crop.shape[0], *orig_shape), dtype=np.float32)
        app_reverted = np.zeros((app_crop.shape[0], *orig_shape), dtype=np.float32)
        reverted = self.insert_crop_into_image(reverted, final, bbox)
        fused_reverted = self.insert_crop_into_image(fused_reverted, fused_crop, bbox)
        sem_reverted = self.insert_crop_into_image(sem_reverted, sem_crop, bbox)
        app_reverted = self.insert_crop_into_image(app_reverted, app_crop, bbox)
        self.last_text_layer_preds = {}
        for name, prob in text_probs.items():
            crop = prob[(slice(None), slice(None), *slicer_revert_padding[1:])].cpu().numpy()[0]
            restored = np.zeros((crop.shape[0], *orig_shape), dtype=np.float32)
            restored = self.insert_crop_into_image(restored, crop, bbox)
            self.last_text_layer_preds[name] = restored[0] >= self.config.pred_threshold
        self.last_directional_primary_pred = None
        self.last_candidate_expand_score = None
        self.last_candidate_shrink_score = None
        self.last_candidate_expand_raw = None
        self.last_candidate_shrink_raw = None
        if has_directional and directional_primary_prob is not None and candidate_expand_score is not None and candidate_shrink_score is not None:
            directional_primary_reverted = _restore_tensor_to_original(directional_primary_prob, slicer_revert_padding, bbox, orig_shape, self.insert_crop_into_image)
            expand_score_reverted = _restore_tensor_to_original(candidate_expand_score, slicer_revert_padding, bbox, orig_shape, self.insert_crop_into_image)
            shrink_score_reverted = _restore_tensor_to_original(candidate_shrink_score, slicer_revert_padding, bbox, orig_shape, self.insert_crop_into_image)
            expand_raw_reverted = _restore_tensor_to_original(candidate_expand_raw, slicer_revert_padding, bbox, orig_shape, self.insert_crop_into_image)
            shrink_raw_reverted = _restore_tensor_to_original(candidate_shrink_raw, slicer_revert_padding, bbox, orig_shape, self.insert_crop_into_image)
            self.last_directional_primary_pred = directional_primary_reverted >= self.config.pred_threshold
            self.last_candidate_expand_score = expand_score_reverted.astype(np.float32, copy=False)
            self.last_candidate_shrink_score = shrink_score_reverted.astype(np.float32, copy=False)
            self.last_candidate_expand_raw = expand_raw_reverted >= 0.5
            self.last_candidate_shrink_raw = shrink_raw_reverted >= 0.5
        self.last_raw_disagreement = np.abs(sem_reverted[0].astype(np.float32) - app_reverted[0].astype(np.float32))
        self.last_fused_pred = fused_reverted[0] >= self.config.pred_threshold
        self.last_semantic_pred = sem_reverted[0] >= self.config.pred_threshold
        self.last_appearance_pred = app_reverted[0] >= self.config.pred_threshold
        self.last_case_diagnostics.update(geometry_stats)
        return reverted, props


def _restore_tensor_to_original(prob: torch.Tensor, slicer_revert_padding, bbox, orig_shape, insert_crop_into_image) -> np.ndarray:
    crop = prob[(slice(None), slice(None), *slicer_revert_padding[1:])].detach().cpu().numpy()[0]
    restored = np.zeros((crop.shape[0], *orig_shape), dtype=np.float32)
    restored = insert_crop_into_image(restored, crop, bbox)
    return restored[0]


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
        raise ValueError("float volume must have shape [D, H, W]")
    restored = volume.transpose((2, 1, 0))
    img = nibabel.Nifti1Image(restored, affine=properties["nibabel_stuff"]["reoriented_affine"])
    img_ornt = io_orientation(properties["nibabel_stuff"]["original_affine"])
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    img_reoriented = img.as_reoriented(from_canonical)
    nibabel.save(img_reoriented, output_fname)


def compute_binary_case_metrics(pred: np.ndarray, gt: np.ndarray, semantic_pred: np.ndarray | None = None, appearance_pred: np.ndarray | None = None) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    base = _binary_metric_values(pred, gt)
    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    metrics: dict[str, float | int] = {
        "dice": base["dice"], "iou": base["iou"], "precision": base["precision"], "recall": base["recall"], "tp_voxels": int(base["tp_voxels"]), "fp_voxels": int(base["fp_voxels"]), "fn_voxels": int(base["fn_voxels"]), "fused_dice": base["dice"], "gt_voxels": gt_voxels, "pred_gt_volume_ratio": float(pred_voxels / max(gt_voxels, 1)), "connected_components": connected_components_3d(pred),
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
        metrics.update({"semantic_dice": sem["dice"], "semantic_iou": sem["iou"], "semantic_precision": sem["precision"], "semantic_recall": sem["recall"], "appearance_dice": app["dice"], "appearance_iou": app["iou"], "appearance_precision": app["precision"], "appearance_recall": app["recall"], "oracle_union_dice": union_dice, "oracle_union_iou": union_iou, "oracle_intersection_dice": intersection_dice, "oracle_intersection_iou": intersection_iou})
    return metrics


def compute_directional_candidate_metrics(primary_pred: np.ndarray, gt: np.ndarray, expand_score: np.ndarray, shrink_score: np.ndarray, spacing: tuple[float, float, float], boundary_band_mm: float) -> dict[str, float | int]:
    primary = np.asarray(primary_pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    expand_score = np.asarray(expand_score, dtype=np.float32)
    shrink_score = np.asarray(shrink_score, dtype=np.float32)
    if primary.shape != gt.shape or expand_score.shape != gt.shape or shrink_score.shape != gt.shape:
        raise ValueError("directional candidate arrays and gt must have identical shapes")
    boundary = gt_boundary_band(primary, spacing, boundary_band_mm)
    expand_region = boundary & (~primary)
    shrink_region = boundary & primary
    boundary_fn = expand_region & gt
    boundary_fp = shrink_region & (~gt)
    metrics: dict[str, float | int] = {
        "directional_expand_raw_ratio": float((expand_score > 0).mean()) if expand_score.size else 0.0,
        "directional_shrink_raw_ratio": float((shrink_score > 0).mean()) if shrink_score.size else 0.0,
        "directional_expand_boundary_candidate_ratio": float(((expand_score > 0) & expand_region).sum() / max(int(expand_region.sum()), 1)),
        "directional_shrink_boundary_candidate_ratio": float(((shrink_score > 0) & shrink_region).sum() / max(int(shrink_region.sum()), 1)),
        "directional_boundary_fn_ratio": float(boundary_fn.sum() / max(int(expand_region.sum()), 1)),
        "directional_boundary_fp_ratio": float(boundary_fp.sum() / max(int(shrink_region.sum()), 1)),
        "directional_boundary_voxel_ratio": float(boundary.mean()) if boundary.size else 0.0,
        "directional_boundary_fn_voxels": int(boundary_fn.sum()),
        "directional_boundary_fp_voxels": int(boundary_fp.sum()),
    }
    for ratio, suffix in ((0.01, "top01"), (0.02, "top02"), (0.05, "top05")):
        metrics.update(_directional_top_metrics(expand_score, expand_region, boundary_fn, ratio, f"directional_expand_{suffix}", "boundary_fn"))
        metrics.update(_directional_top_metrics(shrink_score, shrink_region, boundary_fp, ratio, f"directional_shrink_{suffix}", "boundary_fp"))
    return metrics


def _directional_top_metrics(score: np.ndarray, eligible: np.ndarray, error_mask: np.ndarray, ratio: float, prefix: str, error_name: str) -> dict[str, float | int]:
    score = np.asarray(score, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)
    error_mask = np.asarray(error_mask, dtype=bool)
    eligible_count = int(eligible.sum())
    error_count = int(error_mask.sum())
    baseline = float(error_count / eligible_count) if eligible_count else 0.0
    positive_indices = np.flatnonzero((eligible & (score > 0)).ravel())
    target_count = max(1, int(np.ceil(eligible_count * float(ratio)))) if eligible_count else 0
    selected_count = min(target_count, int(positive_indices.size))
    selected = np.zeros(score.size, dtype=bool)
    if selected_count > 0:
        positive_scores = score.ravel()[positive_indices]
        if selected_count == positive_indices.size:
            chosen = positive_indices
        else:
            chosen_local = np.argpartition(positive_scores, -selected_count)[-selected_count:]
            chosen = positive_indices[chosen_local]
        selected[chosen] = True
    selected = selected.reshape(score.shape)
    hits = int((selected & error_mask).sum())
    hit_rate = float(hits / selected_count) if selected_count else 0.0
    coverage = float(hits / error_count) if error_count else 0.0
    lift = float(hit_rate / baseline) if baseline > 0 else 0.0
    return {f"{prefix}_hit_rate": hit_rate, f"{prefix}_{error_name}_coverage": coverage, f"{prefix}_lift": lift, f"{prefix}_selected_voxels": selected_count, f"{prefix}_selected_region_ratio": float(selected_count / eligible_count) if eligible_count else 0.0}


def compute_raw_disagreement_metrics(raw_disagreement: np.ndarray, semantic_pred: np.ndarray | None = None, appearance_pred: np.ndarray | None = None, gt: np.ndarray | None = None, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0), boundary_band_mm: float = 2.0) -> dict[str, float]:
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
    return {"geometry_effective_gate_mean": float(gate.mean().cpu()), "geometry_effective_gate_max": float(gate.max().cpu()) if gate.numel() else 0.0, "geometry_active_gate_voxel_ratio": float(active.float().mean().cpu()) if gate.numel() else 0.0, "geometry_displacement_active_mean_mm": disp_mean, "geometry_abs_displacement_active_mean_mm": disp_abs_mean, "geometry_abs_displacement_max_mm": disp_abs_max, "geometry_modified_voxel_ratio": float(modified.float().mean().cpu()) if modified.numel() else 0.0, "geometry_modified_voxels": int(modified.sum().cpu())}


def compute_geometry_case_metrics(fused_pred: np.ndarray, final_pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float], surface_tolerance_mm: float) -> dict[str, float | int]:
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
    return {"fused_before_geometry_dice": fused_base["dice"], "geometry_after_dice": final_base["dice"], "geometry_delta_dice": float(final_base["dice"] - fused_base["dice"]), "fused_before_geometry_surface_dice": fused_surface["surface_dice"], "geometry_after_surface_dice": final_surface["surface_dice"], "geometry_delta_surface_dice": float(final_surface["surface_dice"] - fused_surface["surface_dice"]), "fused_before_geometry_hd95": fused_surface["hd95"], "geometry_after_hd95": final_surface["hd95"], "fused_before_geometry_assd": fused_surface["assd"], "geometry_after_assd": final_surface["assd"], "geometry_modified_voxel_ratio_case": float(modified.mean()) if modified.size else 0.0, "geometry_tp_gained_voxels": int(tp_gained.sum()), "geometry_tp_lost_voxels": int(tp_lost.sum()), "geometry_fn_recovered_voxels": int(tp_gained.sum()), "geometry_fn_added_voxels": int(tp_lost.sum()), "geometry_fp_added_voxels": int(fp_added.sum()), "geometry_fp_removed_voxels": int(fp_removed.sum()), "geometry_tp_delta": int(final_base["tp_voxels"] - fused_base["tp_voxels"]), "geometry_fn_delta": int(final_base["fn_voxels"] - fused_base["fn_voxels"]), "geometry_fp_delta": int(final_base["fp_voxels"] - fused_base["fp_voxels"])}


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
    return {"surface_dice": float(within / denom), "hd95": float(np.percentile(combined, 95)), "assd": float(combined.mean())}


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
        return {f"{prefix}_mean": 0.0, f"{prefix}_p50": 0.0, f"{prefix}_p75": 0.0, f"{prefix}_p90": 0.0, f"{prefix}_p95": 0.0}
    return {f"{prefix}_mean": float(values.mean()), f"{prefix}_p50": float(np.percentile(values, 50)), f"{prefix}_p75": float(np.percentile(values, 75)), f"{prefix}_p90": float(np.percentile(values, 90)), f"{prefix}_p95": float(np.percentile(values, 95))}


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
    return {"dice": dice, "iou": iou, "tp_voxels": tp, "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)), "fp_voxels": fp, "fn_voxels": fn}
