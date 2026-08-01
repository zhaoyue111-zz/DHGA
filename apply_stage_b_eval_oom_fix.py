#!/usr/bin/env python3
from pathlib import Path

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"[skip] {label} already applied")
        return text
    if old not in text:
        raise RuntimeError(f"Cannot find expected source block for: {label}")
    return text.replace(old, new, 1)

root = Path.cwd()
evaluation_path = root / "dhga" / "evaluation.py"
trainer_path = root / "dhga" / "trainer.py"
if not evaluation_path.exists() or not trainer_path.exists():
    raise RuntimeError("Run this script from the DHGA repository root")

evaluation = evaluation_path.read_text()
trainer = trainer_path.read_text()
for source_path in (evaluation_path, trainer_path):
    backup_path = source_path.with_suffix(source_path.suffix + '.before_oom_fix')
    if not backup_path.exists():
        backup_path.write_text(source_path.read_text())

evaluation = replace_once(
    evaluation,
    'from __future__ import annotations\n\nimport json\nfrom pathlib import Path\n',
    'from __future__ import annotations\n\nimport gc\nimport json\nfrom pathlib import Path\nfrom types import SimpleNamespace\n',
    'evaluation imports',
)

evaluation = replace_once(
    evaluation,
    '''        rows = []
        for path in tqdm(paths, desc="DHGA evaluation-only", dynamic_ncols=True):
            probs, image_props = self.predict_case(path)
''',
    '''        rows = []
        for path in tqdm(paths, desc="DHGA evaluation-only", dynamic_ncols=True):
            if self.device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
            probs, image_props = self.predict_case(path)
''',
    'clear CUDA cache before each validation case',
)

evaluation = replace_once(
    evaluation,
    '''        text_candidate_sum = torch.zeros_like(sem_sum)
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
''',
    '''        text_candidate_sum = torch.zeros_like(sem_sum)
        lightweight_text_stage_b = self.config.dhga_stage == "B" and self.config.dhga_stage_b_method == "text_layer_ensemble" and not self.config.dhga_geometry_enabled
        collect_directional = bool(self.config.dhga_debug_outputs)
        text_norm_disagreement_sum = torch.zeros_like(sem_sum) if self.config.dhga_geometry_enabled else None
        directional_primary_sum = torch.zeros_like(sem_sum) if collect_directional else None
        directional_expand_score_sum = torch.zeros_like(sem_sum) if collect_directional else None
        directional_shrink_score_sum = torch.zeros_like(sem_sum) if collect_directional else None
        directional_expand_raw_sum = torch.zeros_like(sem_sum) if collect_directional else None
        directional_shrink_raw_sum = torch.zeros_like(sem_sum) if collect_directional else None
        has_directional = False
        visual_sum = None
        self.model.eval()
        with torch.inference_mode():
''',
    'lightweight Stage B validation allocations',
)

evaluation = replace_once(
    evaluation,
    '''                    norm_dis = F.interpolate(out.layer_ensemble["normalized_disagreement"], size=target_shape, mode="trilinear", align_corners=False)
                    text_norm_disagreement_sum[(slice(None), slice(None), *slicer[1:])] += norm_dis
                    directional_keys = ("directional_primary_prob", "candidate_expand_score", "candidate_shrink_score", "candidate_expand_raw", "candidate_shrink_raw")
                    if all(key in out.layer_ensemble for key in directional_keys):
''',
    '''                    if text_norm_disagreement_sum is not None:
                        norm_dis = F.interpolate(out.layer_ensemble["normalized_disagreement"], size=target_shape, mode="trilinear", align_corners=False)
                        text_norm_disagreement_sum[(slice(None), slice(None), *slicer[1:])] += norm_dis
                    directional_keys = ("directional_primary_prob", "candidate_expand_score", "candidate_shrink_score", "candidate_expand_raw", "candidate_shrink_raw")
                    if collect_directional and all(key in out.layer_ensemble for key in directional_keys):
                        assert directional_primary_sum is not None and directional_expand_score_sum is not None and directional_shrink_score_sum is not None and directional_expand_raw_sum is not None and directional_shrink_raw_sum is not None
''',
    'disable directional accumulation during normal periodic validation',
)

evaluation = replace_once(
    evaluation,
    '''                count[(slice(None), slice(None), *slicer[1:])] += 1
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
''',
    '''                count[(slice(None), slice(None), *slicer[1:])] += 1
                if not lightweight_text_stage_b:
                    visual = self.model.geometry_visual_proj(out.features.encoder_stages[self.model.geometry_feature_idx])
                    visual = F.interpolate(visual, size=target_shape, mode="trilinear", align_corners=False)
                    if visual_sum is None:
                        visual_sum = torch.zeros((1, visual.shape[1], *data.shape[1:]), device=self.device)
                    visual_sum[(slice(None), slice(None), *slicer[1:])] += visual
                del out, patch, sem, app
            sem_prob = sem_sum / count.clamp_min(1)
            app_prob = app_sum / count.clamp_min(1)
            text_probs = {name: value / count.clamp_min(1) for name, value in text_layer_sums.items()}
''',
    'skip full-volume visual accumulation for Stage B text ensemble',
)

evaluation = replace_once(
    evaluation,
    '''            text_norm_disagreement = None
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
''',
    '''            text_norm_disagreement = None
            directional_primary_prob = directional_primary_sum / count.clamp_min(1) if has_directional and directional_primary_sum is not None else None
            candidate_expand_score = directional_expand_score_sum / count.clamp_min(1) if has_directional and directional_expand_score_sum is not None else None
            candidate_shrink_score = directional_shrink_score_sum / count.clamp_min(1) if has_directional and directional_shrink_score_sum is not None else None
            candidate_expand_raw = directional_expand_raw_sum / count.clamp_min(1) if has_directional and directional_expand_raw_sum is not None else None
            candidate_shrink_raw = directional_shrink_raw_sum / count.clamp_min(1) if has_directional and directional_shrink_raw_sum is not None else None
            if text_probs:
                final_text_prob = text_probs["text_candidate_enhanced"]
                base_text_prob = text_probs["text_base_fusion"]
                candidate_prob = text_candidate_sum / count.clamp_min(1)
                text_norm_disagreement = text_norm_disagreement_sum / count.clamp_min(1) if text_norm_disagreement_sum is not None else None
            else:
                final_text_prob = None
                base_text_prob = None
                candidate_prob = None
            if lightweight_text_stage_b:
                if final_text_prob is None:
                    raise RuntimeError("Stage B text-layer evaluation did not produce text probabilities")
                disagreement = (sem_prob - app_prob).abs().clamp(0, 1)
                half = torch.full_like(disagreement, 0.5)
                zero = torch.zeros_like(disagreement)
                router = SimpleNamespace(fused_prob=final_text_prob, disagreement=disagreement, w_sem=half, w_app=half, w_geo=zero, geometry_disagreement_weight=candidate_prob.clamp(0, 1) if candidate_prob is not None else zero)
                visual_feature = None
            else:
                if visual_sum is None:
                    raise RuntimeError("Router evaluation requires accumulated visual features")
                visual_feature = visual_sum / count.clamp_min(1)
                router = self.model.router(sem_prob, app_prob, visual_context=visual_feature.mean(dim=1, keepdim=True))
            if text_probs:
                router.fused_prob = final_text_prob
                router.geometry_disagreement_weight = torch.maximum(router.geometry_disagreement_weight, candidate_prob.clamp(0, 1))
''',
    'replace legacy full-volume Router in Stage B validation',
)

evaluation = replace_once(
    evaluation,
    '''        self.last_case_diagnostics.update(geometry_stats)
        return reverted, props
''',
    '''        self.last_case_diagnostics.update(geometry_stats)
        del data_t, sem_sum, app_sum, count, text_candidate_sum, sem_prob, app_prob, router
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        return reverted, props
''',
    'release validation tensors before returning',
)

trainer = replace_once(
    trainer,
    'from __future__ import annotations\n\nimport json\n',
    'from __future__ import annotations\n\nimport gc\nimport json\n',
    'trainer imports',
)

trainer = replace_once(
    trainer,
    '''        eval_config = DHGAConfig.from_mapping(values)
        eval_dir = self.save_dir / f"eval_epoch_{epoch:04d}"
        was_training = self.model.training
        evaluator = DHGAEvaluator(
''',
    '''        eval_config = DHGAConfig.from_mapping(values)
        eval_dir = self.save_dir / f"eval_epoch_{epoch:04d}"
        was_training = self.model.training
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        evaluator = DHGAEvaluator(
''',
    'clear cache before periodic validation',
)

trainer = replace_once(
    trainer,
    '''        try:
            metrics = evaluator.evaluate_split("test", 0)
        finally:
            self.model.train(was_training)
''',
    '''        try:
            metrics = evaluator.evaluate_split("test", 0)
        finally:
            self.model.train(was_training)
            del evaluator
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
''',
    'clear cache after periodic validation',
)

evaluation_path.write_text(evaluation)
trainer_path.write_text(trainer)
print("OOM fix applied to dhga/evaluation.py and dhga/trainer.py")
