# DHGA Manual Commands

These commands are for manual execution. Codex should not run full training or full validation automatically.

## Static Smoke Test

```bash
conda run -n voxtell python run_3d_dhga.py \
  --smoke_test \
  --device cpu \
  --save_dir .save/dhga/smoke \
  --prompts liver
```

## Unit Tests

```bash
conda run -n voxtell python -m unittest tests.test_dhga_static
```

## Baseline Closed / Stage A Dry Run

```bash
conda run -n voxtell python run_3d_dhga.py \
  --dry_run \
  --dhga_stage A \
  --no-dhga_geometry_enabled \
  --save_dir .save/dhga/stage_a_dry \
  --prompts liver
```

## Dual Expert Debug / Stage B

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --dry_run \
  --dhga_stage B \
  --dhga_semantic_adapter_target cross \
  --dhga_semantic_adapter_rank 8 \
  --dhga_appearance_feature_layers -1 -2 -3 \
  --dhga_appearance_feature_dropout 0.05 \
  --dhga_debug_outputs \
  --save_dir .save/dhga/stage_b_debug \
  --prompts liver
```

## Synthetic Geometry Test

```bash
conda run -n voxtell python -m unittest \
  tests.test_dhga_static.DHGAStaticTests.test_sdf_sign_and_displacement \
  tests.test_dhga_static.DHGAStaticTests.test_ray_sampler_coordinate_order_and_spacing \
  tests.test_dhga_static.DHGAStaticTests.test_bidirectional_corruption_recovery_sign
```

## Stage B Training Template

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --dhga_stage B \
  --dhga_semantic_adapter_target cross \
  --dhga_semantic_adapter_rank 8 \
  --dhga_appearance_feature_layers -1 -2 -3 \
  --dhga_anchor_weight 0.25 \
  --dhga_cross_supervision_weight 1.0 \
  --dhga_weak_strong_weight 0.5 \
  --no-dhga_geometry_enabled \
  --save_dir .save/dhga/stage_b \
  --prompts liver
```

## Stage C Geometry Training Template

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --dhga_stage C \
  --dhga_geometry_enabled \
  --dhga_search_radius_mm 6.0 \
  --dhga_ray_step_mm 1.0 \
  --dhga_corruption_max_offset_mm 3.0 \
  --dhga_corruption_modes inward outward \
  --dhga_boundary_recovery_weight 1.0 \
  --dhga_minimal_transport_weight 0.01 \
  --save_dir .save/dhga/stage_c \
  --prompts liver
```

## Stage D Natural Disagreement Template

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --dhga_stage D \
  --dhga_geometry_enabled \
  --dhga_router_normalization case_rank \
  --dhga_boundary_chunk_size 1024 \
  --dhga_debug_outputs \
  --save_dir .save/dhga/stage_d \
  --prompts liver
```

## Checkpoint Test

```bash
conda run -n voxtell python -m unittest tests.test_dhga_static.DHGAStaticTests.test_checkpoint_strict_roundtrip
```

## Evaluation-Only Metrics Template

Use the legacy VoxTell evaluation path only for evaluation-only GT metrics. Do not pass labels to DHGA training.

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python scripts/voxtell_sfda.py \
  --voxtell_repo /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --data_dir /data/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /data/zy/MLMP/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --val_label_dir /data/zy/CT_MRI_DATA_3D/labels/P0 \
  --label_values 5 \
  --val_interval 1 \
  --val_max_cases 1 \
  --prompts liver \
  --save_dir .save/dhga/eval_only \
  --epochs 0 \
  --steps_per_volume 0
```
