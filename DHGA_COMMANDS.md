# DHGA Commands

Run these manually from `/data/zy/DHGA`. The commands below match the real `run_3d_dhga.py` CLI.

## Self Check

```bash
conda run -n voxtell python run_3d_dhga.py \
  --self_check \
  --device cpu \
  --save_dir .save/dhga/self_check \
  --prompts liver
```

## Unit Tests

```bash
conda run -n voxtell python -m unittest tests.test_dhga_static
```

## Dry Run

```bash
conda run -n voxtell python run_3d_dhga.py \
  --dry_run \
  --config_json configs/dhga_default.json \
  --save_dir .save/dhga/dry \
  --prompts liver
```

## Common Data Arguments

Append these to real Stage A/B/C/D training commands.

```bash
  --voxtell_repo /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --data_dir /data/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /data/zy/MLMP/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --prompts liver \
  --prompt_templates "{}" \
  --device cuda \
  --epochs 1 \
  --steps_per_volume 1 \
  --max_cases 1
```

## Stage A Baseline/Anchor Check

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --train \
  --dhga_stage A \
  --no-dhga_geometry_enabled \
  --save_dir .save/dhga/stage_a \
  --voxtell_repo /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --data_dir /data/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /data/zy/MLMP/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --prompts liver \
  --prompt_templates "{}" \
  --device cuda \
  --epochs 1 \
  --steps_per_volume 1 \
  --max_cases 1
```

## Stage B Dual Expert Training

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --train \
  --dhga_stage B \
  --no-dhga_geometry_enabled \
  --dhga_semantic_adapter_target cross \
  --dhga_semantic_adapter_rank 8 \
  --dhga_appearance_feature_layers -1 -2 -3 \
  --dhga_anchor_weight 0.25 \
  --dhga_cross_supervision_weight 1.0 \
  --dhga_weak_strong_weight 0.5 \
  --save_dir .save/dhga/stage_b \
  --voxtell_repo /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --data_dir /data/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /data/zy/MLMP/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --prompts liver \
  --prompt_templates "{}" \
  --device cuda \
  --epochs 10 \
  --steps_per_volume 2
```

## Stage C Geometry Recovery Training

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --train \
  --dhga_stage C \
  --init_checkpoint .save/dhga/stage_b/checkpoint_final.pt \
  --dhga_geometry_enabled \
  --dhga_search_radius_mm 6.0 \
  --dhga_surface_tolerance_mm 1.0 \
  --dhga_ray_step_mm 1.0 \
  --dhga_max_boundary_points 4096 \
  --dhga_boundary_chunk_size 1024 \
  --dhga_corruption_max_offset_mm 3.0 \
  --dhga_corruption_modes inward outward \
  --dhga_boundary_recovery_weight 1.0 \
  --dhga_minimal_transport_weight 0.01 \
  --save_dir .save/dhga/stage_c \
  --voxtell_repo /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --data_dir /data/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /data/zy/MLMP/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --prompts liver \
  --prompt_templates "{}" \
  --device cuda \
  --epochs 10 \
  --steps_per_volume 2
```

## Stage D Natural Disagreement Geometry

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --train \
  --dhga_stage D \
  --init_checkpoint .save/dhga/stage_c/checkpoint_final.pt \
  --dhga_geometry_enabled \
  --dhga_router_normalization case_rank \
  --dhga_boundary_chunk_size 1024 \
  --dhga_debug_outputs \
  --save_dir .save/dhga/stage_d \
  --voxtell_repo /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --data_dir /data/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /data/zy/MLMP/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --prompts liver \
  --prompt_templates "{}" \
  --device cuda \
  --epochs 10 \
  --steps_per_volume 2
```

## Resume Interrupted Training

Use `--resume_checkpoint` for the same stage. This restores model, optimizer, EMA, AMP scaler, epoch, global step, and RNG state.

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --train \
  --dhga_stage C \
  --resume_checkpoint .save/dhga/stage_c/checkpoint_last.pt \
  --save_dir .save/dhga/stage_c \
  --voxtell_repo /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --data_dir /data/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /data/zy/MLMP/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --prompts liver \
  --device cuda \
  --epochs 10 \
  --steps_per_volume 2
```

## Focused Geometry Tests

```bash
conda run -n voxtell python -m unittest \
  tests.test_dhga_static.DHGAStaticTests.test_sdf_normals_point_outward \
  tests.test_dhga_static.DHGAStaticTests.test_boundary_points_and_dense_displacement \
  tests.test_dhga_static.DHGAStaticTests.test_ray_sampler_coordinate_order_and_spacing
```
