#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" conda run -n voxtell python run_3d_dhga.py \
  --train \
  --dhga_stage C \
  --init_checkpoint .save/dhga/stage_b/best_stage_b.pt \
  --dhga_geometry_enabled \
  --dhga_geometry_max_displacement_mm 3.0 \
  --dhga_geometry_boundary_band_mm 6.0 \
  --dhga_search_radius_mm 6.0 \
  --dhga_surface_tolerance_mm 1.0 \
  --dhga_ray_step_mm 1.0 \
  --dhga_max_boundary_points 128 \
  --dhga_boundary_chunk_size 128 \
  --dhga_corruption_max_offset_mm 3.0 \
  --dhga_corruption_modes inward outward \
  --save_dir .save/dhga/stage_c_smoke \
  --voxtell_repo /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --data_dir /data/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /data/zy/DHGA/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --prompts liver \
  --prompt_templates "{}" \
  --device cuda \
  --epochs 1 \
  --steps_per_volume 1 \
  --max_cases 1
