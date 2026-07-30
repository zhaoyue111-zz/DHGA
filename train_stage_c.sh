#!/usr/bin/env bash
set -euo pipefail

# Progress bars are shown in the terminal.
# TensorBoard logs are written under .save/dhga/tensorboard by default:
#   tensorboard --logdir .save/dhga/tensorboard
python run_3d_dhga.py --train \
  --dhga_stage C \
  --init_checkpoint /mnt/afs2/zy/DHGA/.save/dhga/stage_b_full_v3/checkpoint_best.pt \
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
  --steps_per_volume 10 \
  --lr 5e-5 \
  --prompt_templates "{}" "{} organ" \
  --weight_decay 0.0 \
  --epochs 30 \
  --max_cases 0 \
  --dhga_validation_interval_epochs 2 \
  --dhga_boundary_recovery_weight 1.0 \
  --dhga_minimal_transport_weight 0.01 \
  --save_dir .save/dhga/stage_c_full_v1