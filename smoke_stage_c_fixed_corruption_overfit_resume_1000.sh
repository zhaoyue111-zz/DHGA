#!/usr/bin/env bash
set -euo pipefail

RESUME_CHECKPOINT=".save/dhga/stage_c_v42_fixed_corruption_overfit_resume_1000/fixed_overfit_checkpoints/step_0200.pt"

if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
  echo "Missing ${RESUME_CHECKPOINT}."
  echo "Run smoke_stage_c_fixed_corruption_overfit_backfill_200_checkpoint.sh first to create it from the fixed sample."
  exit 1
fi

python run_3d_dhga.py \
  --stage_c_fixed_corruption_overfit \
  --dhga_stage C \
  --init_checkpoint .save/dhga/stage_b_full_v42/best_stage_b.pt \
  --dhga_geometry_enabled \
  --dhga_validation_interval_epochs 99 \
  --dhga_geometry_max_displacement_mm 3.0 \
  --dhga_geometry_boundary_band_mm 6.0 \
  --dhga_search_radius_mm 6.0 \
  --dhga_surface_tolerance_mm 1.0 \
  --dhga_ray_step_mm 1.0 \
  --dhga_max_boundary_points 128 \
  --dhga_boundary_chunk_size 128 \
  --dhga_corruption_max_offset_mm 3.0 \
  --dhga_corruption_modes inward outward \
  --dhga_boundary_recovery_zero_weight 0.25 \
  --dhga_minimal_transport_weight 0.0 \
  --save_dir .save/dhga/stage_c_v42_fixed_corruption_overfit_resume_1000 \
  --stage_c_fixed_sample_path .save/dhga/stage_c_v42_fixed_corruption_overfit/fixed_stage_c_sample.pt \
  --stage_c_fixed_overfit_resume_checkpoint "${RESUME_CHECKPOINT}" \
  --voxtell_repo /mnt/afs2/zy/VoxTell_from_disk \
  --model_dir /mnt/afs2/zy/VoxTell_from_disk/model \
  --data_dir /mnt/afs2/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /mnt/afs2/zy/DHGA/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --prompts liver \
  --prompt_templates "{}" \
  --device cuda \
  --max_cases 1 \
  --steps_per_volume 2 \
  --stage_c_fixed_overfit_steps 1000 \
  --stage_c_fixed_overfit_log_interval 50 \
  --stage_c_fixed_overfit_snapshot_steps 200 300 500 750 1000 \
  --stage_c_fixed_overfit_resample_attempts 64
