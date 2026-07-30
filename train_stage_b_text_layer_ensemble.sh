#!/usr/bin/env bash
set -euo pipefail

python run_3d_dhga.py \
  --train \
  --dhga_stage B \
  --dhga_stage_b_method text_layer_ensemble \
  --epochs "${EPOCHS:-2}" \
  --steps_per_volume "${STEPS_PER_VOLUME:-1}" \
  --max_cases "${MAX_CASES:-2}" \
  --dhga_validation_interval_epochs 1 \
  --no-dhga_geometry_enabled \
  --save_dir "${SAVE_DIR:-.save/dhga/stage_b_text_layer_ensemble}"
