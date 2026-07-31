#!/usr/bin/env bash
set -euo pipefail

python run_3d_dhga.py --train \
  --dhga_stage B \
  --dhga_stage_b_method text_layer_ensemble \
  --label_values 5 \
  \
  --dhga_semantic_adapter_rank 8 \
  --dhga_semantic_adapter_target cross \
  --dhga_appearance_feature_layers -1 -2 -3 \
  --dhga_appearance_hidden_ratio 0.25 \
  --dhga_appearance_feature_dropout 0.05 \
  \
  --dhga_text_layer_temperature 0.1 \
  --dhga_text_layer_candidate_max_ratio 0.10 \
  --dhga_text_layer_candidate_alpha 0.2 \
  --dhga_text_layer_candidate_weight 0.1 \
  --dhga_text_layer_stability_threshold 0.08 \
  --dhga_text_layer_reliable_fg_threshold 0.80 \
  --dhga_text_layer_reliable_bg_threshold 0.20 \
  --dhga_text_layer_foreground_support_threshold 0.50 \
  --dhga_text_layer_disagreement_threshold 0.05 \
  \
  --dhga_use_ema_teacher \
  --dhga_ema_decay 0.99 \
  \
  --no-dhga_geometry_enabled \
  --lr 5e-5 \
  --prompt_templates "{}" "{} organ" \
  --weight_decay 0.0 \
  --seed 42 \
  --epochs 2 \
  --steps_per_volume 5 \
  --max_cases 2 \
  --dhga_validation_interval_epochs 1 \
  --save_dir .save/dhga/stage_b_v3_smoke