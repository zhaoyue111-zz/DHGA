#!/usr/bin/env bash
set -euo pipefail

# Progress bars are shown in the terminal.
# TensorBoard logs are written under .save/dhga/tensorboard by default:
#   tensorboard --logdir .save/dhga/tensorboard
python run_3d_dhga.py --train \
  --dhga_stage B \
  --no-dhga_geometry_enabled \
  --dhga_semantic_adapter_target cross \
  --dhga_semantic_adapter_rank 8 \
  --dhga_appearance_feature_layers -1 -2 -3 \
  --dhga_appearance_hidden_ratio 0.25 \
  --dhga_appearance_feature_dropout 0.05 \
  --dhga_anchor_weight 0.25 \
  --dhga_cross_supervision_weight 1.0 \
  --dhga_weak_strong_weight 0.5 \
  --dhga_use_ema_teacher \
  --dhga_ema_decay 0.99 \
  --dhga_stage_b_anchor_candidate_patches 32 \
  --no-dhga_stage_b_include_background_patch \
  --steps_per_volume 10 \
  --lr 5e-5 \
  --prompt_templates "{}" "{} organ" \
  --weight_decay 0.0 \
  --epochs 30 \
  --max_cases 0 \
  --dhga_validation_interval_epochs 2 \
  --dhga_appearance_anchor_weight 0.25 \
  --dhga_appearance_expansion_weight 0.1 \
  --dhga_cross_supervision_min_weight 0.05 \
  --dhga_router_target_weight 0.5 \
  --save_dir .save/dhga/stage_b_full_v3