#!/usr/bin/env bash
set -euo pipefail

# Progress bars are shown in the terminal.
# TensorBoard logs are written under .save/dhga/tensorboard by default:
#   tensorboard --logdir .save/dhga/tensorboard
CUDA_VISIBLE_DEVICES=0 conda run -n voxtell python run_3d_dhga.py \
  --train \
  --dhga_stage B \
  --no-dhga_geometry_enabled \
  --dhga_semantic_adapter_target cross \
  --dhga_semantic_adapter_rank 8 \
  --dhga_appearance_feature_layers -1 -2 -3 \
  --dhga_anchor_weight 0.25 \
  --dhga_appearance_anchor_weight 0.25 \
  --dhga_appearance_expansion_weight 0.1 \
  --dhga_cross_supervision_weight 1.0 \
  --dhga_cross_supervision_min_weight 0.05 \
  --dhga_weak_strong_weight 0.5 \
  --dhga_router_target_weight 0.5 \
  --dhga_prompt_ranking_weight 0 \
  --save_dir .save/dhga/stage_b \
  --sequences P0 \
  --prompts liver \
  --prompt_templates "{}" \
  --epochs 15 \
  --steps_per_volume 1
