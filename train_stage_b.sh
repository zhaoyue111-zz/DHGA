#!/usr/bin/env bash
set -euo pipefail

# Progress bars are shown in the terminal.
# TensorBoard logs are written under .save/dhga/tensorboard by default:
#   tensorboard --logdir .save/dhga/tensorboard
#python run_3d_dhga.py --train \
#  --dhga_stage B \
#  --no-dhga_geometry_enabled \
#  --dhga_semantic_adapter_target cross \
#  --dhga_semantic_adapter_rank 8 \
#  --dhga_appearance_feature_layers -1 -2 -3 \
#  --dhga_appearance_hidden_ratio 0.25 \
#  --dhga_appearance_feature_dropout 0.05 \
#  --dhga_anchor_weight 0.25 \
#  --dhga_cross_supervision_weight 1.0 \
#  --dhga_weak_strong_weight 0.5 \
#  --dhga_use_ema_teacher \
#  --dhga_ema_decay 0.99 \
#  --dhga_stage_b_anchor_candidate_patches 32 \
#  --no-dhga_stage_b_include_background_patch \
#  --steps_per_volume 10 \
#  --lr 5e-5 \
#  --prompt_templates "{}" "{} organ" \
#  --weight_decay 0.0 \
#  --epochs 30 \
#  --max_cases 0 \
#  --dhga_validation_interval_epochs 2 \
#  --dhga_appearance_anchor_weight 0.25 \
#  --dhga_appearance_expansion_weight 0.1 \
#  --dhga_cross_supervision_min_weight 0.05 \
#  --dhga_router_target_weight 0.5 \
#  --save_dir .save/dhga/stage_b_full_v3

python run_3d_dhga.py \
  --train \
  --dhga_stage C \
  --init_checkpoint .save/dhga/stage_b_full_v42/best_stage_b.pt \
  --dhga_geometry_enabled \
  --voxtell_repo /mnt/afs2/zy/VoxTell_from_disk \
  --model_dir /mnt/afs2/zy/VoxTell_from_disk/model \
  --data_dir /mnt/afs2/zy/CT_MRI_DATA_3D/images/P0 \
  --val_label_dir /mnt/afs2/zy/CT_MRI_DATA_3D/labels/P0 \
  --split_manifest /mnt/afs2/zy/DHGA/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --sequences P0 \
  --prompts liver \
  --prompt_templates "{}" \
  --label_values 5 \
  --dhga_stage_b_method text_layer_ensemble \
  --dhga_semantic_adapter_rank 8 \
  --dhga_geometry_max_displacement_mm 3.0 \
  --dhga_geometry_boundary_band_mm 6.0 \
  --dhga_search_radius_mm 6.0 \
  --dhga_surface_tolerance_mm 1.0 \
  --dhga_ray_step_mm 1.0 \
  --dhga_max_boundary_points 128 \
  --dhga_boundary_chunk_size 128 \
  --dhga_corruption_max_offset_mm 3.0 \
  --dhga_corruption_modes inward outward \
  --dhga_boundary_recovery_weight 1.0 \
  --dhga_boundary_recovery_zero_weight 1.0 \
  --dhga_minimal_transport_weight 0.01 \
  --steps_per_volume 2 \
  --max_cases 2 \
  --epochs 10 \
  --lr 5e-5 \
  --weight_decay 0.0 \
  --dhga_validation_interval_epochs 2 \
  --save_dir /mnt/afs2/zy/DHGA/.save/dhga/stage_c_v42_balanced_w10_smoke_10ep 2>&1 | tee stage_c_v42_balanced_w10_smoke_10ep.log