# DHGA Implementation Notes

## Scope

This repository now contains an independent `dhga/` namespace and a new `run_3d_dhga.py` entrypoint. The implementation is intentionally separated from `scripts/voxtell_sfda.py`; the old MLMP local entropy, pseudo CLS entropy, multi-prompt entropy averaging, and foreground-prior losses remain available only through the legacy entrypoint and are not imported by DHGA.

The current DHGA code provides typed configuration, migrated VoxTell utility code, a real `DHGAVoxTellModel`, Stage A/B/C/D trainer paths, random-tensor smoke checks, static tests, strict checkpoint format, and a command plan matching the CLI. Real training is only launched when the user explicitly passes `--train`.

## Reused MLMP-VoxTell Components

- `voxtell_sfda/`: migrated into this repository so DHGA can import prompt parsing, NIfTI discovery/loading helpers, VoxTell import preparation, LoRA wrappers, and split manifest utilities without importing MLMP.
- `voxtell_sfda.adapter.load_prompts`: reused for prompt text or prompt-file parsing in the new DHGA entrypoint.
- Existing VoxTell integration locations identified: `VoxTellSFDAAdapter.__init__` initializes `VoxTellPredictor`, embeds text prompts, loads NIfTI IO, and selects device.
- Existing shared forward locations identified: `_encoder_selected_skips`, `_forward_outputs_and_features`, `_decoder_selected_outputs`, and `_predict_volume_probabilities` contain encoder reuse, prompt decoder memory construction, decoder projection, and sliding-window aggregation logic.
- Existing LoRA utility identified: `voxtell_sfda/lora.py` wraps `nn.MultiheadAttention` and injects LoRA into `transformer_decoder.layers.*.multihead_attn` for cross-attention or `self_attn` for self-attention.

## DHGA Modules

- `dhga/config.py`: validates all DHGA-prefixed method fields in one place.
- `dhga/shared_voxtell.py`: defines `SharedVoxTellFeatures` and `SharedEncoderOnce`; experts consume explicit feature caches rather than hooks.
- `dhga/experts/semantic_expert.py`: semantic expert wrapper for prompt-decoder-biased logits, intended to use cross-attention LoRA in `transformer_decoder.layers.*.multihead_attn`.
- `dhga/experts/appearance_expert.py`: residual 3D feature adapters on selected shared skip features. Each adapter uses GroupNorm, channel bottleneck plus depthwise 3D convolution, and a zero-initialized residual output. Feature dropout is applied once in `adapt_features()` and selected feature replacement uses explicit `selected_feature_idx` metadata.
- `dhga/routing/disagreement_router.py`: lightweight spatial router from semantic probability, appearance probability, low-dimensional visual context, stable foreground/background, and disagreement to per-voxel `w_sem(x)`, `w_app(x)`, `w_geo(x)`, consensus supervision weight, geometry disagreement weight, and initial fusion probability.
- `dhga/geometry/sdf.py`: SDF uses one global convention: inside is negative, outside is positive.
- `dhga/geometry/ray_sampler.py`: 3D ray sampling converts z,y,x voxel points to x,y,z `grid_sample` coordinates with `align_corners=True`; offsets are in millimeters and divided by per-axis spacing.
- `dhga/geometry/boundary_corruption.py`: creates inward and outward SDF perturbations with recovery targets derived by sign convention.
- `dhga/geometry/transport_head.py`: lightweight ray-token head with an explicit zero-displacement center bias and fallback for fully invalid rays.
- `dhga/geometry/boundary_points.py`: extracts zero-level-set surface points using `dhga_surface_tolerance_mm`; empty and full masks produce no valid surface. Sparse displacements are diffused back with a physical millimeter Gaussian kernel using `scatter_add_` accumulation, and can return dense valid weights for final gating.
- `dhga/voxtell_model.py`: real VoxTell-backed DHGA model. One encoder pass builds a shared feature cache; semantic output enables prompt-decoder LoRA, appearance output disables semantic LoRA and uses residual skip adapters, and anchor/baseline output disables both LoRA and appearance adapters. Geometry receives image/probability/SDF/geometry-gate/ray-offset evidence, prompt embeddings, and explicitly passed projected VoxTell intermediate visual features.
- `dhga/trainer.py`: implements real Stage A/B/C/D training loops over unlabeled NIfTI volumes. Labels are not loaded by the trainer. Stage A calls the model baseline path directly. Stage C returns raw boundary-recovery and minimal-transport losses and combines them exactly once in the training step. Stage D separately combines its router-target loss, Stage C auxiliary recovery, equivariance, prompt ranking if enabled, and weak natural minimal transport. Stage D reuses the same-patch teacher output, student output, and teacher SDF for its Stage C auxiliary path to avoid redundant encoder and SDF work. Stage C/D patch ordering is guided by EMA/student pseudo-boundary scores rather than GT. `--init_checkpoint` transfers Stage B to C and Stage C to D; after init loading the EMA teacher is hard-synchronized to the loaded model. `--resume_checkpoint` restores model, optimizer, lightweight EMA, AMP scaler, epoch, global step, and RNG, and same-stage resume is enforced.
- `dhga/evaluation.py`: evaluation-only sliding prediction. It aggregates semantic/app probabilities and low-dimensional projected VoxTell visual features in crop space, applies one SDF geometry correction, restores original space, saves NIfTI, and reads GT only for optional metrics including Dice, IoU, precision, recall, FP/FN voxels, volume ratio, connected components, oracle union/intersection, expert complementarity, and disagreement statistics.
- `dhga/inference.py`: owns the unified final geometry fusion function used by model forward, complete-volume evaluation, and mask finalization wrappers.
- `dhga/losses.py`: cross-supervision is weighted by stable consensus and downweighted by disagreement, so high-disagreement voxels are not forced into agreement.
- `dhga/checkpoint.py`: DHGA checkpoint payloads are versioned and loaded strictly; non-DHGA checkpoints are refused.

## Tensor Shapes

- Volume/image tensors: `[B, C, D, H, W]`.
- Expert logits/probabilities: `[B, 1, D, H, W]` for binary foreground prompts in the first implementation.
- SDF: `[B, 1, D, H, W]`, negative inside, positive outside.
- Normals: `[B, 1, D, H, W, 3]` in z,y,x axis order.
- Ray points: `[B, N, 3]` in z,y,x voxel indices.
- Ray samples: `[B, N, K, C]`, where `K = 2 * radius / step + 1`.
- Ray tokens: `[B, N, K, 31]` by default: image intensity, semantic probability, appearance probability, disagreement, fused probability, SDF value, 8-D projected VoxTell visual feature, ray offset, and a 16-D projected prompt embedding.
- Sparse displacement: `[B, N]` in millimeters, scattered to dense `[B, 1, D, H, W]` in the narrow band.

## Gradient Flow And Freezing

Default DHGA configuration sets `dhga_freeze_voxtell=True`. Trainable parameters are intended to be:

- Stage B trains semantic prompt-decoder LoRA, appearance residual adapters, and the spatial router warm-up parameters.
- Stage C freezes region experts and trains geometry visual projection, ray prompt projection, and transport head.
- Stage D freezes region experts by default and trains spatial router plus geometry modules.
- optional EMA teacher copies are updated without gradient.
- EMA teacher is lightweight: it shadows trainable DHGA parameters only and temporarily applies them to the student model for teacher forward. Frozen VoxTell backbone/decoder weights are shared.

Frozen VoxTell encoder, text encoder, original prompt decoder base weights, and original segmentation decoder weights should not receive gradients. `trainable_parameter_summary` prints module-level trainable parameter names and counts for smoke/dry-run checks.

## Geometry Sign Convention

`mask_to_sdf(mask)` returns `outside_distance - inside_distance`.

- Inside foreground: negative SDF.
- Outside background: positive SDF.
- Surface outward normal: normalized gradient of SDF.
- Positive displacement means moving the boundary outward and is applied as `phi_new = phi - displacement`.
- Outward corruption has positive perturbation and therefore requires negative recovery displacement.
- Inward corruption has negative perturbation and therefore requires positive recovery displacement.

## Training Stages

- Stage A: forced VoxTell baseline checks through `baseline_forward()`, independent of the `dhga_enabled` CLI default.
- Stage B: semantic and appearance expert warm-up with frozen VoxTell anchor preservation, weak/strong intensity consistency, and stable-consensus cross supervision.
- Stage C: self-supervised geometry pretraining from bidirectional SDF perturbation recovery. The target recovery displacement is derived from the known SDF perturbation sign.
- Stage D: EMA-teacher-derived router target supervision plus one-step geometry decision, Stage C recovery retained with weights applied once, dense displacement equivariance on common teacher-SDF boundary points, and minimal transport as a weak regularizer. Prompt ranking is disabled by default (`dhga_prompt_ranking_weight=0`) until a real visual-text inside/outside ranking implementation is used.

## Spacing And Category Scope

Reader spacing is read from NIfTI properties and used as returned because the reader properties already match array axis order. The code does not reverse xyz to zyx. The same physical spacing is used by SDF, normals, rays, boundary perturbation, and dense displacement diffusion. Current DHGA geometry is explicitly single-class binary adaptation; model construction raises if geometry is enabled with multiple prompts. Multi-class support should build per-class SDFs, boundary point sets, prompt-conditioned ray tokens, and losses.

## Router And Geometry Gate

The spatial router predicts `w_sem(x)`, `w_app(x)`, and `w_geo(x)`. Region probability is computed by renormalizing only `w_sem` and `w_app`, so `w_geo` cannot suppress foreground probability by stealing mass. The untrained router initializes near an even semantic/appearance blend with a small geometry gate. `w_geo` is used as geometry gate and contributes to boundary point sampling, ray tokens, displacement magnitude, and final narrow-band replacement. Evaluation computes router output with the same low-dimensional visual context used by training.

## Configuration Override Semantics

When `--config_json` is used, only CLI options explicitly present in the command override the JSON values. This includes stage, data paths, checkpoints, epochs, AMP, losses, and DHGA module settings, so commands cannot silently run the stage stored in the JSON when the user passes a different `--dhga_stage`.

## External References

- VoxTell official code (`https://github.com/MIC-DKFZ/VoxTell`): predictor/model interface and frozen foundation-model role.
- User MLMP-VoxTell code (`https://github.com/zhaoyue111-zz/mlmp`): local VoxTell predictor initialization, prompt embedding, NIfTI reading, sliding-window inference, LoRA injection, and checkpoint/logging patterns.
- CCT (`https://github.com/yassouali/CCT`): shared feature cache with heterogeneous heads as a design idea only.
- CPS (`https://github.com/charlesCXK/TorchSemiSeg`): cross-pseudo-supervision data flow, restricted here to stable consensus.
- UniMatch (`https://github.com/LiheYoung/UniMatch`): weak/strong consistency concept, without 2D CutMix.
- MC-Net/MC-Net+ (`https://github.com/ycwu1997/MC-Net`): heterogeneous branch motivation, reimplemented as semantic LoRA versus appearance skip adapters.
- DTC (`https://github.com/Luoxd1996/DTC`): SDF representation idea, reimplemented with DHGA sign and displacement rules.
- PyTorch `grid_sample` docs: coordinate order and normalization behavior for 3D sampling.
- SciPy `distance_transform_edt` docs: CPU SDF construction with spacing.
- Self-Supervised Correction Learning (`https://github.com/ReaFly/SemiMedSeg`): known-error recovery idea, reimplemented as bidirectional SDF perturbation recovery.

## Known Risks Before Real Data

- The real VoxTell integration path compiles and is importable, but still needs a user-run GPU data smoke run with actual checkpoint/model paths.
- Sliding-window feature aggregation for geometry can be memory-heavy if high-resolution features are retained.
- Semantic and appearance experts can collapse if stable consensus masks are too broad; diagnostics record expert correlation and disagreement ratio.
- Geometry can collapse to zero displacement if minimal transport is overweighted; default minimal transport weight is intentionally small.
- Teacher SDF boundaries are pseudo-boundaries, not GT; noisy stable consensus can teach wrong local corrections.
- DHGA checkpoints are strict and will not silently load legacy MLMP LoRA checkpoints.
