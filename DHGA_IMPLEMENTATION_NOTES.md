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
- `dhga/experts/appearance_expert.py`: residual 3D feature adapters on selected shared skip features. Each adapter is channel bottleneck plus depthwise 3D convolution and is zero-initialized at the residual output.
- `dhga/routing/disagreement_router.py`: continuous router from semantic probability, appearance probability, stability weights, and disagreement to stable foreground/background, disagreement weight, and initial fusion probability.
- `dhga/geometry/sdf.py`: SDF uses one global convention: inside is negative, outside is positive.
- `dhga/geometry/ray_sampler.py`: 3D ray sampling converts z,y,x voxel points to x,y,z `grid_sample` coordinates with `align_corners=True`; offsets are in millimeters and divided by per-axis spacing.
- `dhga/geometry/boundary_corruption.py`: creates inward and outward SDF perturbations with recovery targets derived by sign convention.
- `dhga/geometry/transport_head.py`: lightweight ray-token head initialized to a uniform offset distribution, making zero expected displacement when offsets are symmetric.
- `dhga/geometry/boundary_points.py`: extracts physical narrow-band boundary points and scatters sparse predicted displacements back to a dense narrow-band displacement field.
- `dhga/voxtell_model.py`: real VoxTell-backed DHGA model. One encoder pass builds a shared feature cache; semantic output uses prompt-decoder LoRA, appearance output uses residual skip adapters and the frozen decoder path; geometry receives image/probability/SDF/ray offset evidence plus prompt embeddings.
- `dhga/trainer.py`: implements real Stage A/B/C/D training loops over unlabeled NIfTI volumes. Labels are not loaded by the trainer.
- `dhga/losses.py`: cross-supervision is weighted by stable consensus and downweighted by disagreement, so high-disagreement voxels are not forced into agreement.
- `dhga/checkpoint.py`: DHGA checkpoint payloads are versioned and loaded strictly; non-DHGA checkpoints are refused.

## Tensor Shapes

- Volume/image tensors: `[B, C, D, H, W]`.
- Expert logits/probabilities: `[B, 1, D, H, W]` for binary foreground prompts in the first implementation.
- SDF: `[B, 1, D, H, W]`, negative inside, positive outside.
- Normals: `[B, 1, D, H, W, 3]` in z,y,x axis order.
- Ray points: `[B, N, 3]` in z,y,x voxel indices.
- Ray samples: `[B, N, K, C]`, where `K = 2 * radius / step + 1`.
- Ray tokens: `[B, N, K, 23]` by default: image intensity, semantic probability, appearance probability, disagreement, fused probability, SDF value, ray offset, and a 16-D projected prompt embedding.
- Sparse displacement: `[B, N]` in millimeters, scattered to dense `[B, 1, D, H, W]` in the narrow band.

## Gradient Flow And Freezing

Default DHGA configuration sets `dhga_freeze_voxtell=True`. Trainable parameters are intended to be:

- semantic expert adapters in the prompt decoder cross-attention path;
- appearance residual feature adapters;
- the continuous disagreement router;
- the geometry transport head;
- optional EMA teacher copies are updated without gradient.
- prompt-conditioned ray token projection in the geometry path.

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

- Stage A: DHGA disabled or geometry disabled baseline checks. The expected behavior is identity relative to frozen VoxTell output when all DHGA adapters are zero/disabled.
- Stage B: semantic and appearance expert warm-up with frozen VoxTell anchor preservation, weak/strong intensity consistency, and stable-consensus cross supervision.
- Stage C: self-supervised geometry pretraining from bidirectional SDF perturbation recovery. The target recovery displacement is derived from the known SDF perturbation sign.
- Stage D: natural semantic-appearance disagreement routing plus one-step geometry decision with minimal transport regularization.

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
