# Minimal Student-Teacher LoRA SFDA

This directory is intentionally separate from the existing DHGA implementation.
It provides the smallest reusable student-teacher setup for binary segmentation:

- teacher: a frozen copy of the pretrained model, updated with EMA
- student: the same pretrained model with LoRA injected into existing attention modules
- trainable parameters: LoRA attention parameters only
- supervision: teacher output vs student output with BCE loss + Dice loss
- validation/test logging: Dice and binary mIoU

## Usage

```python
import torch

from sample_student_teacher import StudentTeacherLoRA
from sample_student_teacher.train_step import train_one_step

pretrained_voxtell = ...  # must return binary segmentation logits

model = StudentTeacherLoRA(
    pretrained_voxtell,
    rank=4,
    alpha=8.0,
    ema_decay=0.999,
)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
)

image = torch.randn(1, 1, 64, 64, 64)
metrics = train_one_step(model, optimizer, image)
print(metrics)  # loss, bce_loss, dice_loss, dice, miou
```

The wrapper accepts models that return logits directly, dictionaries containing
`logits`, `seg_logits`, or `mask_logits`, or objects with a `logits` attribute.

## DHGA/VoxTell runnable training

The runnable entry point follows the DHGA source project data strategy:

- split file: `worst_zeroshot_split_p0/worst_zeroshot_split.json`
- image root: `/data/zy/CT_MRI_DATA_3D/images/P0`
- label root: `/data/zy/CT_MRI_DATA_3D/labels/P0`
- NIfTI reader: `NibabelIOWithReorient`
- preprocessing, padding and sliding windows: VoxTell predictor methods

Run from the repository root:

```bash
python -m sample_student_teacher.train_voxtell_sfda \
  --voxtell_repo /data/zy/VoxTell_from_disk \
  --model_dir /data/zy/VoxTell_from_disk/model \
  --data_dir /data/zy/CT_MRI_DATA_3D/images/P0 \
  --split_manifest /data/zy/DHGA/worst_zeroshot_split_p0/worst_zeroshot_split.json \
  --val_label_dir /data/zy/CT_MRI_DATA_3D/labels/P0 \
  --prompts liver \
  --label_values 5 \
  --epochs 1 \
  --steps_per_volume 1
```

Each epoch writes `checkpoint_latest.pt`, `history.json`, and evaluation metrics
with `dice` and binary `miou` under `.save/simple_student_teacher`.
