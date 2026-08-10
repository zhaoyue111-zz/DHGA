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
