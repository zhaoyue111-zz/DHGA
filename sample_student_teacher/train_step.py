from __future__ import annotations

import torch
from torch import Tensor

from .model import StudentTeacherLoRA


def train_one_step(
    model: StudentTeacherLoRA,
    optimizer: torch.optim.Optimizer,
    image: Tensor,
    amp: bool = True,
) -> dict[str, float]:
    device_type = image.device.type
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type, enabled=amp and device_type == "cuda"):
        loss, metrics = model.training_step(image)
    loss.backward()
    optimizer.step()
    model.update_teacher_ema()
    return metrics
