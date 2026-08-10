from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn

from .lora_attention import inject_lora_attention, mark_only_lora_trainable
from .losses import bce_dice_loss
from .metrics import binary_dice_miou


def _logits_from_output(output: Any, output_key: str = "logits") -> Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        if output_key in output:
            return output[output_key]
        for key in ("logits", "seg_logits", "mask_logits"):
            if key in output:
                return output[key]
    if hasattr(output, output_key):
        value = getattr(output, output_key)
        if torch.is_tensor(value):
            return value
    raise TypeError("Model output must be a tensor, a dict containing logits, or an object with logits")


class StudentTeacherLoRA(nn.Module):
    """SFDA wrapper: frozen pretrained teacher, LoRA-only student, EMA teacher update."""

    def __init__(
        self,
        pretrained_model: nn.Module,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        ema_decay: float = 0.999,
        output_key: str = "logits",
    ) -> None:
        super().__init__()
        self.student = pretrained_model
        self.output_key = output_key
        self.ema_decay = float(ema_decay)

        self.injected_lora = inject_lora_attention(
            self.student,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        mark_only_lora_trainable(self.student)

        self.teacher = copy.deepcopy(self.student)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def student_logits(self, image: Tensor) -> Tensor:
        return _logits_from_output(self.student(image), self.output_key)

    @torch.no_grad()
    def teacher_prob(self, image: Tensor) -> Tensor:
        was_training = self.teacher.training
        self.teacher.eval()
        logits = _logits_from_output(self.teacher(image), self.output_key)
        self.teacher.train(was_training)
        return logits.float().sigmoid()

    def forward(self, image: Tensor) -> Tensor:
        return self.student_logits(image)

    def training_step(self, image: Tensor) -> tuple[Tensor, dict[str, float]]:
        teacher_prob = self.teacher_prob(image)
        student_logits = self.student_logits(image)
        loss, metrics = bce_dice_loss(student_logits, teacher_prob)
        metrics.update(binary_dice_miou(student_logits.sigmoid(), teacher_prob))
        return loss, metrics

    @torch.no_grad()
    def update_teacher_ema(self) -> None:
        teacher_params = dict(self.teacher.named_parameters())
        for name, student_param in self.student.named_parameters():
            teacher_param = teacher_params.get(name)
            if teacher_param is None:
                continue
            teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1.0 - self.ema_decay)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.student.named_parameters() if param.requires_grad]
