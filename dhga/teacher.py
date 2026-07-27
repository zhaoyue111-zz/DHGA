from __future__ import annotations

import copy

from torch import nn


class EMATeacher:
    def __init__(self, student: nn.Module, decay: float = 0.99) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(student).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def update(self, student: nn.Module) -> None:
        student_state = student.state_dict()
        teacher_state = self.module.state_dict()
        for key, value in teacher_state.items():
            if key in student_state and value.is_floating_point():
                value.mul_(self.decay).add_(student_state[key].detach(), alpha=1.0 - self.decay)
            elif key in student_state:
                value.copy_(student_state[key])
