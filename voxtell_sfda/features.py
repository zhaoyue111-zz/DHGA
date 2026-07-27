from typing import List, Tuple

import torch
from torch import nn


@torch.no_grad()
def encoder_stage_embeddings(network: nn.Module, image: torch.Tensor) -> List[torch.Tensor]:
    """Return VoxTell encoder stage embeddings, one tensor per 3D resolution."""
    return list(network.encoder(image))


@torch.no_grad()
def pooled_pseudo_cls_tokens(network: nn.Module, image: torch.Tensor) -> List[torch.Tensor]:
    """Build CLS-like global tokens by pooling each CNN encoder stage."""
    stages = encoder_stage_embeddings(network, image)
    return [stage.mean(dim=(2, 3, 4)) for stage in stages]


def stage_shapes(stages: List[torch.Tensor]) -> List[Tuple[int, ...]]:
    return [tuple(stage.shape) for stage in stages]

