from __future__ import annotations

import torch
from torch import Tensor, nn

from dhga.shared_voxtell import SharedVoxTellFeatures


class SemanticExpert(nn.Module):
    """Prompt-decoder-biased expert.

    The trainable parameters live in the wrapped VoxTell prompt decoder, typically
    LoRA modules inserted at cross-attention sites. The expert consumes an already
    computed shared feature cache.
    """

    def __init__(self, decoder_forward: nn.Module | None = None) -> None:
        super().__init__()
        self.decoder_forward = decoder_forward

    def forward(self, features: SharedVoxTellFeatures, logits: Tensor | None = None) -> dict[str, Tensor]:
        if logits is None:
            if self.decoder_forward is None:
                raise ValueError("SemanticExpert requires logits or a decoder_forward module")
            logits = self.decoder_forward(features)
        probs = logits.float().sigmoid()
        return {"logits": logits, "prob": probs}
