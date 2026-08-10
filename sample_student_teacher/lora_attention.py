from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRAMultiheadAttention(nn.Module):
    """LoRA adapter for a frozen nn.MultiheadAttention module."""

    def __init__(
        self,
        base: nn.MultiheadAttention,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not base._qkv_same_embed_dim:
            raise ValueError("LoRA wrapper expects q/k/v to share embed_dim")
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.embed_dim = int(base.embed_dim)

        for param in self.base.parameters():
            param.requires_grad_(False)

        self.q_lora = self._make_lora()
        self.k_lora = self._make_lora()
        self.v_lora = self._make_lora()
        self.out_lora = self._make_lora()

    def _make_lora(self) -> nn.ModuleDict:
        down = nn.Linear(self.embed_dim, self.rank, bias=False)
        up = nn.Linear(self.rank, self.embed_dim, bias=False)
        nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        nn.init.zeros_(up.weight)
        return nn.ModuleDict({"down": down, "up": up})

    def _delta(self, lora: nn.ModuleDict, reference: Tensor) -> Tensor:
        down = lora["down"].weight.to(device=reference.device, dtype=reference.dtype)
        up = lora["up"].weight.to(device=reference.device, dtype=reference.dtype)
        return (up @ down) * self.scaling

    def _merged_in_proj_weight(self) -> Tensor:
        weight = self.base.in_proj_weight
        return weight + torch.cat(
            [
                self._delta(self.q_lora, weight),
                self._delta(self.k_lora, weight),
                self._delta(self.v_lora, weight),
            ],
            dim=0,
        )

    def _merged_out_proj_weight(self) -> Tensor:
        weight = self.base.out_proj.weight
        return weight + self._delta(self.out_lora, weight)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        query = self.dropout(query)
        if self.base.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        out, weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.base.embed_dim,
            num_heads=self.base.num_heads,
            in_proj_weight=self._merged_in_proj_weight(),
            in_proj_bias=self.base.in_proj_bias,
            bias_k=self.base.bias_k,
            bias_v=self.base.bias_v,
            add_zero_attn=self.base.add_zero_attn,
            dropout_p=self.base.dropout if self.training else 0.0,
            out_proj_weight=self._merged_out_proj_weight(),
            out_proj_bias=self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.base.batch_first:
            out = out.transpose(0, 1)
        return out, weights


def inject_lora_attention(
    module: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    name_keywords: Iterable[str] = ("attn", "attention"),
) -> dict[str, LoRAMultiheadAttention]:
    """Replace existing MultiheadAttention modules whose names look like attention."""

    injected: dict[str, LoRAMultiheadAttention] = {}
    keywords = tuple(key.lower() for key in name_keywords)

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.MultiheadAttention) and any(key in full_name.lower() for key in keywords):
                wrapped = LoRAMultiheadAttention(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, wrapped)
                injected[full_name] = wrapped
            else:
                visit(child, full_name)

    visit(module)
    if not injected:
        raise RuntimeError("No nn.MultiheadAttention modules matched for LoRA injection")
    return injected


def mark_only_lora_trainable(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)
    for child in module.modules():
        if isinstance(child, LoRAMultiheadAttention):
            for name, param in child.named_parameters():
                if not name.startswith("base."):
                    param.requires_grad_(True)
