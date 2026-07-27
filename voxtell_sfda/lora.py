import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRAMultiheadAttention(nn.Module):
    """LoRA wrapper for ``nn.MultiheadAttention`` with frozen base weights."""

    def __init__(
        self,
        base: nn.MultiheadAttention,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        adapt_q: bool = True,
        adapt_k: bool = True,
        adapt_v: bool = True,
        adapt_out: bool = True,
    ) -> None:
        super().__init__()
        if base._qkv_same_embed_dim is not True:
            raise ValueError("LoRA wrapper currently expects q/k/v to share embed_dim")
        if base.batch_first:
            raise ValueError("VoxTell decoder uses sequence-first attention; batch_first is not supported here")

        self.base = base
        self.embed_dim = base.embed_dim
        self.num_heads = base.num_heads
        self.dropout = nn.Dropout(dropout)
        self.rank = rank
        self.scaling = alpha / rank
        self.adapt_q = adapt_q
        self.adapt_k = adapt_k
        self.adapt_v = adapt_v
        self.adapt_out = adapt_out

        for param in self.base.parameters():
            param.requires_grad_(False)

        self.q_lora = self._make_lora() if adapt_q else None
        self.k_lora = self._make_lora() if adapt_k else None
        self.v_lora = self._make_lora() if adapt_v else None
        self.out_lora = self._make_lora() if adapt_out else None

    def _make_lora(self) -> nn.ModuleDict:
        down = nn.Linear(self.embed_dim, self.rank, bias=False)
        up = nn.Linear(self.rank, self.embed_dim, bias=False)
        nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        nn.init.zeros_(up.weight)
        return nn.ModuleDict({"down": down, "up": up})

    def _delta_weight(self, lora: nn.ModuleDict, dtype: torch.dtype, device: torch.device) -> Tensor:
        down = lora["down"].weight.to(device=device, dtype=dtype)
        up = lora["up"].weight.to(device=device, dtype=dtype)
        return (up @ down) * self.scaling

    def _merged_in_proj_weight(self) -> Tensor:
        weight = self.base.in_proj_weight
        deltas = []
        for lora in (self.q_lora, self.k_lora, self.v_lora):
            if lora is None:
                deltas.append(torch.zeros_like(weight[: self.embed_dim]))
            else:
                deltas.append(self._delta_weight(lora, weight.dtype, weight.device))
        return weight + torch.cat(deltas, dim=0)

    def _merged_out_proj_weight(self) -> Tensor:
        weight = self.base.out_proj.weight
        if self.out_lora is None:
            return weight
        return weight + self._delta_weight(self.out_lora, weight.dtype, weight.device)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask=None,
        need_weights: bool = True,
        attn_mask=None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        query = self.dropout(query)
        return F.multi_head_attention_forward(
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


def inject_lora_into_voxtell_decoder(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target: str = "cross",
) -> Dict[str, LoRAMultiheadAttention]:
    """Replace VoxTell transformer decoder attention modules with LoRA wrappers."""
    if target not in {"cross", "self", "both"}:
        raise ValueError("target must be one of: cross, self, both")

    injected = {}
    for layer_idx, layer in enumerate(model.transformer_decoder.layers):
        if target in {"self", "both"}:
            name = f"transformer_decoder.layers.{layer_idx}.self_attn"
            layer.self_attn = LoRAMultiheadAttention(
                layer.self_attn, rank=rank, alpha=alpha, dropout=dropout
            )
            injected[name] = layer.self_attn
        if target in {"cross", "both"}:
            name = f"transformer_decoder.layers.{layer_idx}.multihead_attn"
            layer.multihead_attn = LoRAMultiheadAttention(
                layer.multihead_attn, rank=rank, alpha=alpha, dropout=dropout
            )
            injected[name] = layer.multihead_attn
    return injected


def mark_only_lora_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRAMultiheadAttention):
            for name, param in module.named_parameters():
                if not name.startswith("base."):
                    param.requires_grad_(True)


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (param for param in model.parameters() if param.requires_grad)


def lora_state_dict(model: nn.Module) -> Dict[str, Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "q_lora." in name
        or "k_lora." in name
        or "v_lora." in name
        or "out_lora." in name
    }


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if "lora" in key]
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected}")
    missing_lora = [key for key in missing if "lora" in key]
    if missing_lora:
        raise RuntimeError(f"Missing LoRA keys: {missing_lora}")
