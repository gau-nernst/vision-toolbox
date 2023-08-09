# https://arxiv.org/abs/2103.14030
# https://github.com/microsoft/Swin-Transformer

from __future__ import annotations

from functools import partial
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import torch_hub_download
from .base import _act, _norm
from .vit import MHA, MLP


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, int, int]:
    B, H, W, C = x.shape
    nH, nW = H // window_size, W // window_size
    x = x.view(B, nH, window_size, nW, window_size, C)
    x = x.transpose(2, 3).reshape(B * nH * nW, window_size * window_size, C)
    return x, nH, nW


def window_unpartition(x: Tensor, window_size: int, nH: int, nW: int) -> Tensor:
    B = x.shape[0] // (nH * nW)
    C = x.shape[2]
    x = x.view(B, nH, nW, window_size, window_size, C)
    x = x.transpose(2, 3).reshape(B, nH * window_size, nW * window_size, C)
    return x


class WindowAttention(MHA):
    def __init__(
        self, d_model: int, window_size: int, shift: int, n_heads: int, bias: bool = True, dropout: float = 0.0
    ) -> None:
        super().__init__(d_model, n_heads, bias, dropout)
        self.window_size = window_size
        self.shift = shift

        self.relative_pe_table = nn.Parameter(torch.empty(n_heads, (2 * window_size - 1) ** 2))
        nn.init.trunc_normal_(self.relative_pe_table, 0, 0.02)

        xy = torch.cartesian_prod(torch.arange(window_size), torch.arange(window_size))  # all possible (x,y) pairs
        diff = xy.unsqueeze(1) - xy.unsqueeze(0)  # difference between all (x,y) pairs
        index = (diff[:, :, 0] + window_size - 1) * (2 * window_size - 1) + diff[:, :, 1] + window_size - 1
        self.register_buffer("relative_pe_index", index.flatten(), False)
        self.relative_pe_index: Tensor

    def forward(self, x: Tensor) -> Tensor:
        if self.shift > 0:
            x = x.roll((self.shift, self.shift), (1, 2))
        x, nH, nW = window_partition(x, self.window_size)  # (B * nH * nW, win_size * win_size, C)

        attn_bias = self.relative_pe_table[:, self.relative_pe_index].unsqueeze(0)
        x = super().forward(x, attn_bias)

        x = window_unpartition(x, self.window_size, nH, nW)
        return x


class SwinBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        shift: int,
        mlp_ratio: float = 4.0,
        bias: bool = True,
        dropout: float = 0.0,
        norm: _norm = nn.LayerNorm,
        act: _act = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1 = norm(d_model)
        self.mha = WindowAttention(d_model, window_size, shift, n_heads, bias, dropout)
        self.norm2 = norm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), act)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mha(self.norm1(x))
        x = self.mlp(self.norm2(x))
        return x


class SwinTransformer(nn.Module):
    def __init__(self, d_model: int, n_layers: tuple[int, int, int, int]) -> None:
        super().__init__()

    @staticmethod
    def from_config(variant: str, pretrained: bool = False) -> SwinTransformer:
        d_model, n_layers = dict(
            T=(96, (2, 2, 6, 2)),
            S=(96, (2, 2, 18, 2)),
            B=(128, (2, 2, 18, 2)),
            L=(192, (2, 2, 18, 2)),
        )[variant]
        m = SwinTransformer(d_model, n_layers)
