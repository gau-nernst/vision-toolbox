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
from .vit import MHA


class WindowAttention(MHA):
    def __init__(self, d_model: int, window_size: int, n_heads: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__(d_model, n_heads, bias, dropout)
        self.relative_pe_table = nn.Parameter(torch.empty(n_heads, (2 * window_size - 1) ** 2))
        nn.init.trunc_normal_(self.relative_pe_table, 0, 0.02)

        xy = torch.cartesian_prod(torch.arange(window_size), torch.arange(window_size))  # all possible (x,y) pairs
        diff = xy.unsqueeze(1) - xy.unsqueeze(0)  # difference between all (x,y) pairs
        index = (diff[:, :, 0] + window_size - 1) * (2 * window_size - 1) + diff[:, :, 1] + window_size - 1
        self.register_buffer("relative_pe_index", index.flatten(), False)
        self.relative_pe_index: Tensor

    def forward(self, x: Tensor) -> Tensor:
        attn_bias = self.relative_pe_table[:, self.relative_pe_index].unsqueeze(0)
        return super().forward(x, attn_bias)
