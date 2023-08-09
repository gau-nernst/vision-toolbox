# https://arxiv.org/abs/2103.14030
# https://github.com/microsoft/Swin-Transformer

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn

from ..utils import torch_hub_download
from .base import BaseBackbone, _act, _norm
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
        self,
        input_size: int,
        d_model: int,
        n_heads: int,
        window_size: int = 7,
        shift: bool = False,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(d_model, n_heads, bias, dropout)
        self.input_size = input_size
        self.window_size = window_size

        if shift:
            self.shift = window_size // 2

            img_mask = torch.zeros(1, input_size, input_size, 1)
            slices = (slice(0, -window_size), slice(-window_size, -self.shift), slice(-self.shift, None))
            for i, (h_slice, w_slice) in enumerate(itertools.product(slices, slices)):
                img_mask[0, h_slice, w_slice, 0] = i

            windows_mask, _, _ = window_partition(img_mask, window_size)  # (nH * nW, win_size * win_size, 1)
            windows_mask = windows_mask.transpose(1, 2)  # (nH * nW, 1, win_size * win_size)
            attn_mask = windows_mask.unsqueeze(2) - windows_mask.unsqueeze(3)
            self.register_buffer("attn_mask", (attn_mask != 0) * (-100), False)
            self.attn_mask: Tensor

        else:
            self.shift = 0
            self.attn_mask = None

        self.relative_pe_table = nn.Parameter(torch.empty(1, n_heads, (2 * window_size - 1) ** 2))
        nn.init.trunc_normal_(self.relative_pe_table, 0, 0.02)

        xy = torch.cartesian_prod(torch.arange(window_size), torch.arange(window_size))  # all possible (x,y) pairs
        diff = xy.unsqueeze(1) - xy.unsqueeze(0)  # difference between all (x,y) pairs
        index = (diff[:, :, 0] + window_size - 1) * (2 * window_size - 1) + diff[:, :, 1] + window_size - 1
        self.register_buffer("relative_pe_index", index, False)
        self.relative_pe_index: Tensor

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.input_size, (x.shape[1], self.input_size)
        attn_bias = self.relative_pe_table[..., self.relative_pe_index]
        if self.shift > 0:
            x = x.roll((self.shift, self.shift), (1, 2))
            attn_bias = attn_bias + self.attn_mask

        x, nH, nW = window_partition(x, self.window_size)  # (B * nH * nW, win_size * win_size, C)
        x = super().forward(x, attn_bias)
        x = window_unpartition(x, self.window_size, nH, nW)

        if self.shift > 0:
            x = x.roll((-self.shift, -self.shift), (1, 2))
        return x


class SwinBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_heads: int,
        window_size: int = 7,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        bias: bool = True,
        dropout: float = 0.0,
        norm: _norm = nn.LayerNorm,
        act: _act = nn.GELU,
    ) -> None:
        super().__init__()
        self.norm1 = norm(d_model)
        self.mha = WindowAttention(input_size, d_model, n_heads, window_size, shift, bias, dropout)
        self.norm2 = norm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), dropout, act)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mha(self.norm1(x))
        x = self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, d_model: int, norm: _norm = nn.LayerNorm) -> None:
        super().__init__()
        self.norm = norm(d_model * 4)
        self.reduction = nn.Linear(d_model * 4, d_model * 2, False)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // 2, 2, W // 2, 2, C).transpose(2, 3).flatten(-3)
        x = self.reduction(self.norm(x))
        x = x.view(B, H // 2, W // 2, C * 2)
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size: int,
        d_model: int,
        n_heads: int,
        depths: tuple[int, ...],
        patch_size: int = 4,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        bias: bool = True,
        dropout: float = 0.0,
        norm: _norm = nn.LayerNorm,
        act: _act = nn.GELU,
    ) -> None:
        assert img_size % patch_size == 0
        assert img_size % window_size == 0
        assert d_model % n_heads == 0
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.norm = norm(d_model)
        self.dropout = nn.Dropout(dropout)

        input_size = img_size // patch_size
        self.stages = nn.Sequential()
        for i, depth in enumerate(depths):
            stage = nn.Sequential()
            for i in range(depth):
                blk = SwinBlock(input_size, d_model, n_heads, window_size, i % 2, mlp_ratio, bias, dropout, norm, act)
                stage.append(blk)

            if i < len(depths) - 1:
                stage.append(PatchMerging(d_model, norm))
                input_size //= 2
                d_model *= 2
                n_heads *= 2

            self.stages.append(stage)

        self.head_norm = norm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(self.norm(self.patch_embed(x).permute(0, 2, 3, 1)))
        x = self.stages(x)
        x = self.head_norm(x).mean((1, 2))
        return x

    @staticmethod
    def from_config(variant: str, img_size: int, pretrained: bool = False) -> SwinTransformer:
        d_model, n_heads, depths = dict(
            T=(96, 3, (2, 2, 6, 2)),
            S=(96, 3, (2, 2, 18, 2)),
            B=(128, 4, (2, 2, 18, 2)),
            L=(192, 6, (2, 2, 18, 2)),
        )[variant]
        m = SwinTransformer(img_size, d_model, n_heads, depths)

        return m
