# https://arxiv.org/abs/2010.11929
# https://arxiv.org/abs/2106.10270
# https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


__all__ = ["ViT"]


configs = dict(
    Ti=dict(n_layers=12, d_model=192, n_heads=3),
    S=dict(n_layers=12, d_model=384, n_heads=6),
    B=dict(n_layers=12, d_model=768, n_heads=12),
    L=dict(n_layers=24, d_model=1024, n_heads=16),
    H=dict(n_layers=32, d_model=1280, n_heads=16),
)


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 3, bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.in_proj(x).chunk(3, -1)
        q, k, v = map(lambda x: x.unflatten(-1, (self.n_heads, -1)).transpose(-2, -3), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        out = out.transpose(-2, -3).flatten(-2)
        out = self.out_proj(out)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = True, dropout: float = 0.0, norm_eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, norm_eps)
        self.mha = MHA(d_model, n_heads, bias, dropout)
        self.norm2 = nn.LayerNorm(d_model, norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        patch_size: int,
        img_size: int,
        cls_token: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if cls_token else None

        pe_size = (img_size // patch_size) ** 2
        if cls_token:
            pe_size += 1
        self.pe = nn.Parameter(torch.empty(1, pe_size, d_model))
        nn.init.normal_(self.pe, 0, 0.02)

        self.layers = nn.Sequential()
        for _ in range(n_layers):
            self.layers.append(TransformerEncoderLayer(d_model, n_heads, bias, dropout))
        self.norm = nn.LayerNorm(d_model, norm_eps)

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.patch_embed(imgs)
        out = out.flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        if self.cls_token is not None:
            out = torch.cat([self.cls_token.expand(out.shape[0], -1, -1), out], 1)
        out = out + self.pe
        out = self.layers(out)
        out = self.norm(out)
        out = out[:, 0] if self.cls_token is not None else out.mean(1)
        return out

    @torch.no_grad()
    def resize_pe(self, size: int, interpolation_mode: str = "bicubic") -> None:
        pe = self.pe if self.cls_token is None else self.pe[:, 1:]

        old_size = int(pe.shape[1] ** 0.5)
        new_size = size // self.patch_embed.weight.shape[2]
        pe = pe.unflatten(1, (old_size, old_size)).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, (new_size, new_size), mode=interpolation_mode)
        pe = pe.permute(0, 2, 3, 1).flatten(1, 2)

        if self.cls_token is not None:
            pe = torch.cat((self.pe[:, 0:1], pe), 1)

        self.pe = nn.Parameter(pe)

    @staticmethod
    def from_config(variant: str, patch_size: int, img_size: int) -> ViT:
        return ViT(**configs[variant], patch_size=patch_size, img_size=img_size)

    # weights from https://github.com/google-research/vision_transformer
    @torch.no_grad()
    @staticmethod
    def from_jax_weights(path: str) -> ViT:
        jax_weights: Mapping[str, np.ndarray] = np.load(path)

        def get_w(key: str) -> Tensor:
            return torch.from_numpy(jax_weights[key])

        n_layers = 1
        while True:
            if f"Transformer/encoderblock_{n_layers}/LayerNorm_0/bias" not in jax_weights:
                break
            n_layers += 1

        d_model = jax_weights["cls"].shape[-1]
        n_heads = jax_weights["Transformer/encoderblock_0/MultiHeadDotProductAttention_1/key/bias"].shape[0]
        patch_size = jax_weights["embedding/kernel"].shape[0]
        img_size = int((jax_weights["Transformer/posembed_input/pos_embedding"].shape[1] - 1) ** 0.5) * patch_size

        m = ViT(n_layers, d_model, n_heads, patch_size, img_size)

        m.cls_token.copy_(get_w("cls"))
        m.patch_embed.weight.copy_(get_w("embedding/kernel").permute(3, 2, 0, 1))
        m.patch_embed.bias.copy_(get_w("embedding/bias"))
        m.pe.copy_(get_w("Transformer/posembed_input/pos_embedding"))

        for idx, layer in enumerate(m.layers):
            prefix = f"Transformer/encoderblock_{idx}/"
            mha_prefix = prefix + "MultiHeadDotProductAttention_1/"

            layer.norm1.weight.copy_(get_w(prefix + "LayerNorm_0/scale"))
            layer.norm1.bias.copy_(get_w(prefix + "LayerNorm_0/bias"))
            w = torch.stack([get_w(mha_prefix + x + "/kernel") for x in ["query", "key", "value"]], 1)
            b = torch.stack([get_w(mha_prefix + x + "/bias") for x in ["query", "key", "value"]], 0)
            layer.mha.in_proj.weight.copy_(w.flatten(1).T)
            layer.mha.in_proj.bias.copy_(b.flatten())
            layer.mha.out_proj.weight.copy_(get_w(mha_prefix + "out/kernel").flatten(0, 1).T)
            layer.mha.out_proj.bias.copy_(get_w(mha_prefix + "out/bias"))

            layer.norm2.weight.copy_(get_w(prefix + "LayerNorm_2/scale"))
            layer.norm2.bias.copy_(get_w(prefix + "LayerNorm_2/bias"))
            layer.mlp[0].weight.copy_(get_w(prefix + "MlpBlock_3/Dense_0/kernel").T)
            layer.mlp[0].bias.copy_(get_w(prefix + "MlpBlock_3/Dense_0/bias"))
            layer.mlp[2].weight.copy_(get_w(prefix + "MlpBlock_3/Dense_1/kernel").T)
            layer.mlp[2].bias.copy_(get_w(prefix + "MlpBlock_3/Dense_1/bias"))

        m.norm.weight.copy_(get_w("Transformer/encoder_norm/scale"))
        m.norm.bias.copy_(get_w("Transformer/encoder_norm/bias"))
        return m
