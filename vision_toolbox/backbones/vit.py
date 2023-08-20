# https://arxiv.org/abs/2010.11929
# https://arxiv.org/abs/2106.10270
# https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

from __future__ import annotations

from functools import partial
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..components import LayerScale, StochasticDepth
from ..utils import torch_hub_download
from .base import _act, _norm


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias)
        self.k_proj = nn.Linear(d_model, d_model, bias)
        self.v_proj = nn.Linear(d_model, d_model, bias)
        self.out_proj = nn.Linear(d_model, d_model, bias)
        self.n_heads = n_heads
        self.dropout = dropout
        self.scale = (d_model // n_heads) ** (-0.5)

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        q = self.q_proj(x).unflatten(-1, (self.n_heads, -1)).transpose(-2, -3)  # (B, n_heads, L, head_dim)
        k = self.k_proj(x).unflatten(-1, (self.n_heads, -1)).transpose(-2, -3)
        v = self.v_proj(x).unflatten(-1, (self.n_heads, -1)).transpose(-2, -3)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, attn_bias, self.dropout if self.training else 0.0)
        else:
            attn = q @ (k * self.scale).transpose(-1, -2)
            if attn_bias is not None:
                attn = attn + attn_bias
            out = F.dropout(torch.softmax(attn, -1), self.dropout, self.training) @ v

        out = out.transpose(-2, -3).flatten(-2)
        out = self.out_proj(out)
        return out


class MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: float, dropout: float = 0.0, act: _act = nn.GELU) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.act = act()
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)


class ViTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float | None = None,
        stochastic_depth: float = 0.0,
        norm: _norm = partial(nn.LayerNorm, eps=1e-6),
        act: _act = nn.GELU,
    ) -> None:
        super().__init__()
        self.mha = nn.Sequential(
            norm(d_model),
            MHA(d_model, n_heads, bias, dropout),
            LayerScale(d_model, layer_scale_init) if layer_scale_init is not None else nn.Identity(),
            StochasticDepth(stochastic_depth),
        )
        self.mlp = nn.Sequential(
            norm(d_model),
            MLP(d_model, int(d_model * mlp_ratio), dropout, act),
            LayerScale(d_model, layer_scale_init) if layer_scale_init is not None else nn.Identity(),
            StochasticDepth(stochastic_depth),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(x)
        x = x + self.mlp(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int,
        n_heads: int,
        patch_size: int,
        img_size: int,
        cls_token: bool = True,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float | None = None,
        stochastic_depth: float = 0.0,
        norm: _norm = partial(nn.LayerNorm, eps=1e-6),
        act: _act = nn.GELU,
    ) -> None:
        assert img_size % patch_size == 0
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if cls_token else None

        pe_size = (img_size // patch_size) ** 2
        if cls_token:
            pe_size += 1
        self.pe = nn.Parameter(torch.empty(1, pe_size, d_model))
        nn.init.normal_(self.pe, 0, 0.02)

        self.layers = nn.Sequential()
        for _ in range(depth):
            block = ViTBlock(d_model, n_heads, bias, mlp_ratio, dropout, layer_scale_init, stochastic_depth, norm, act)
            self.layers.append(block)

        self.norm = norm(d_model)

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        if self.cls_token is not None:
            out = torch.cat([self.cls_token.expand(out.shape[0], -1, -1), out], 1)
        out = self.layers(out + self.pe)
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
    def from_config(variant: str, patch_size: int, img_size: int, pretrained: bool = False) -> ViT:
        d_model, depth, n_heads = dict(
            Ti=(192, 12, 3),
            S=(384, 12, 6),
            B=(768, 12, 12),
            L=(1024, 24, 16),
            H=(1280, 32, 16),
        )[variant]
        m = ViT(d_model, depth, n_heads, patch_size, img_size)

        if pretrained:
            ckpt = {
                ("Ti", 16): "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
                ("S", 32): "S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
                ("S", 16): "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
                ("B", 32): "B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
                ("B", 16): "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
                ("L", 16): "L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",
            }[(variant, patch_size)]
            base_url = "https://storage.googleapis.com/vit_models/augreg/"
            m.load_jax_weights(torch_hub_download(base_url + ckpt))
            if img_size != 224:
                m.resize_pe(img_size)

        return m

    # weights from https://github.com/google-research/vision_transformer
    @torch.no_grad()
    def load_jax_weights(self, path: str) -> ViT:
        jax_weights: Mapping[str, np.ndarray] = np.load(path)

        def get_w(key: str) -> Tensor:
            return torch.from_numpy(jax_weights[key])

        self.cls_token.copy_(get_w("cls"))
        self.patch_embed.weight.copy_(get_w("embedding/kernel").permute(3, 2, 0, 1))
        self.patch_embed.bias.copy_(get_w("embedding/bias"))
        self.pe.copy_(get_w("Transformer/posembed_input/pos_embedding"))

        for idx, layer in enumerate(self.layers):
            layer: ViTBlock
            prefix = f"Transformer/encoderblock_{idx}/"
            mha_prefix = prefix + "MultiHeadDotProductAttention_1/"

            layer.mha[0].weight.copy_(get_w(prefix + "LayerNorm_0/scale"))
            layer.mha[0].bias.copy_(get_w(prefix + "LayerNorm_0/bias"))
            layer.mha[1].q_proj.weight.copy_(get_w(mha_prefix + "query/kernel").flatten(1).T)
            layer.mha[1].k_proj.weight.copy_(get_w(mha_prefix + "key/kernel").flatten(1).T)
            layer.mha[1].v_proj.weight.copy_(get_w(mha_prefix + "value/kernel").flatten(1).T)
            layer.mha[1].q_proj.bias.copy_(get_w(mha_prefix + "query/bias").flatten())
            layer.mha[1].k_proj.bias.copy_(get_w(mha_prefix + "key/bias").flatten())
            layer.mha[1].v_proj.bias.copy_(get_w(mha_prefix + "value/bias").flatten())
            layer.mha[1].out_proj.weight.copy_(get_w(mha_prefix + "out/kernel").flatten(0, 1).T)
            layer.mha[1].out_proj.bias.copy_(get_w(mha_prefix + "out/bias"))

            layer.mlp[0].weight.copy_(get_w(prefix + "LayerNorm_2/scale"))
            layer.mlp[0].bias.copy_(get_w(prefix + "LayerNorm_2/bias"))
            layer.mlp[1].linear1.weight.copy_(get_w(prefix + "MlpBlock_3/Dense_0/kernel").T)
            layer.mlp[1].linear1.bias.copy_(get_w(prefix + "MlpBlock_3/Dense_0/bias"))
            layer.mlp[1].linear2.weight.copy_(get_w(prefix + "MlpBlock_3/Dense_1/kernel").T)
            layer.mlp[1].linear2.bias.copy_(get_w(prefix + "MlpBlock_3/Dense_1/bias"))

        self.norm.weight.copy_(get_w("Transformer/encoder_norm/scale"))
        self.norm.bias.copy_(get_w("Transformer/encoder_norm/bias"))
        return self
