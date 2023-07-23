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


class ViT(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        patch_size: int,
        img_size: int,
        mlp_dim: int | None = None,
        cls_token: bool = True,
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

        mlp_dim = mlp_dim or d_model * 4
        layer = nn.TransformerEncoderLayer(d_model, n_heads, mlp_dim, dropout, "gelu", norm_eps, True, True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, nn.LayerNorm(d_model, norm_eps))

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.patch_embed(imgs)
        out = out.flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        if self.cls_token is not None:
            out = torch.cat([self.cls_token.expand(out.shape[0], -1, -1), out], 1)
        out = out + self.pe
        out = self.encoder(out)
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
    @staticmethod
    def from_jax_weights(path: str) -> ViT:
        jax_weights: Mapping[str, np.ndarray] = np.load(path)

        n_layers = 1
        while True:
            if f"Transformer/encoderblock_{n_layers}/LayerNorm_0/bias" not in jax_weights:
                break
            n_layers += 1

        d_model = jax_weights["cls"].shape[-1]
        n_heads = jax_weights["Transformer/encoderblock_0/MultiHeadDotProductAttention_1/key/bias"].shape[0]
        patch_size = jax_weights["embedding/kernel"].shape[0]
        img_size = int((jax_weights["Transformer/posembed_input/pos_embedding"].shape[1] - 1) ** 0.5) * patch_size

        def _get(key: str) -> Tensor:
            return torch.from_numpy(jax_weights[key])

        torch_weights = dict()

        def _convert_layer_norm(jax_prefix: str, torch_prefix: str) -> None:
            torch_weights[f"{torch_prefix}.weight"] = _get(f"{jax_prefix}/scale")
            torch_weights[f"{torch_prefix}.bias"] = _get(f"{jax_prefix}/bias")

        def _convert_linear(jax_prefix: str, torch_prefix: str) -> None:
            torch_weights[f"{torch_prefix}.weight"] = _get(f"{jax_prefix}/kernel").T
            torch_weights[f"{torch_prefix}.bias"] = _get(f"{jax_prefix}/bias")

        def _convert_mha(jax_prefix: str, torch_prefix: str) -> None:
            w = torch.stack([_get(f"{jax_prefix}/{x}/kernel") for x in ["query", "key", "value"]], 1)
            torch_weights[f"{torch_prefix}.in_proj_weight"] = w.flatten(1).T

            b = torch.stack([_get(f"{jax_prefix}/{x}/bias") for x in ["query", "key", "value"]], 0)
            torch_weights[f"{torch_prefix}.in_proj_bias"] = b.flatten()

            torch_weights[f"{torch_prefix}.out_proj.weight"] = _get(f"{jax_prefix}/out/kernel").flatten(0, 1).T
            torch_weights[f"{torch_prefix}.out_proj.bias"] = _get(f"{jax_prefix}/out/bias")

        torch_weights["cls_token"] = _get("cls")
        torch_weights["patch_embed.weight"] = _get("embedding/kernel").permute(3, 2, 0, 1)
        torch_weights["patch_embed.bias"] = _get("embedding/bias")
        torch_weights["pe"] = _get("Transformer/posembed_input/pos_embedding")

        for idx in range(n_layers):
            jax_prefix = f"Transformer/encoderblock_{idx}"
            torch_prefix = f"encoder.layers.{idx}"

            _convert_layer_norm(f"{jax_prefix}/LayerNorm_0", f"{torch_prefix}.norm1")
            _convert_mha(f"{jax_prefix}/MultiHeadDotProductAttention_1", f"{torch_prefix}.self_attn")
            _convert_layer_norm(f"{jax_prefix}/LayerNorm_2", f"{torch_prefix}.norm2")
            _convert_linear(f"{jax_prefix}/MlpBlock_3/Dense_0", f"{torch_prefix}.linear1")
            _convert_linear(f"{jax_prefix}/MlpBlock_3/Dense_1", f"{torch_prefix}.linear2")

        _convert_layer_norm("Transformer/encoder_norm", "encoder.norm")

        model = ViT(n_layers, d_model, n_heads, patch_size, img_size)
        model.load_state_dict(torch_weights)
        return model
