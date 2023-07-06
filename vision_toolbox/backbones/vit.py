# https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

import re

import numpy as np
import torch
from torch import Tensor, nn


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
        img_size: int | tuple[int, int],
        mlp_dim: int | None = None,
        cls_token: bool = True,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model)) if cls_token else None

        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        pe_size = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        if cls_token:
            pe_size += 1
        self.pe = nn.Parameter(torch.empty(1, pe_size, d_model))

        mlp_dim = mlp_dim or d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            mlp_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, nn.LayerNorm(d_model))

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: init self.patch_embed and self.encoder
        if self.cls_token is not None:
            nn.init.zeros_(self.cls_token)
        nn.init.normal_(self.pe, 0, 0.02)

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.patch_embed(imgs)
        out = out.flatten(-2).transpose(-1, -2)  # (N, C, H, W) -> (N, H*W, C)
        if self.cls_token is not None:
            out = torch.cat([self.cls_token.expand(out.shape[0], -1, -1), out], 1)
        out = out + self.pe
        out = self.encoder(out)
        out = out[:, 0] if self.cls_token is not None else out.mean(1)
        return out

    @staticmethod
    def from_config(variant: str, patch_size: int, img_size: int | tuple[int, int]) -> "ViT":
        return ViT(**configs[variant], patch_size=patch_size, img_size=img_size)


def convert_jax_weights(jax_weights: dict[str, np.ndarray]) -> dict[str, Tensor]:
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

        torch_weights[f"{torch_prefix}.out_proj.weight"] = _get(f"{jax_prefix}/out/kernel").flatten(0, 1)
        torch_weights[f"{torch_prefix}.out_proj.bias"] = _get(f"{jax_prefix}/out/bias")

    n_layers = 0
    for key in jax_weights:
        match = re.search("Transformer/encoderblock_(\d+)/", key)
        if match is not None:
            n_layers = max(n_layers, int(match.group(1)) + 1)

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

    return torch_weights
