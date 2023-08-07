# https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_mixer.py

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from ..utils import torch_hub_download


class MLP(nn.Sequential):
    def __init__(self, d_model: int, mlp_dim: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, d_model)


class MixerBlock(nn.Module):
    def __init__(self, n_tokens: int, d_model: int, tokens_mlp_dim: int, channels_mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mixing = MLP(n_tokens, tokens_mlp_dim)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mixing = MLP(d_model, channels_mlp_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x -> (B, n_tokens, d_model)
        x = x + self.token_mixing(self.norm1(x).transpose(-1, -2)).transpose(-1, -2)
        x = x + self.channel_mixing(self.norm2(x))
        return x


class MLPMixer(nn.Module):
    def __init__(
        self, n_layers: int, d_model: int, patch_size: int, img_size: int, tokens_mlp_dim: int, channels_mlp_dim: int
    ) -> None:
        assert img_size % patch_size == 0
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        n_tokens = (img_size // patch_size) ** 2
        self.layers = nn.Sequential(
            *[MixerBlock(n_tokens, d_model, tokens_mlp_dim, channels_mlp_dim) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x).flatten(-2).transpose(-1, -2)
        x = self.layers(x)
        x = self.norm(x)
        x = x.mean(-2)
        return x

    @staticmethod
    def from_config(variant: str, patch_size: int, img_size: int, pretrained: bool = False) -> MLPMixer:
        n_layers, d_model, tokens_mlp_dim, channels_mlp_dim = dict(
            S=(8, 512, 256, 2048), B=(12, 768, 384, 3072), L=(24, 1024, 512, 4096), H=(32, 1280, 640, 5120)
        )[variant]
        m = MLPMixer(n_layers, d_model, patch_size, img_size, tokens_mlp_dim, channels_mlp_dim)
        if pretrained:
            ckpt = {("B", 16): "Mixer-B_16.npz", ("L", 16): "Mixer-L_16.npz"}[(variant, patch_size)]
            base_url = "https://storage.googleapis.com/mixer_models/imagenet21k/"
            m.load_jax_weights(torch_hub_download(base_url + ckpt))
        return m

    @torch.no_grad()
    def load_jax_weights(self, path: str) -> MLPMixer:
        jax_weights: Mapping[str, np.ndarray] = np.load(path)

        def get_w(key: str) -> Tensor:
            return torch.from_numpy(jax_weights[key])

        self.patch_embed.weight.copy_(get_w("stem/kernel").permute(3, 2, 0, 1))
        self.patch_embed.bias.copy_(get_w("stem/bias"))
        for i, layer in enumerate(self.layers):
            layer: MixerBlock
            prefix = f"MixerBlock_{i}/"
            layer.norm1.weight.copy_(get_w(prefix + "LayerNorm_0/scale"))
            layer.norm1.bias.copy_(get_w(prefix + "LayerNorm_0/bias"))
            layer.token_mixing.linear1.weight.copy_(get_w(prefix + "token_mixing/Dense_0/kernel").T)
            layer.token_mixing.linear1.bias.copy_(get_w(prefix + "token_mixing/Dense_0/bias"))
            layer.token_mixing.linear2.weight.copy_(get_w(prefix + "token_mixing/Dense_1/kernel").T)
            layer.token_mixing.linear2.bias.copy_(get_w(prefix + "token_mixing/Dense_1/bias"))
            layer.norm2.weight.copy_(get_w(prefix + "LayerNorm_1/scale"))
            layer.norm2.bias.copy_(get_w(prefix + "LayerNorm_1/bias"))
            layer.channel_mixing.linear1.weight.copy_(get_w(prefix + "channel_mixing/Dense_0/kernel").T)
            layer.channel_mixing.linear1.bias.copy_(get_w(prefix + "channel_mixing/Dense_0/bias"))
            layer.channel_mixing.linear2.weight.copy_(get_w(prefix + "channel_mixing/Dense_1/kernel").T)
            layer.channel_mixing.linear2.bias.copy_(get_w(prefix + "channel_mixing/Dense_1/bias"))
        self.norm.weight.copy_(get_w("pre_head_layer_norm/scale"))
        self.norm.bias.copy_(get_w("pre_head_layer_norm/bias"))
        return self
