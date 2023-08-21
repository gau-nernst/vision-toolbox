# https://arxiv.org/abs/2207.10666
# https://github.com/microsoft/Cream

from __future__ import annotations

import itertools
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..components import ConvNormAct, StochasticDepth
from .base import BaseBackbone
from .swin import window_partition, window_unpartition
from .vit import MHA, ViTBlock


class MBConv(nn.Module):
    def __init__(self, dim: int, expansion_ratio: float = 4.0, stochastic_depth: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.conv = nn.Sequential(
            ConvNormAct(dim, hidden_dim, 1, norm="bn", act="gelu"),
            ConvNormAct(hidden_dim, hidden_dim, 3, groups=hidden_dim, norm="bn", act="gelu"),
            ConvNormAct(hidden_dim, dim, 1, norm="bn", act="none"),
            StochasticDepth(stochastic_depth),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv(x))


class Attention(MHA):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = True, dropout: float = 0.0, window_size: int = 7
    ) -> None:
        super().__init__(d_model, n_heads, bias, dropout)
        self.window_size = window_size
        indices, attn_offset_size = self.build_attention_bias(window_size)
        self.attention_biases = nn.Parameter(torch.zeros(n_heads, attn_offset_size))
        self.register_buffer("attention_bias_idxs", indices, persistent=False)
        self.attention_bias_idxs: Tensor

    @staticmethod
    def build_attention_bias(resolution: tuple[int, int]) -> tuple[Tensor, int]:
        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        attention_offsets: dict[tuple[int, int], int] = {}
        idxs: list[int] = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        N = len(points)
        indices = torch.LongTensor(idxs).view(N, N)
        attn_offset_size = len(attention_offsets)
        return indices, attn_offset_size

    def forward(self, x: Tensor) -> Tensor:
        x, nH, nW = window_partition(x, self.window_size)
        x = super().forward(x, self.attention_biases[:, self.attention_bias_idxs])
        x = window_unpartition(x, self.window_size, nH, nW)
        return x


class TinyViTBlock(ViTBlock):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = True,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float | None = None,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        # fmt: off
        super().__init__(
            d_model, n_heads, bias, mlp_ratio, dropout,
            layer_scale_init, stochastic_depth, norm_eps,
            partial(Attention, d_model, n_heads, bias, dropout, window_size),
        )
        # fmt: on
        self.local_conv = ConvNormAct(d_model, d_model, 3, groups=d_model, norm="bn", act="gelu")

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mha(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)
        x = x + self.mlp(x)
        return x


class TinyViT(BaseBackbone):
    def __init__(
        self,
        stem_dim: int,
        d_models: tuple[int, ...],
        depths: tuple[int, ...] = (2, 6, 2),
        window_sizes: tuple[int, ...] = (7, 14, 7),
        head_dim: int = 32,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        mbconv_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float | None = None,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, stem_dim // 2, 3, 2, norm="bn", act="gelu"),
            ConvNormAct(stem_dim // 2, stem_dim, 3, 2, norm="bn", act="none"),
            MBConv(stem_dim, mbconv_ratio),
            MBConv(stem_dim, mbconv_ratio),
        )

        in_dim = stem_dim
        self.stages = nn.Sequential()
        for d_model, depth, window_size in zip(d_models, depths, window_sizes):
            stage = nn.Sequential()

            downsample = nn.Sequential(
                ConvNormAct(in_dim, d_model, 1, norm="bn", act="gelu"),
                ConvNormAct(d_model, d_model, 3, 2, groups=d_model, norm="bn", act="gelu"),
                ConvNormAct(d_model, d_model, 1, norm="bn", act="none"),
            )
            stage.append(downsample)
            in_dim = d_model

            for _ in range(depth):
                # fmt: off
                block = TinyViTBlock(
                    d_model, d_model // head_dim, bias, window_size, mlp_ratio,
                    dropout, layer_scale_init, stochastic_depth, norm_eps
                )
                # fmt: on
                stage.append(block)

        self.norm = nn.LayerNorm(in_dim)

    def get_feature_maps(self, x: Tensor) -> Tensor:
        out = [self.stem(x)]
        for stage in self.stages:
            out.append(stage(out[-1]))
        return out

    def forward(self, x: Tensor) -> Tensor:
        x = self.get_feature_maps(x)[-1].mean(1)
        return self.norm(x)

    @staticmethod
    def from_config(variant: str, pretrained: bool = False) -> TinyViT:
        stem_dim, d_models = {
            "5m": (64, (128, 160, 320)),
            "11m": (64, (128, 256, 512)),
            "21m": (96, (192, 384, 576)),
        }[variant]
        m = TinyViT(stem_dim, d_models)

        if pretrained:
            name = f"tiny_vit_{variant}_22k_distill.pth"
            base_url = "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/"
            state_dict = torch.hub.load_state_dict_from_url(base_url + name)["model"]
            m.load_official_ckpt(state_dict)

        return m

    @torch.no_grad()
    def load_official_ckpt(self, state_dict: dict[str, Tensor]) -> None:
        raise NotImplementedError()


def _load_pretrained(model: TinyViT, url: str) -> TinyViT:
    model_state_dict = model.state_dict()
    state_dict = torch.hub.load_state_dict_from_url(url)

    # official checkpoint has "model" key
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # https://github.com/microsoft/Cream/blob/8dc38822b99fff8c262c585a32a4f09ac504d693/TinyViT/utils.py#L163
    # bicubic interpolate attention biases
    ab_keys = [k for k in state_dict.keys() if "attention_biases" in k]
    for k in ab_keys:
        n_heads1, L1 = state_dict[k].shape
        n_heads2, L2 = model_state_dict[k].shape

        if L1 != L2:
            S1 = int(L1**0.5)
            S2 = int(L2**0.5)
            attention_biases = state_dict[k].view(1, n_heads1, S1, S1)
            attention_biases = F.interpolate(attention_biases, size=(S2, S2), mode="bicubic")
            state_dict[k] = attention_biases.view(n_heads2, L2)

    if state_dict["head.weight"].shape[0] != model.head.out_features:
        state_dict["head.weight"] = torch.zeros_like(model.head.weight)
        state_dict["head.bias"] = torch.zeros_like(model.head.bias)

    model.load_state_dict(state_dict)
    return model
