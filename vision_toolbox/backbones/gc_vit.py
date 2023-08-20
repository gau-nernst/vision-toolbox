# https://arxiv.org/abs/2206.09959
# https://github.com/NVlabs/GCViT

import torch
from torch import Tensor, nn

from ..components import Permute
from .base import BaseBackbone
from .swin import WindowAttention


class SqueezeExcitation(nn.Module):
    def __init__(self, dim: int, expansion_ratio: float = 0.25, bias: bool = False) -> None:
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias),
            nn.Sigmoid(),
        )

    def foward(self, x: Tensor) -> Tensor:
        B, C = x.shape[:2]
        return x * self.gate(x.mean((2, 3))).view(B, C, 1, 1)


class Downsample(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, expansion_ratio: float = 0.25, bias: bool = False, norm_eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim, norm_eps)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim, bias=bias),  # dw-conv
            nn.GELU(),
            SqueezeExcitation(in_dim, expansion_ratio),
            nn.Conv2d(in_dim, in_dim, 1, bias=bias),  # pw-conv
        )
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 2, 1, bias=bias)
        self.norm2 = nn.LayerNorm(in_dim, norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x).permute(0, 3, 1, 2)
        x = x + self.conv1(x)
        x = self.norm2(self.conv2(x).permute(0, 2, 3, 1))
        return x


class GCViTStage(nn.Module):
    def __init__(
        self,
        d_model: int,
    ) -> None:
        super().__init__()


class GCViT(BaseBackbone):
    def __init__(
        self,
        d_model: int,
        expansion_ratio: float = 0.25,
        bias: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, d_model, 3, 2, 1),
            Permute(0, 2, 3, 1),
            Downsample(d_model, d_model, expansion_ratio, bias, norm_eps),
        )
