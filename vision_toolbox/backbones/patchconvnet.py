# https://arxiv.org/abs/2112.13692
# https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import SqueezeExcitation

from .base import BaseBackbone


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


class PatchConvBlockLN(nn.Module):
    def __init__(self, embed_dim: int, drop_path: float = 0.3, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        # LayerNorm version. Primary format is (N, H, W, C)
        # follow this approach https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
        self.layers = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            Permute(0, 3, 1, 2),  # (N, H, W, C) -> (N, C, H, W)
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),  # dw-conv
            nn.GELU(),
            SqueezeExcitation(embed_dim, embed_dim // 4),
            Permute(0, 2, 3, 1),  # (N, C, H, W) -> (N, H, W, C)
            nn.Linear(embed_dim, embed_dim),
        )
        self.layer_scale = nn.Parameter(torch.full((embed_dim,), layer_scale_init))
        self.drop_path = StochasticDepth(drop_path, "row") if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.layers(x) * self.layer_scale)


class PatchConvBlockBN(nn.Module):
    def __init__(self, embed_dim: int, drop_path: float = 0.3, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        # BatchNorm version. Primary format is (N, C, H, W)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
            nn.GELU(),
            SqueezeExcitation(embed_dim, embed_dim // 4),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )
        self.layer_scale = nn.Parameter(torch.full((embed_dim, 1, 1), layer_scale_init))
        self.drop_path = StochasticDepth(drop_path, "row") if drop_path > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.layers(x) * self.layer_scale)


class AttentionPooling(nn.Module):
    def __init__(
        self, embed_dim: int, mlp_ratio: int = 3, drop_path: float = 0.3, layer_scale_init: float = 1e-6
    ) -> Tensor:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(embed_dim))

        self.norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.layer_scale_1 = nn.Parameter(torch.full((embed_dim,), layer_scale_init))
        self.drop_path1 = StochasticDepth(drop_path, "row") if drop_path > 0 else nn.Identity()

        self.norm_2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, embed_dim))
        self.layer_scale_2 = nn.Parameter(torch.full((embed_dim,), layer_scale_init))
        self.drop_path2 = StochasticDepth(drop_path, "row") if drop_path > 0 else nn.Identity()

        self.norm_3 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # (N, HW, C)
        cls_token = self.cls_token.expand(x.shape[0], 1, -1)
        out = torch.cat((cls_token, x), dim=1)

        # attention pooling. q = cls_token. k = v = (cls_token, x)
        out = self.norm_1(out)
        out = self.attn(out[:, :1], out, out, need_weights=False)[0]
        cls_token = cls_token + self.drop_path1(out * self.layer_scale_1)  # residual + layer scale + dropout

        # mlp
        out = self.mlp(self.norm_2(cls_token))
        cls_token = cls_token + self.drop_path2(out * self.layer_scale_2)

        out = self.norm_3(cls_token).squeeze(1)  # (N, 1, C) -> (N, C)
        return out


class PatchConvNet(BaseBackbone):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        mlp_ratio: int = 3,
        drop_path: float = 0.3,
        layer_scale_init: float = 1e-6,
        norm_type: str = "bn",
    ) -> None:
        assert norm_type in ("bn", "ln")
        super().__init__()
        self.norm_type = norm_type
        self.out_channels_list = (embed_dim,)
        self.stride = 16

        # stem has no bias and no last activation layer
        # https://github.com/facebookresearch/deit/issues/151
        conv3x3_s2 = partial(nn.Conv2d, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem = nn.Sequential(
            conv3x3_s2(3, embed_dim // 8),
            nn.GELU(),
            conv3x3_s2(embed_dim // 8, embed_dim // 4),
            nn.GELU(),
            conv3x3_s2(embed_dim // 4, embed_dim // 2),
            nn.GELU(),
            conv3x3_s2(embed_dim // 2, embed_dim),
        )

        blk = PatchConvBlockLN if norm_type == "ln" else PatchConvBlockBN
        self.trunk = nn.Sequential(
            Permute(0, 2, 3, 1) if norm_type == "ln" else nn.Identity(),
            *[blk(embed_dim, drop_path, layer_scale_init) for _ in range(depth)],
            Permute(0, 2, 3, 1) if norm_type == "bn" else nn.Identity(),
        )
        self.pool = AttentionPooling(embed_dim, mlp_ratio, drop_path, layer_scale_init)

        # weight initialization
        nn.init.trunc_normal_(self.pool.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pool.attn.in_proj_weight, std=0.02)
        nn.init.trunc_normal_(self.pool.attn.out_proj.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        out = self.stem(x)
        out = self.trunk(out)
        out = out.flatten(1, 2)
        out = self.pool(out)
        return [out]

    @staticmethod
    def from_config(variant: str, depth: int, pretrained: bool = False) -> PatchConvNet:
        embed_dim = dict(S=384, B=768, L=1024)[variant]
        m = PatchConvNet(embed_dim, depth)
        if pretrained:
            raise ValueError
        return m
