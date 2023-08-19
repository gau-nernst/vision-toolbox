# https://arxiv.org/abs/2201.03545

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn

from ..components import Permute, StochasticDepth
from .base import BaseBackbone, _act, _norm


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        expansion_ratio: float = 4.0,
        bias: bool = True,
        layer_scale_init: float = 1e-6,
        stochastic_depth: float = 0.0,
        norm: _norm = partial(nn.LayerNorm, eps=1e-6),
        act: _act = nn.GELU,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * expansion_ratio)
        self.layers = nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(d_model, d_model, 7, padding=3, groups=d_model, bias=bias),
            Permute(0, 2, 3, 1),
            norm(d_model),
            nn.Linear(d_model, hidden_dim, bias=bias),
            act(),
            nn.Linear(hidden_dim, d_model, bias=bias),
        )
        self.layer_scale = nn.Parameter(torch.full((d_model,), layer_scale_init)) if layer_scale_init > 0 else None
        self.drop = StochasticDepth(stochastic_depth)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop(self.layers(x) * self.layer_scale)


class ConvNeXt(BaseBackbone):
    def __init__(
        self,
        d_model: int,
        depths: tuple[int, ...],
        expansion_ratio: float = 4.0,
        bias: bool = True,
        layer_scale_init: float = 1e-6,
        stochastic_depth: float = 0.0,
        norm: _norm = partial(nn.LayerNorm, eps=1e-6),
        act: _act = nn.GELU,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, d_model, 4, 4), Permute(0, 2, 3, 1), norm(d_model))

        stochastic_depth_rates = torch.linspace(0, stochastic_depth, sum(depths))
        self.stages = nn.Sequential()

        for stage_idx, depth in enumerate(depths):
            stage = nn.Sequential()
            if stage_idx > 0:
                # equivalent to PatchMerging in SwinTransformer
                downsample = nn.Sequential(
                    norm(d_model),
                    Permute(0, 3, 1, 2),
                    nn.Conv2d(d_model, d_model * 2, 2, 2),
                    Permute(0, 2, 3, 1),
                )
                d_model *= 2
            else:
                downsample = nn.Identity()
            stage.append(downsample)

            for block_idx in range(depth):
                rate = stochastic_depth_rates[sum(depths[:stage_idx]) + block_idx]
                block = ConvNeXtBlock(d_model, expansion_ratio, bias, layer_scale_init, rate, norm, act)
                stage.append(block)

            self.stages.append(stage)

        self.head_norm = norm(d_model)

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        out = [self.stem(x)]
        for stage in self.stages:
            out.append(stage(out[-1]))
        return out[-1:]

    def forward(self, x: Tensor) -> Tensor:
        return self.head_norm(self.get_feature_maps(x)[-1].mean((1, 2)))

    @staticmethod
    def from_config(variant: str, pretrained: bool = False) -> ConvNeXt:
        d_model, depths = dict(
            T=(96, (3, 3, 9, 3)),
            S=(96, (3, 3, 27, 3)),
            B=(128, (3, 3, 27, 3)),
            L=(192, (3, 3, 27, 3)),
            XL=(256, (3, 3, 27, 3)),
        )[variant]
        m = ConvNeXt(d_model, depths)

        if pretrained:
            # TODO: also add torchvision checkpoints?
            ckpt = dict(
                T="convnext_tiny_22k_224.pth",
                S="convnext_small_22k_224.pth",
                B="convnext_base_22k_224.pth",
                L="convnext_large_22k_224.pth",
                XL="convnext_xlarge_22k_224.pth",
            )[variant]
            base_url = "https://dl.fbaipublicfiles.com/convnext/"
            state_dict = torch.hub.load_state_dict_from_url(base_url + ckpt)["model"]
            m.load_official_ckpt(state_dict)

        return m

    @torch.no_grad()
    def load_official_ckpt(self, state_dict: dict[str, Tensor]) -> None:
        def copy_(m: nn.Conv2d | nn.Linear | nn.LayerNorm, prefix: str):
            m.weight.copy_(state_dict.pop(prefix + ".weight"))
            m.bias.copy_(state_dict.pop(prefix + ".bias"))

        copy_(self.stem[0], "downsample_layers.0.0")  # Conv2d
        copy_(self.stem[2], "downsample_layers.0.1")  # LayerNorm

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx > 0:
                copy_(stage[0][0], f"downsample_layers.{stage_idx}.0")  # LayerNorm
                copy_(stage[0][2], f"downsample_layers.{stage_idx}.1")  # Conv2d

            for block_idx in range(1, len(stage)):
                block: ConvNeXtBlock = stage[block_idx]
                prefix = f"stages.{stage_idx}.{block_idx - 1}."

                copy_(block.layers[1], prefix + "dwconv")
                copy_(block.layers[3], prefix + "norm")
                copy_(block.layers[4], prefix + "pwconv1")
                copy_(block.layers[6], prefix + "pwconv2")
                block.layer_scale.copy_(state_dict.pop(prefix + "gamma"))

        copy_(self.head_norm, "norm")
        assert len(state_dict) == 2
