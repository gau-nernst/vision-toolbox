# Papers:
# VoVNetV1: https://arxiv.org/abs/1904.09730
# VoVNetV2: https://arxiv.org/abs/1911.06667 (CenterMask)
# https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ..components import ConvNormAct
from .base import BaseBackbone


_BASE_URL = "https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/"


class ESEBlock(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Conv2d(num_channels, num_channels, 1)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(self.linear(self.pool(x)))


class OSABlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        num_layers: int,
        out_channels: int,
        ese: bool = True,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [ConvNormAct(in_channels if i == 0 else mid_channels, mid_channels) for i in range(num_layers)]
        )
        concat_channels = in_channels + mid_channels * num_layers
        self.out_conv = ConvNormAct(concat_channels, out_channels, 1)

        self.ese = ESEBlock(out_channels) if ese else None
        self.residual = in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        outputs = [x]
        for conv in self.convs:
            outputs.append(conv(outputs[-1]))

        out = torch.cat(outputs, dim=1)
        out = self.out_conv(out)

        if self.ese is not None:
            out = self.ese(out)
        if self.residual:
            out = out + x

        return out


class VoVNetStageConfig(NamedTuple):
    n_blocks: int
    mid_channels: int
    n_layers: int
    out_channels: int


class VoVNet(BaseBackbone):
    def __init__(
        self,
        stem_channels: int,
        stage_configs: list[VoVNetStageConfig | tuple[int, int, int, int]],
        ese: bool = True,
    ) -> None:
        super().__init__()
        self.out_channels_list = (stem_channels,) + tuple(cfg[3] for cfg in stage_configs)
        self.stride = 2 ** len(self.out_channels_list)

        self.stem = nn.Sequential(
            ConvNormAct(3, stem_channels // 2, 3, 2),
            ConvNormAct(stem_channels // 2, stem_channels // 2),
            ConvNormAct(stem_channels // 2, stem_channels),
        )

        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for n_blocks, mid_ch, n_layers, out_ch in stage_configs:
            stage = nn.Sequential()
            stage.add_module("max_pool", nn.MaxPool2d(3, 2, 1))
            for i in range(n_blocks):
                stage.add_module(f"module_{i}", OSABlock(in_ch, mid_ch, n_layers, out_ch, ese))
                in_ch = out_ch
            self.stages.append(stage)

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        outputs = [self.stem(x)]
        for s in self.stages:
            outputs.append(s(outputs[-1]))
        return outputs

    @staticmethod
    def from_config(variant: int, slim: bool = False, ese: bool = False, pretrained: bool = False) -> VoVNet:
        stem_channels = 128
        mid_channels_list = (64, 80, 96, 112) if slim else (128, 160, 192, 224)
        out_channels_list = (128, 256, 384, 512) if slim else (256, 512, 768, 1024)
        n_blocks_list, n_layers_list = {
            19: ((1, 1, 1, 1), (3, 3, 3, 3)),
            27: ((1, 1, 1, 1), (5, 5, 5, 5)),
            39: ((1, 1, 2, 2), (5, 5, 5, 5)),
            57: ((1, 1, 4, 3), (5, 5, 5, 5)),
            99: ((1, 3, 9, 3), (5, 5, 5, 5)),
        }[variant]
        stage_configs = list(zip(n_blocks_list, mid_channels_list, n_layers_list, out_channels_list))
        m = VoVNet(stem_channels, stage_configs, ese)

        if pretrained:
            ckpt = {
                # VoVNetV1
                (27, True, False): "vovnet27_slim-dd43306a.pth",
                (39, False, False): "vovnet39-4c79d629.pth",
                (57, False, False): "vovnet57-ecb9cc34.pth",
                # VoVNetV2
                (19, True, True): "vovnet19_slim_ese-f8075640.pth",
                (19, False, True): "vovnet19_ese-a077657e.pth",
                (39, False, True): "vovnet39_ese-9ce81b0d.pth",
                (57, False, True): "vovnet57_ese-ae1a7f89.pth",
                (99, False, True): "vovnet99_ese-713f3062.pth",
            }[(variant, slim, ese)]
            m._load_state_dict_from_url(_BASE_URL + ckpt)

        return m
