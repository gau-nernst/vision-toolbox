# Papers
# YOLOv2: https://arxiv.org/abs/1612.08242
# YOLOv3: https://arxiv.org/abs/1804.02767
# CSPNet: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf

from __future__ import annotations

from typing import Callable, NamedTuple

import torch
from torch import Tensor, nn

from ..components import ConvNormAct
from .base import BaseBackbone


_BASE_URL = "https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/"


class DarknetBlock(nn.Module):
    def __init__(self, in_channels: int, expansion: float = 0.5) -> None:
        super().__init__()
        mid_channels = int(in_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, mid_channels, 1)
        self.conv2 = ConvNormAct(mid_channels, in_channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class DarknetStage(nn.Sequential):
    def __init__(self, n: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # TODO: use self.append() instead
        self.conv = ConvNormAct(in_channels, out_channels, stride=2)
        self.blocks = nn.Sequential(*[DarknetBlock(out_channels) for _ in range(n)])


class CSPDarknetStage(nn.Module):
    def __init__(self, n: int, in_channels: int, out_channels: int) -> None:
        assert n > 0
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, stride=2)

        half_channels = out_channels // 2
        self.conv1 = ConvNormAct(out_channels, half_channels, 1)
        self.conv2 = ConvNormAct(out_channels, half_channels, 1)
        self.blocks = nn.Sequential(*[DarknetBlock(half_channels, expansion=1) for _ in range(n)])
        self.out_conv = ConvNormAct(out_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = torch.cat([self.conv1(out), self.blocks(self.conv2(out))], dim=1)
        out = self.out_conv(out)
        return out


class DarknetStageConfig(NamedTuple):
    n_blocks: int
    out_channels: int


class Darknet(BaseBackbone):
    def __init__(
        self,
        stem_channels: int,
        stage_configs: list[DarknetStageConfig | tuple[int, int]],
        stage_cls: Callable[..., nn.Module] = DarknetStage,
    ):
        assert len(stage_configs) > 0
        super().__init__()
        self.out_channels_list = tuple(cfg[1] for cfg in stage_configs)
        self.stride = 32

        self.stem = ConvNormAct(3, stem_channels)
        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for n_blocks, out_ch in stage_configs:
            stage = stage_cls(n_blocks, in_ch, out_ch) if n_blocks else ConvNormAct(in_ch, out_ch, 3, 2)
            self.stages.append(stage)
            in_ch = out_ch

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        outputs = [self.stem(x)]
        for s in self.stages:
            outputs.append(s(outputs[-1]))
        return outputs[1:]

    @staticmethod
    def from_config(variant: str, pretrained: bool = False) -> Darknet:
        n_blocks_list, stage_cls, ckpt = dict(
            darknet19=((0, 1, 1, 2, 2), DarknetStage, "darknet19-2cb641ca.pth"),  # YOLOv2
            darknet53=((1, 2, 8, 8, 4), DarknetStage, "darknet53-94427f5b.pth"),  # YOLOv3
            cspdarknet53=((1, 2, 8, 8, 4), CSPDarknetStage, "cspdarknet53-3bfa0423.pth"),  # CSPNet/YOLOv4
        )[variant]
        stage_configs = list(zip(n_blocks_list, (64, 128, 256, 512, 1024)))
        m = Darknet(32, stage_configs, stage_cls)
        if pretrained:
            m._load_state_dict_from_url(_BASE_URL + ckpt)
        return m


class DarknetYOLOv5(BaseBackbone):
    def __init__(self, stem_channels: int, stage_configs: list[DarknetStageConfig | tuple[int, int]]) -> None:
        super().__init__()
        self.out_channels_list = (stem_channels,) + tuple(cfg[1] for cfg in stage_configs)
        self.stride = 2 ** len(self.out_channels_list)

        self.stem = ConvNormAct(3, stem_channels, 6, 2)
        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for n_blocks, out_ch in stage_configs:
            self.stages.append(CSPDarknetStage(n_blocks, in_ch, out_ch))
            in_ch = out_ch

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        outputs = [self.stem(x)]
        for s in self.stages:
            outputs.append(s(outputs[-1]))
        return outputs

    @staticmethod
    def from_config(variant: str, pretrained: bool = False) -> DarknetYOLOv5:
        depth_scale, width_scale, ckpt = dict(
            n=(1 / 3, 1 / 4, "darknet_yolov5n-68f182f1.pth"),
            s=(1 / 3, 1 / 2, "darknet_yolov5s-175f7462.pth"),
            m=(2 / 3, 3 / 4, "darknet_yolov5m-9866aa40.pth"),
            l=(1 / 1, 1 / 1, "darknet_yolov5l-8e25d388.pth"),
            x=(4 / 3, 5 / 4, "darknet_yolov5x-0ed0c035.pth"),
        )[variant]
        stage_configs = [
            (int(d * depth_scale), int(w * width_scale)) for d, w in zip((3, 6, 9, 3), (128, 256, 512, 1024))
        ]
        m = DarknetYOLOv5(int(64 * width_scale), stage_configs)
        if pretrained:
            m._load_state_dict_from_url(_BASE_URL + ckpt)
        return m
