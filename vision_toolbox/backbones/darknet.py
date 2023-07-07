# Papers
# YOLOv2: https://arxiv.org/abs/1612.08242
# YOLOv3: https://arxiv.org/abs/1804.02767
# CSPNet: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf

from typing import Callable, Iterable

import torch
from torch import Tensor, nn

from ..components import ConvNormAct
from .base import BaseBackbone


__all__ = [
    "Darknet",
    "DarknetYolov5",
    "darknet19",
    "darknet53",
    "cspdarknet53",
    "darknet_yolov5n",
    "darknet_yolov5s",
    "darknet_yolov5m",
    "darknet_yolov5l",
    "darknet_yolov5x",
]


class DarknetBlock(nn.Module):
    def __init__(self, in_channels: int, expansion: float = 0.5):
        super().__init__()
        mid_channels = int(in_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, mid_channels, 1)
        self.conv2 = ConvNormAct(mid_channels, in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x + out


class DarknetStage(nn.Sequential):
    def __init__(self, n: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvNormAct(in_channels, out_channels, stride=2)
        self.blocks = nn.Sequential(*[DarknetBlock(out_channels) for _ in range(n)])


class CSPDarknetStage(nn.Module):
    def __init__(self, n: int, in_channels: int, out_channels: int):
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

        # using 2 convs is faster than using 1 conv then split
        out1 = self.conv1(out)
        out2 = self.conv2(out)
        out2 = self.blocks(out2)
        out = torch.cat((out1, out2), dim=1)
        out = self.out_conv(out)
        return out


class Darknet(BaseBackbone):
    def __init__(
        self,
        stem_channels: int,
        num_blocks_list: list[int],
        num_channels_list: list[int],
        stage_cls: Callable[..., nn.Module] = DarknetStage,
    ):
        super().__init__()
        self.out_channels_list = tuple(num_channels_list)
        self.stride = 32

        self.stem = ConvNormAct(3, stem_channels)
        self.stages = nn.ModuleList()
        in_c = stem_channels
        for n, c in zip(num_blocks_list, num_channels_list):
            self.stages.append(stage_cls(n, in_c, c) if n > 0 else ConvNormAct(in_c, c, stride=2))
            in_c = c

    def get_feature_maps(self, x):
        outputs = []
        out = self.stem(x)
        for s in self.stages:
            out = s(out)
            outputs.append(out)
        return outputs


class DarknetYolov5(BaseBackbone):
    def __init__(
        self,
        stem_channels: int,
        num_blocks_list: list[int],
        num_channels_list: list[int],
        stage_cls: Callable[..., nn.Module] = CSPDarknetStage,
    ):
        super().__init__()
        self.out_channels_list = (stem_channels,) + tuple(num_channels_list)
        self.stride = 32

        self.stem = ConvNormAct(3, stem_channels, 6, 2)
        self.stages = nn.ModuleList()
        in_c = stem_channels
        for n, c in zip(num_blocks_list, num_channels_list):
            self.stages.append(stage_cls(n, in_c, c))
            in_c = c

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        outputs = [self.stem(x)]
        for s in self.stages:
            outputs.append(s(outputs[-1]))
        return outputs


_base = dict(stem_channels=32, num_channels_list=(64, 128, 256, 512, 1024))
_darknet_yolov5_stem_channels = 64
_darknet_yolov5_num_blocks_list = (3, 6, 9, 3)
_darknet_yolov5_num_channels_list = (128, 256, 512, 1024)
configs = {
    # from YOLOv2
    "darknet-19": dict(
        **_base,
        num_blocks_list=(0, 1, 1, 2, 2),
        weights="https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet19-2cb641ca.pth",
    ),
    # from YOLOv3
    "darknet-53": dict(
        **_base,
        num_blocks_list=(1, 2, 8, 8, 4),
        weights="https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet53-94427f5b.pth",
    ),
    # from CSPNet/YOLOv4
    "cspdarknet-53": dict(
        **_base,
        num_blocks_list=(1, 2, 8, 8, 4),
        stage_cls=CSPDarknetStage,
        weights="https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/cspdarknet53-3bfa0423.pth",
    ),
    # from YOLOv5
    "darknet-yolov5n": dict(
        stem_channels=int(_darknet_yolov5_stem_channels / 4),
        num_blocks_list=tuple(int(x / 3) for x in _darknet_yolov5_num_blocks_list),
        num_channels_list=tuple(int(x / 4) for x in _darknet_yolov5_num_channels_list),
        weights="https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet_yolov5n-68f182f1.pth",
    ),
    "darknet-yolov5s": dict(
        stem_channels=int(_darknet_yolov5_stem_channels / 2),
        num_blocks_list=tuple(int(x / 3) for x in _darknet_yolov5_num_blocks_list),
        num_channels_list=tuple(int(x / 2) for x in _darknet_yolov5_num_channels_list),
        weights="https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet_yolov5s-175f7462.pth",
    ),
    "darknet-yolov5m": dict(
        stem_channels=int(_darknet_yolov5_stem_channels * 3 / 4),
        num_blocks_list=tuple(int(x * 2 / 3) for x in _darknet_yolov5_num_blocks_list),
        num_channels_list=tuple(int(x * 3 / 4) for x in _darknet_yolov5_num_channels_list),
        weights="https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet_yolov5m-9866aa40.pth",
    ),
    "darknet-yolov5l": dict(
        stem_channels=_darknet_yolov5_stem_channels,
        num_blocks_list=_darknet_yolov5_num_blocks_list,
        num_channels_list=_darknet_yolov5_num_channels_list,
        weights="https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet_yolov5l-8e25d388.pth",
    ),
    "darknet-yolov5x": dict(
        stem_channels=int(_darknet_yolov5_stem_channels * 5 / 4),
        num_blocks_list=tuple(int(x * 4 / 3) for x in _darknet_yolov5_num_blocks_list),
        num_channels_list=tuple(int(x * 5 / 4) for x in _darknet_yolov5_num_channels_list),
        weights="https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet_yolov5x-0ed0c035.pth",
    ),
}


def darknet19(pretrained=False, **kwargs):
    return Darknet.from_config(configs["darknet-19"], pretrained=pretrained, **kwargs)


def darknet53(pretrained=False, **kwargs):
    return Darknet.from_config(configs["darknet-53"], pretrained=pretrained, **kwargs)


def cspdarknet53(pretrained=False, **kwargs):
    return Darknet.from_config(configs["cspdarknet-53"], pretrained=pretrained, **kwargs)


def darknet_yolov5n(pretrained=False, **kwargs):
    return DarknetYolov5.from_config(configs["darknet-yolov5n"], pretrained=pretrained, **kwargs)


def darknet_yolov5s(pretrained=False, **kwargs):
    return DarknetYolov5.from_config(configs["darknet-yolov5s"], pretrained=pretrained, **kwargs)


def darknet_yolov5m(pretrained=False, **kwargs):
    return DarknetYolov5.from_config(configs["darknet-yolov5m"], pretrained=pretrained, **kwargs)


def darknet_yolov5l(pretrained=False, **kwargs):
    return DarknetYolov5.from_config(configs["darknet-yolov5l"], pretrained=pretrained, **kwargs)


def darknet_yolov5x(pretrained=False, **kwargs):
    return DarknetYolov5.from_config(configs["darknet-yolov5x"], pretrained=pretrained, **kwargs)
