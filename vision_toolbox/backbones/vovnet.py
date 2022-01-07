# Papers:
# https://arxiv.org/abs/1904.09730
# https://arxiv.org/abs/1911.06667

import torch
from torch import nn

from .base import BaseBackbone
from ..components import ConvBnAct, ESEBlock


__all__ = [
    "VoVNet", "vovnet19_slim", "vovnet19", "vovnet39", "vovnet57", "vovnet99"
]

# https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py
_stage_channels = (128, 160, 192, 224)
_out_channels = (256, 512, 768, 1024)
configs = {
    "vovnet-19-slim": {
        "stem_channels": (64, 64, 128),
        "num_blocks": (1, 1, 1, 1),
        "stage_channels": (64, 80, 96, 112),
        "num_layers": (3, 3, 3, 3),
        "out_channels": (112, 256, 384, 512),
        "weights": "https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet19_slim-231d7449.pth"
    },
    "vovnet-19": {
        "stem_channels": (64, 64, 128),
        "num_blocks": (1, 1, 1, 1),
        "stage_channels": _stage_channels,
        "num_layers": (3, 3, 3, 3),
        "out_channels": _out_channels,
        "weights": "https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet19-4410fc5f.pth"
    },
    "vovnet-39": {
        "stem_channels": (64, 64, 128),
        "num_blocks": (1, 1, 2, 2),
        "stage_channels": _stage_channels,
        "num_layers": (5, 5, 5, 5),
        "out_channels": _out_channels,
        "weights": "https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet39-b73bdbe9.pth"
    },
    "vovnet-57": {
        "stem_channels": (64, 64, 128),
        "num_blocks": (1, 1, 4, 3),
        "stage_channels": _stage_channels,
        "num_layers": (5, 5, 5, 5),
        "out_channels": _out_channels,
        "weights": "https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet57-630a88d1.pth"
    },
    "vovnet-99": {
        "stem_channels": (64, 64, 128),
        "num_blocks": (1, 3, 9, 3),
        "stage_channels": _stage_channels,
        "num_layers": (5, 5, 5, 5),
        "out_channels": _out_channels
    }
}


class OSABlock(nn.Module):
    def __init__(self, in_channels, mid_channels, num_layers, out_channels, residual=True, ese=True):
        super().__init__()
        self.convs = nn.ModuleList([ConvBnAct(in_channels if i == 0 else mid_channels, mid_channels) for i in range(num_layers)])
        concat_channels = in_channels + mid_channels * num_layers
        self.out_conv = ConvBnAct(concat_channels, out_channels, kernel_size=1, padding=0)

        self.ese = ESEBlock(out_channels) if ese else None
        self.residual = residual and (in_channels == out_channels)

    def forward(self, x):
        outputs = []
        out = x
        outputs.append(out)
        for conv_layer in self.convs:
            out = conv_layer(out)
            outputs.append(out)

        out = torch.concat(outputs, dim=1)
        out = self.out_conv(out)
        
        if self.ese is not None:
            out = self.ese(out)
        if self.residual:
            out = out + x

        return out


class OSAStage(nn.Sequential):
    def __init__(self, num_blocks, in_channels, stage_channels, num_layers, out_channels, residual=True, ese=True):
        super().__init__()
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        for i in range(num_blocks):
            in_c = in_channels if i == 0 else out_channels
            self.add_module(f"module_{i}", OSABlock(in_c, stage_channels, num_layers, out_channels, residual=residual, ese=ese))


class VoVNet(BaseBackbone):
    # to make VoVNetV1, pass residual=False and ese=False
    def __init__(self, stem_channels, num_blocks, stage_channels, num_layers, out_channels, residual=True, ese=True):
        super().__init__()
        self.out_channels = tuple(out_channels)
        self.stem = nn.Sequential()
        in_c = 3
        for i, c in enumerate(stem_channels):
            self.stem.add_module(str(i), ConvBnAct(in_c, c, stride=2 if i == 0 else 1))
            in_c = c
        
        self.stages = nn.ModuleList()
        for n, stage_c, n_l, out_c in zip(num_blocks, stage_channels, num_layers, out_channels):
            self.stages.append(OSAStage(n, in_c, stage_c, n_l, out_c, residual=residual, ese=ese))
            in_c = out_c

    def forward_features(self, x):
        outputs = []
        out = self.stem(x)
        outputs.append(out)
        for s in self.stages:
            out = s(out)
            outputs.append(out)

        return outputs


def vovnet19_slim(pretrained=False): return VoVNet.from_config(configs["vovnet-19-slim"], pretrained=pretrained)
def vovnet19(pretrained=False): return VoVNet.from_config(configs["vovnet-19"], pretrained=pretrained)
def vovnet39(pretrained=False): return VoVNet.from_config(configs["vovnet-39"], pretrained=pretrained)
def vovnet57(pretrained=False): return VoVNet.from_config(configs["vovnet-57"], pretrained=pretrained)
def vovnet99(pretrained=False): return VoVNet.from_config(configs["vovnet-99"], pretrained=pretrained)
