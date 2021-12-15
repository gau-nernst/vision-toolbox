import torch
from torch import nn

from .base import BaseBackbone
from .components import ConvBnAct

__all__ = [
    "VoVNet",
    "vovnet19_slim",
    "vovnet39",
    "vovnet57",
    "vovnet99"
]

# https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py
configs = {
    "vovnet-19-slim": {
        "stem_channels": (64, 64, 128),
        "num_blocks": (1, 1, 1, 1),
        "stage_channels": (64, 80, 96, 112),
        "num_layers": (3, 3, 3, 3),
        "out_channels": (112, 256, 384, 512)
    },
    "vovnet-39": {
        "stem_channels": (64, 64, 128),
        "num_blocks": (1, 1, 2, 2),
        "stage_channels": (128, 160, 192, 224),
        "num_layers": (5, 5, 5, 5),
        "out_channels": (256, 512, 784, 1024)
    },
    "vovnet-57": {
        "stem_channels": (64, 64, 128),
        "num_blocks": (1, 1, 4, 3),
        "stage_channels": (128, 160, 192, 224),
        "num_layers": (5, 5, 5, 5),
        "out_channels": (256, 512, 784, 1024)
    },
    "vovnet-99": {
        "stem_channels": (64, 64, 128),
        "num_blocks": (1, 3, 9, 3),
        "stage_channels": (128, 160, 192, 224),
        "num_layers": (5, 5, 5, 5),
        "out_channels": (256, 512, 784, 1024)
    }
}

class OSABlock(nn.Module):
    def __init__(self, in_channels, mid_channels, num_layers, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([ConvBnAct(in_channels if i == 0 else mid_channels, mid_channels) for i in range(num_layers)])
        concat_channels = in_channels + mid_channels * num_layers
        self.out_conv = ConvBnAct(concat_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        outputs = []
        out = x
        outputs.append(out)
        for conv_layer in self.convs:
            out = conv_layer(out)
            outputs.append(out)

        out = torch.concat(outputs, dim=1)
        out = self.out_conv(out)
        return out

class OSAStage(nn.Sequential):
    def __init__(self, n, in_channels, stage_channels, num_layers, out_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.module_0 = OSABlock(in_channels, stage_channels, num_layers, out_channels)
        for i in range(1, n):
            self.add_module(f"module_{i}", OSABlock(out_channels, stage_channels, num_layers, out_channels))

class VoVNet(BaseBackbone):
    def __init__(self, stem_channels, num_blocks, stage_channels, num_layers, out_channels):
        super().__init__()
        self.out_channels = tuple(out_channels)
        self.stem = nn.Sequential()
        in_channels = 3
        for i, c in enumerate(stem_channels):
            self.stem.add_module(str(i), ConvBnAct(in_channels, c, stride=2 if i == 0 else 1))
            in_channels = c
        
        self.stages = nn.ModuleList()
        for n, stage_c, n_l, out_c in zip(num_blocks, stage_channels, num_layers, out_channels):
            self.stages.append(OSAStage(n, in_channels, stage_c, n_l, out_c))
            in_channels = out_c

    def forward_features(self, x):
        outputs = []
        out = self.stem(x)
        outputs.append(out)
        for s in self.stages:
            out = s(out)
            outputs.append(out)

        return outputs

    def get_out_channels(self):
        return self.out_channels

def vovnet19_slim():
    return VoVNet(**configs["vovnet-19-slim"])

def vovnet39():
    return VoVNet(**configs["vovnet-39"])

def vovnet57():
    return VoVNet(**configs["vovnet-57"])

def vovnet99():
    return VoVNet(**configs["vovnet-99"])
