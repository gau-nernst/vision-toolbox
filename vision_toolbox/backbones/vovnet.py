# Papers:
# VoVNetV1: https://arxiv.org/abs/1904.09730
# VoVNetV2: https://arxiv.org/abs/1911.06667 (CenterMask)

import torch
from torch import nn

from .base import BaseBackbone
from ..components import ConvBnAct, ESEBlock


__all__ = [
    "VoVNet", 'vovnet27_slim', 'vovnet39', 'vovnet57',
    "vovnet19_slim_ese", "vovnet19_ese", "vovnet39_ese", "vovnet57_ese", "vovnet99_ese"
]

# https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py
_base = {
    'stem_channels': (64, 64, 128),
    'stage_channels': (128, 160, 192, 224),
    'out_channels': (256, 512, 768, 1024),
    'num_layers': (5, 5, 5, 5)
}
_slim = {
    'stem_channels': (64, 64, 128),
    'stage_channels': (64, 80, 96, 112),
    'out_channels': (128, 256, 384, 512),
    'num_layers': (3, 3, 3, 3)
}
configs = {
    # VoVNetV1
    'vovnet-27-slim': {
        **_slim,
        "num_blocks": (1, 1, 1, 1),
        'ese': False,
        'weights': 'https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet27_slim-7ac333a5.pth'
    },
    'vovnet-39': {
        **_base,
        "num_blocks": (1, 1, 2, 2),
        "ese": False
    },
    'vovnet-57': {
        **_base,
        "num_blocks": (1, 1, 4, 3),
        "ese": False
    },
    # VoVNetV2
    'vovnet-19-slim-ese': {
        **_slim,
        'num_blocks': (1, 1, 1, 1),
        'weights': 'https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet19_slim_ese-446e2ae9.pth'
    },
    'vovnet-19-ese': {
        **_base,
        'num_blocks': (1, 1, 1, 1),
        'num_layers': (3, 3, 3, 3),
        'weights': 'https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet19_ese-4410fc5f.pth'
    },
    'vovnet-39-ese': {
        **_base,
        'num_blocks': (1, 1, 2, 2),
        'weights': 'https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet39_ese-b73bdbe9.pth'
    },
    'vovnet-57-ese': {
        **_base,
        'num_blocks': (1, 1, 4, 3),
        'weights': 'https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet57_ese-630a88d1.pth'
    },
    'vovnet-99-ese': {
        **_base,
        'num_blocks': (1, 3, 9, 3),
        'weights': 'https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/vovnet99_ese-56fd52f5.pth'
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

        out = torch.cat(outputs, dim=1)
        out = self.out_conv(out)
        
        if self.ese is not None:
            out = self.ese(out)
        if self.residual:
            out = out + x

        return out


class VoVNet(BaseBackbone):
    def __init__(self, stem_channels, num_blocks, stage_channels, num_layers, out_channels, residual=True, ese=True, num_returns=4):
        super().__init__()
        self.out_channels = tuple(out_channels)[-num_returns:]
        self.stride = 32
        self.num_returns = num_returns
        
        self.stem = nn.Sequential()
        in_c = 3
        for i, c in enumerate(stem_channels):
            self.stem.add_module(str(i), ConvBnAct(in_c, c, stride=2 if i == 0 else 1))
            in_c = c
        
        self.stages = nn.ModuleList()
        for n, stage_c, n_l, out_c in zip(num_blocks, stage_channels, num_layers, out_channels):
            stage = nn.Sequential()
            stage.add_module('max_pool', nn.MaxPool2d(3, 2, padding=1))
            for i in range(n):
                stage.add_module(f'module_{i}', OSABlock(in_c, stage_c, n_l, out_c, residual=residual, ese=ese))
                in_c = out_c
            self.stages.append(stage)

    def forward_features(self, x):
        outputs = []
        out = self.stem(x)
        outputs.append(out)
        for s in self.stages:
            out = s(out)
            outputs.append(out)
        
        return outputs[-self.num_returns:]


def vovnet27_slim(pretrained=False, **kwargs): return VoVNet.from_config(configs["vovnet-27-slim"], pretrained=pretrained, **kwargs)
def vovnet39(pretrained=False, **kwargs): return VoVNet.from_config(configs["vovnet-39"], pretrained=pretrained, **kwargs)
def vovnet57(pretrained=False, **kwargs): return VoVNet.from_config(configs["vovnet-57"], pretrained=pretrained, **kwargs)
def vovnet19_slim_ese(pretrained=False, **kwargs): return VoVNet.from_config(configs["vovnet-19-slim-ese"], pretrained=pretrained, **kwargs)
def vovnet19_ese(pretrained=False, **kwargs): return VoVNet.from_config(configs["vovnet-19-ese"], pretrained=pretrained, **kwargs)
def vovnet39_ese(pretrained=False, **kwargs): return VoVNet.from_config(configs["vovnet-39-ese"], pretrained=pretrained, **kwargs)
def vovnet57_ese(pretrained=False, **kwargs): return VoVNet.from_config(configs["vovnet-57-ese"], pretrained=pretrained, **kwargs)
def vovnet99_ese(pretrained=False, **kwargs): return VoVNet.from_config(configs["vovnet-99-ese"], pretrained=pretrained, **kwargs)
