# Papers
# YOLOv2: https://arxiv.org/abs/1612.08242
# YOLOv3: https://arxiv.org/abs/1804.02767
# CSPNet: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf

import torch
from torch import nn

from .base import BaseBackbone
from ..components import ConvBnAct


__all__ = [
    "Darknet", "darknet19", "darknet53", "cspdarknet53",
    'darknet_yolov5n', 'darknet_yolov5s', 'darknet_yolov5m', 'darknet_yolov5l', 'darknet_yolov5x'
]


class DarknetBlock(nn.Module):
    def __init__(self, in_channels, expansion=0.5):
        super().__init__()
        mid_channels = int(in_channels * expansion)
        self.conv1 = ConvBnAct(in_channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBnAct(mid_channels, in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return x + out


class DarknetStage(nn.Module):
    def __init__(self, n, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBnAct(in_channels, out_channels, stride=2)
        self.blocks = nn.Sequential(*[DarknetBlock(out_channels) for _ in range(n)])
    
    def forward(self, x):
        out = self.conv(x)
        out = self.blocks(out)
        return out


class CSPDarknetStage(nn.Module):
    def __init__(self, n, in_channels, out_channels):
        assert n > 0
        super().__init__()
        self.conv = ConvBnAct(in_channels, out_channels, stride=2)

        half_channels = out_channels // 2
        self.conv1 = ConvBnAct(out_channels, half_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBnAct(out_channels, half_channels, kernel_size=1, padding=0)
        self.blocks = nn.Sequential(*[DarknetBlock(half_channels, expansion=1) for _ in range(n)])
        self.out_conv = ConvBnAct(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        
        # using 2 convs is faster than using 1 conv then split
        out1 = self.conv1(out)
        out2 = self.conv2(out)

        out2 = self.blocks(out2)
        out = torch.cat((out1, out2), dim=1)
        out = self.out_conv(out)
        return out


class Darknet(BaseBackbone):
    def __init__(self, stem_channels, num_blocks, num_channels, block_fn=DarknetStage, num_returns=4):
        assert num_returns <= 5
        super().__init__() 
        self.out_channels = tuple(num_channels)[-num_returns:]
        self.stride = 32
        self.num_returns = num_returns

        self.stem = ConvBnAct(3, stem_channels)
        self.stages = nn.ModuleList()
        in_c = stem_channels
        for n, c in zip(num_blocks, num_channels):
            self.stages.append(block_fn(n, in_c, c) if n > 0 else ConvBnAct(in_c, c, stride=2))
            in_c = c
        
    def forward_features(self, x):
        outputs = []
        out = self.stem(x)
        for s in self.stages:
            out = s(out)
            outputs.append(out)
        
        return outputs[-self.num_returns:]


class DarknetYolov5(BaseBackbone):
    def __init__(self, stem_channels, num_blocks, num_channels, block_fn=CSPDarknetStage, num_returns=4):
        assert num_returns <= 5
        super().__init__()
        self.out_channel = tuple(num_channels)[-num_returns:]
        self.stride = 32
        self.num_returns = num_returns

        self.stem = ConvBnAct(3, stem_channels, kernel_size=6, stride=2, padding=2)
        self.stages = nn.ModuleList()
        in_c = stem_channels
        for n, c in zip(num_blocks, num_channels):
            self.stages.append(block_fn(n, in_c, c))
            in_c = c

    def forward_features(self, x):
        outputs = [self.stem(x)]
        for s in self.stages:
            outputs.append(s(outputs[-1]))
        return outputs[-self.num_returns:]


_darknet_yolov5_stem_channels = 64
_darknet_yolov5_num_blocks = (3, 6, 9, 3)
_darknet_yolov5_num_channels = (128, 256, 512, 1024)
configs = {
    # from YOLOv2
    "darknet-19": {
        "stem_channels": 32,
        "num_blocks": (0, 1, 1, 2, 2),
        "num_channels": (64, 128, 256, 512, 1024),
        "block_fn": DarknetStage,
        "weights": "https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet19-da4bd7c9.pth"
    },
    # from YOLOv3
    "darknet-53": {
        "stem_channels": 32,
        "num_blocks": (1, 2, 8, 8, 4),
        "num_channels": (64, 128, 258, 512, 1024),
        "block_fn": DarknetStage,
        "weights": "https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/darknet53-2315d8ae.pth"
    },
    # from YOLOv4
    "cspdarknet-53": {
        "stem_channels": 32,
        "num_blocks": (1, 2, 8, 8, 4),
        "num_channels": (64, 128, 258, 512, 1024),
        "block_fn": CSPDarknetStage,
        "weights": "https://github.com/gau-nernst/vision-toolbox/releases/download/v0.0.1/cspdarknet53-cfb1eaf2.pth"
    },
    # from YOLOv5
    'darknet-yolov5n': {
        'stem_channels': int(_darknet_yolov5_stem_channels/4),
        'num_blocks': tuple(int(x/3) for x in _darknet_yolov5_num_blocks),
        'num_channels': tuple(int(x/4) for x in _darknet_yolov5_num_channels)
    },
    'darknet-yolov5s': {
        'stem_channels': int(_darknet_yolov5_stem_channels/2),
        'num_blocks': tuple(int(x/3) for x in _darknet_yolov5_num_blocks),
        'num_channels': tuple(int(x/2) for x in _darknet_yolov5_num_channels)
    },
    'darknet-yolov5m': {
        'stem_channels': int(_darknet_yolov5_stem_channels*3/4),
        'num_blocks': tuple(int(x*2/3) for x in _darknet_yolov5_num_blocks),
        'num_channels': tuple(int(x*3/4) for x in _darknet_yolov5_num_channels)
    },
    'darknet-yolov5l': {
        'stem_channels': _darknet_yolov5_stem_channels,
        'num_blocks': _darknet_yolov5_num_blocks,
        'num_channels': _darknet_yolov5_num_channels
    },
    'darknet-yolov5x': {
        'stem_channels': int(_darknet_yolov5_stem_channels*5/4),
        'num_blocks': tuple(int(x*4/3) for x in _darknet_yolov5_num_blocks),
        'num_channels': tuple(int(x*5/4) for x in _darknet_yolov5_num_channels)
    }
}

def darknet19(pretrained=False, **kwargs): return Darknet.from_config(configs["darknet-19"], pretrained=pretrained, **kwargs)
def darknet53(pretrained=False, **kwargs): return Darknet.from_config(configs["darknet-53"], pretrained=pretrained, **kwargs)
def cspdarknet53(pretrained=False, **kwargs): return Darknet.from_config(configs["cspdarknet-53"], pretrained=pretrained, **kwargs)
def darknet_yolov5n(pretrained=False, **kwargs): return DarknetYolov5.from_config(configs['darknet-yolov5n'], pretrained=pretrained, **kwargs)
def darknet_yolov5s(pretrained=False, **kwargs): return DarknetYolov5.from_config(configs['darknet-yolov5s'], pretrained=pretrained, **kwargs)
def darknet_yolov5m(pretrained=False, **kwargs): return DarknetYolov5.from_config(configs['darknet-yolov5m'], pretrained=pretrained, **kwargs)
def darknet_yolov5l(pretrained=False, **kwargs): return DarknetYolov5.from_config(configs['darknet-yolov5l'], pretrained=pretrained, **kwargs)
def darknet_yolov5x(pretrained=False, **kwargs): return DarknetYolov5.from_config(configs['darknet-yolov5x'], pretrained=pretrained, **kwargs)
