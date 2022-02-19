from typing import List

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from .base import BaseBackbone
from ..components import ConvBnAct


configs = {
    'HRNet-W32': {
        'stem_channels': 64,
        'num_blocks': (1, 4, 3),
        'num_channels': 32
    },
    'HRNetV1-W48': {
        'stem_channes': 64,
        'num_blocks': (1, 4, 3),
        'num_channels': 48
    }
}

class ExchangeUnit(nn.Module):
    def __init__(self):
        pass


class ExchangeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_ss):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_ss):
            self.layers.append(nn.Sequential(*[BasicBlock(in_channels if i == 0 else out_channels, out_channels) for i in range(4)]))

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for i, layer in enumerate(self.layers):
            outputs.append(layer(x[i]))
        return outputs


class HRStage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        pass


class HRNetV1(BaseBackbone):
    def __init__(self, stem_channels, num_channels):
        super().__init__()
        self.stem = nn.Sequential(
            *[Bottleneck(3 if i == 0 else stem_channels, stem_channels) for i in range(4)],
            ConvBnAct(stem_channels, num_channels)
        )
        self.stages = nn.ModuleList()

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.stem(x)
        return self.stages(out)

class HRNetV2(BaseBackbone):
    def __init__(self, stem_channels):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnAct(3, stem_channels, stride=2),
            ConvBnAct(stem_channels, stem_channels, stride=2)
        )
        self.stages = nn.ModuleList()

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.stem(x)
        return self.stages(out)
