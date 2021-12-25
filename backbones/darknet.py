# Papers
# https://arxiv.org/abs/1612.08242
# https://arxiv.org/abs/1804.02767
# https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf

import torch
from torch import nn

from .base import BaseBackbone
from .components import ConvBnAct


__all__ = [
    "Darknet", "CSPDarknet",
    "darknet19", "darknet53",
    "cspdarknet19", "cspdarknet53"
]

configs = {
    "darknet-19": {
        "stem_channels": 32,
        "num_blocks": (0, 1, 1, 2, 2),
        "num_channels": (64, 128, 256, 612, 1024)
    },
    "darknet-53": {
        "stem_channels": 32,
        "num_blocks": (1, 2, 8, 8, 4),
        "num_channels": (64, 128, 258, 512, 1024)
    }
}


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
        super().__init__()
        self.n = n
        self.conv = ConvBnAct(in_channels, out_channels, stride=2)

        if n > 0:
            half_channels = out_channels // 2
            self.conv1 = ConvBnAct(out_channels, half_channels, kernel_size=1, padding=0)
            self.conv2 = ConvBnAct(out_channels, half_channels, kernel_size=1, padding=0)
            self.blocks = nn.Sequential(*[DarknetBlock(half_channels, expansion=1) for _ in range(n)])
            
            self.out_conv = ConvBnAct(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        if self.n > 0:    
            out1 = self.conv1(out)      # using 2 convs is faster than using 1 conv then split
            out2 = self.conv2(out)

            out2 = self.blocks(out2)
            out = torch.cat((out1, out2), dim=1)
            out = self.out_conv(out)
        
        return out


class Darknet(BaseBackbone):
    def __init__(self, stem_channels, num_blocks, num_channels):
        super().__init__()
        self.out_channels = tuple(num_channels)
        self.stem = ConvBnAct(3, stem_channels)
        
        self.stages = nn.ModuleList()
        in_channels = stem_channels
        for n, c in zip(num_blocks, num_channels):
            new_stage = DarknetStage(n, in_channels, c)
            self.stages.append(new_stage)
            in_channels = c

    def forward_features(self, x):
        outputs = []
        out = self.stem(x)
        for s in self.stages:
            out = s(out)
            outputs.append(out)

        return outputs


class CSPDarknet(Darknet):
    def __init__(self, stem_channels, num_blocks, num_channels):
        nn.Module.__init__(self)
        self.out_channels = tuple(num_channels)
        self.stem = ConvBnAct(3, stem_channels)

        self.stages = nn.ModuleList()
        in_channels = stem_channels
        for n, c in zip(num_blocks, num_channels):
            new_stage = CSPDarknetStage(n, in_channels, c)
            self.stages.append(new_stage)
            in_channels = c


def darknet19(): return Darknet(**configs["darknet-19"])
def darknet53(): return Darknet(**configs["darknet-53"])
def cspdarknet19(): return CSPDarknet(**configs["darknet-19"])
def cspdarknet53(): return CSPDarknet(**configs["darknet-53"])
