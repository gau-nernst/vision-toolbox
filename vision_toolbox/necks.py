from abc import ABCMeta
from typing import List
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from .components import ConvBnAct


# support torchscript
def aggregate_sum(x: List[torch.Tensor]) -> torch.Tensor:
    out = x[0]
    for o in x[1:]:
        out = out + o
    return out

def aggregate_avg(x: List[torch.Tensor]) -> torch.Tensor:
    return aggregate_sum(x) / len(x)

def aggregate_max(x: List[torch.Tensor]) -> torch.Tensor:
    out = x[0]
    for o in x[1:]:
        out = torch.maximum(out, o)
    return out

_aggregate_functions = {
    "concat": partial(torch.concat, dim=1),
    "sum": aggregate_sum,
    "avg": aggregate_avg,
    "max": aggregate_max
}


class BaseNeck(nn.Module, metaclass=ABCMeta):
    def forward_features(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        pass

    def get_out_channels(self) -> int:
        return self.out_channels


class FPN(BaseNeck):
    # https://arxiv.org/abs/1612.03144
    def __init__(self, in_channels, out_channels=256, fuse_fn="sum", block=ConvBnAct):
        super().__init__()
        self.fuse = _aggregate_functions[fuse_fn]
        self.out_channels = out_channels
        self.stride = 2**(len(in_channels)-1)

        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_c, out_channels, kernel_size=1) for in_c in in_channels])
        
        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        self.output_convs = nn.ModuleList([block(in_c, out_channels) for _ in range(len(in_channels)-1)])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.forward_features(x)[0]

    def forward_features(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [l_conv(x[i]) for i, l_conv in enumerate(self.lateral_convs)]
        outputs = [laterals.pop()]
        
        for o_conv in self.output_convs:
            out = F.interpolate(outputs[-1], scale_factor=2., mode="nearest")
            out = self.fuse([out, laterals.pop()])
            outputs.append(o_conv(out))

        return outputs[::-1]


class SemanticFPN(BaseNeck):
    # https://arxiv.org/abs/1901.02446
    def __init__(self, in_channels, fpn_channels=256, out_channels=128, fuse_fn="sum", agg_fn="sum", block=ConvBnAct):
        super().__init__()
        self.aggregate = _aggregate_functions[agg_fn]
        self.out_channels = out_channels * len(in_channels) if agg_fn == "concat" else out_channels
        self.stride = 2**len(in_channels)

        self.fpn = FPN(in_channels, out_channels=fpn_channels, fuse_fn=fuse_fn, block=block)
   
        self.upsamples = nn.ModuleList()
        self.upsamples.append(block(fpn_channels, out_channels))        # no upsample for bottom-most feature map
        for i in range(1, len(in_channels)):
            up = []
            for j in range(i):
                in_c = fpn_channels if j == 0 else out_channels
                up.append(block(in_c, out_channels))
                up.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            
            up = nn.Sequential(*up)
            self.upsamples.append(up)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = self.forward_features(x)
        return self.aggregate(outputs)

    def forward_features(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = self.fpn.forward_features(x)
        outputs = [up(outputs[i]) for i, up in enumerate(self.upsamples)]
        return outputs

class PAN(BaseNeck):
    # https://arxiv.org/abs/1803.01534
    def __init__(self, in_channels, out_channels=256, fuse_fn="sum", agg_fn="max", block=ConvBnAct):
        super().__init__()
        self.fuse = _aggregate_functions[fuse_fn]
        self.aggregate = _aggregate_functions[agg_fn]
        self.out_channels = out_channels * len(in_channels) if agg_fn == "concat" else out_channels
        self.stride = 2**len(in_channels)

        self.fpn = FPN(in_channels, out_channels=out_channels, fuse_fn=fuse_fn, block=block)
        self.downsample_convs = nn.ModuleList([block(out_channels, out_channels, stride=2) for _ in range(len(in_channels)-1)])

        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        self.output_convs = nn.ModuleList([block(in_c, out_channels) for _ in range(len(in_channels)-1)])

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = self.forward_features(x)
        for i in range(1, len(outputs)):
            outputs[i] = F.interpolate(outputs[i], scale_factor=2**i, mode="nearest")

        return self.aggregate(outputs)

    def forward_features(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        fpn_outputs = self.fpn.forward_features(x)[::-1]        # top to bottom, so that .pop() is the bottom
        out = fpn_outputs.pop()
        outputs = [out]

        for d_conv, o_conv in zip(self.downsample_convs, self.output_convs):
            out = d_conv(out)
            out = self.fuse([out, fpn_outputs.pop()])
            out = o_conv(out)
            outputs.append(out)
        
        return outputs


class BiFPNLayer(nn.Module):
    def __init__(self, num_levels, channels, fuse_fn="sum", block=ConvBnAct):
        super().__init__()
        self.fuse = _aggregate_functions[fuse_fn]

        if fuse_fn == "sum":
            self.inter_convs = nn.ModuleList([block(channels, channels) for _ in range(num_levels-1)])
            self.output_convs = nn.ModuleList([block(channels, channels) for _ in range(num_levels-1)])

        else:
            self.inter_convs = nn.ModuleList([block(2*channels, channels) for _ in range(num_levels-1)])    # fuse 2 feature maps
            self.output_convs = nn.ModuleList([block(3*channels, channels) for _ in range(num_levels-2)])   # fuse 3 feature maps
            self.output_convs.append(block(2*channels, channels))                   # only fuse 2 feature maps for top-most level

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # top-down
        inters = [x[-1]]                                                        # P7in
        for i, conv in enumerate(self.inter_convs):
            out = F.interpolate(inters[-1], scale_factor=2., mode="nearest")    # resize(P7td)
            out = self.fuse([x[-2-i], out])                                     # P6in + resize(P7td)
            inters.append(conv(out))                                            # P6td = conv(P6in + resize(P7td))
            
        # bottom-up
        inters = inters[::-1]           # feature maps from bottom to top, same order as input x
        outputs = [inters[0]]
        for i, conv in enumerate(self.output_convs):
            out = F.interpolate(outputs[-1], scale_factor=0.5, mode="nearest")  # resize(P3td)
            if i < len(self.output_convs) - 1:
                out = self.fuse([x[i+1], inters[i+1], out])                     # P4in + P4td + resize(P3td)
            else:
                out = self.fuse([inters[i+1], out])                             # P7in + resize(P6td)
            outputs.append(conv(out))                                           # P4out = conv(P4in + P4td + resize(P3td))

        return outputs


class BiFPN(BaseNeck):
    # https://arxiv.org/pdf/1911.09070.pdf
    # https://github.com/google/automl/blob/master/efficientdet/efficientdet_arch.py
    def __init__(self, in_channels, out_channels=64, num_layers=1, fuse_fn="sum", block=ConvBnAct):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(in_c, out_channels, kernel_size=1) for in_c in in_channels])
        self.layers = nn.ModuleList([BiFPNLayer(len(in_channels), out_channels, fuse_fn=fuse_fn, block=block) for _ in range(num_layers)])

        self.out_channels = out_channels

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.forward_features(x)[0]

    def forward_features(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = [l_conv(x[i]) for i, l_conv in enumerate(self.laterals)]
        for bifpn_layer in self.layers:
            outputs = bifpn_layer(outputs)        
        return outputs
