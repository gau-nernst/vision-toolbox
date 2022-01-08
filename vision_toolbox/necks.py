from typing import List
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from .components import ConvBnAct


def aggregate_sum(x: List[torch.Tensor]):
    out = x[0]
    for o in x[1:]:
        out += o
    return out

def aggregate_avg(x: List[torch.Tensor]):
    return aggregate_sum(x) / len(x)

def aggregate_max(x: List[torch.Tensor]):
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

def fuse_sum(x1, x2):
    return x1 + x2

def fuse_concat(x1, x2):
    return torch.cat((x1, x2), dim=1)

_fuse_functions = {
    "sum": fuse_sum,
    "concat": fuse_concat
}


class FPN(nn.Module):
    # https://arxiv.org/abs/1612.03144
    def __init__(self, in_channels, out_channels=256, fuse_fn="sum", lateral_block=None, output_block=None):
        assert fuse_fn in ("sum", "concat")
        super().__init__()
        self.fuse = _fuse_functions[fuse_fn]
        self.out_channels = out_channels
        self.stride = 2**len(in_channels)

        if lateral_block is None:
            lateral_block = partial(ConvBnAct, kernel_size=1, padding=0)
        if output_block is None:
            output_block = ConvBnAct

        lateral_convs = [lateral_block(in_c, out_channels) for in_c in in_channels]
        self.lateral_convs = nn.ModuleList(lateral_convs)
        
        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        output_convs = [output_block(in_c, out_channels) for _ in range(len(in_channels)-1)]
        self.output_convs = nn.ModuleList(output_convs)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.forward_features(x)[0]

    def forward_features(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = []
        for i, l_conv in enumerate(self.lateral_convs):
            laterals.append(l_conv(x[i]))

        out = laterals.pop()
        outputs = [out]
        
        for o_conv in self.output_convs:
            out = F.interpolate(out, scale_factor=2., mode="nearest")
            lat = laterals.pop()

            out = self.fuse(out, lat)
            out = o_conv(out)
            outputs.append(out)

        return outputs[::-1]

class SemanticFPN(nn.Module):
    # https://arxiv.org/abs/1901.02446
    def __init__(self, in_channels, fpn_channels=256, out_channels=128, fuse_fn="sum", agg_fn="sum", lateral_block=None, output_block=None):
        assert fuse_fn in ("sum", "concat")
        super().__init__()
        self.aggregate = _aggregate_functions[agg_fn]
        self.out_channels = out_channels * len(in_channels) if agg_fn == "concat" else out_channels
        self.stride = 2**len(in_channels)

        if lateral_block is None:
            lateral_block = partial(ConvBnAct, kernel_size=1, padding=0)
        if output_block is None:
            output_block = ConvBnAct

        self.fpn = FPN(in_channels, out_channels=fpn_channels, fuse_fn=fuse_fn, lateral_block=lateral_block, output_block=output_block)
   
        self.upsamples = nn.ModuleList()
        self.upsamples.append(output_block(fpn_channels, out_channels))
        for i in range(1, len(in_channels)):
            up = []
            for j in range(i):
                in_c = fpn_channels if j == 0 else out_channels
                up.append(output_block(in_c, out_channels))
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

class PAN(nn.Module):
    # https://arxiv.org/abs/1803.01534
    def __init__(self, in_channels, out_channels=256, lateral_block=None, output_block=None, downsample_block=None, fuse_fn="sum", agg_fn="max"):
        assert fuse_fn in ("sum", "concat")
        super().__init__()
        self.fuse = _fuse_functions[fuse_fn]
        self.aggregate = _aggregate_functions[agg_fn]
        self.out_channels = out_channels * len(in_channels) if agg_fn == "concat" else out_channels
        self.stride = 2**len(in_channels)

        if lateral_block is None:
            lateral_block = partial(ConvBnAct, kernel_size=1, padding=0)
        if output_block is None:
            output_block = ConvBnAct
        if downsample_block is None:
            downsample_block = partial(ConvBnAct, stride=2)

        self.fpn = FPN(in_channels, out_channels=out_channels, lateral_block=lateral_block, output_block=output_block, fuse_fn=fuse_fn)

        downsample_convs = [downsample_block(out_channels, out_channels) for _ in range(len(in_channels)-1)]
        self.downsample_convs = nn.ModuleList(downsample_convs)

        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        output_convs = [output_block(in_c, out_channels) for _ in range(len(in_channels)-1)]
        self.output_convs = nn.ModuleList(output_convs)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = self.forward_features(x)
        output_dim = outputs[0].shape[2:]
        for i in range(1, len(outputs)):
            outputs[i] = F.interpolate(outputs[i], output_dim, mode="nearest")
    
        return self.aggregate(outputs)

    def forward_features(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        fpn_outputs = self.fpn.forward_features(x)[::-1]        # top to bottom, so that .pop() is the bottom
        out = fpn_outputs.pop()
        outputs = [out]

        for d_conv, o_conv in zip(self.downsample_convs, self.output_convs):
            out = d_conv(out)
            fpn_out = fpn_outputs.pop()
            
            out = self.fuse(out, fpn_out)
            out = o_conv(out)
            outputs.append(out)
        
        return outputs

class BiFPNBlock(nn.Module):
    pass

class BiFPN(nn.Module):
    # https://arxiv.org/pdf/1911.09070.pdf
    pass
