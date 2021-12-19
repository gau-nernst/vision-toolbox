from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from .components import ConvBnAct

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256, fuse_fn="sum", bn_act=False, act_fn=nn.ReLU):
        assert fuse_fn in ("sum", "concat")
        super().__init__()
        self.fuse_fn = fuse_fn
        self.out_channels = out_channels
        conv_constructor = partial(ConvBnAct, act_fn=act_fn) if bn_act else nn.Conv2d

        lateral_convs = [conv_constructor(in_c, out_channels, kernel_size=1, padding=0) for in_c in in_channels]
        self.lateral_convs = nn.ModuleList(lateral_convs)
        
        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        output_convs = [conv_constructor(in_c, out_channels, kernel_size=3, padding=1) for _ in range(len(in_channels)-1)]
        self.output_convs = nn.ModuleList(output_convs)

    def forward(self, x):
        return self.forward_features(x)[0]

    def forward_features(self, x):
        laterals = [l_conv(feature_map) for l_conv, feature_map in zip(self.lateral_convs, x)]
        out = laterals.pop()
        outputs = [out]
        
        for o_conv in self.output_convs:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
            lat = laterals.pop()

            out = out + lat if self.fuse_fn == "sum" else torch.cat([out, lat])
            out = o_conv(out)
            outputs.append(out)

        return outputs[::-1]

class PAN(nn.Module):
    def __init__(self, in_channels, out_channels=256, fuse_fn="sum", agg_fn="max", bn_act=True, act_fn=nn.ReLU):
        assert fuse_fn in ("sum", "concat")
        assert agg_fn in ("max", "sum")
        super().__init__()
        self.fuse_fn = fuse_fn
        self.agg_fn = torch.maximum if agg_fn == "max" else torch.add
        self.out_channels = out_channels
        conv_constructor = partial(ConvBnAct, act_fn=act_fn) if bn_act else nn.Conv2d

        self.fpn = FPN(in_channels, out_channels=out_channels, fuse_fn=fuse_fn, bn_act=bn_act)

        downsample_convs = [conv_constructor(out_channels, out_channels, kernel_size=3, stride=2, padding=1) for _ in range(len(in_channels)-1)]
        self.downsample_convs = nn.ModuleList(downsample_convs)

        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        output_convs = [conv_constructor(in_c, out_channels, kernel_size=3, padding=1) for _ in range(len(in_channels)-1)]
        self.output_convs = nn.ModuleList(output_convs)

    def forward(self, x):
        outputs = self.forward_features(x)
        out = outputs[0]
        output_dim = out.shape[2:]

        for o in outputs[1:]:
            out = self.agg_fn(out, F.interpolate(o, size=output_dim, mode="nearest"))
        
        return out

    def forward_features(self, x):
        fpn_outputs = self.fpn.forward_features(x)[::-1]
        out = fpn_outputs.pop()
        outputs = [out]

        for d_conv, o_conv in zip(self.downsample_convs, self.output_convs):
            out = d_conv(out)
            fpn_out = fpn_outputs.pop()
            
            out = out + fpn_out if self.fuse_fn == "sum" else torch.cat([out, fpn_out])
            out = o_conv(out)
            outputs.append(out)
        
        return outputs
