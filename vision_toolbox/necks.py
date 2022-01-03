import torch
from torch import nn
import torch.nn.functional as F

from .components import ConvBnAct

def aggregate(x, agg_fn):
    assert agg_fn in ("sum", "avg", "max", "concat")    
    if agg_fn == "concat":
        return torch.concat(x, dim=1)
    
    out = x[0]
    if agg_fn == "max":
        for o in x[1:]:
            out = torch.maximum(out, o)
    else:    
        for o in x[1:]:
            out = out + o
        if agg_fn == "avg":
            out = out / len(x)

    return out

class FPN(nn.Module):
    # https://arxiv.org/abs/1612.03144
    def __init__(self, in_channels, out_channels=256, fuse_fn="sum", act_fn=nn.ReLU):
        assert fuse_fn in ("sum", "concat")
        super().__init__()
        self.fuse_fn = fuse_fn
        self.out_channels = out_channels

        lateral_convs = [ConvBnAct(in_c, out_channels, kernel_size=1, padding=0, act_fn=act_fn) for in_c in in_channels]
        self.lateral_convs = nn.ModuleList(lateral_convs)
        
        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        output_convs = [ConvBnAct(in_c, out_channels) for _ in range(len(in_channels)-1)]
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

class SemanticFPN(nn.Module):
    # https://arxiv.org/abs/1901.02446
    def __init__(self, in_channels, fpn_channels=256, out_channels=128, fuse_fn="sum", agg_fn="sum", act_fn=nn.ReLU):
        assert fuse_fn in ("sum", "concat")
        super().__init__()
        self.fuse_fn = fuse_fn
        self.agg_fn = agg_fn

        self.fpn = FPN(in_channels, out_channels=fpn_channels, fuse_fn=fuse_fn, act_fn=act_fn)
        self.upsamples = nn.ModuleList()
        self.upsamples.append(ConvBnAct(fpn_channels, out_channels))

        for i in range(1, len(in_channels)):
            up = []
            for j in range(i):
                in_c = fpn_channels if j == 0 else out_channels
                up.append(ConvBnAct(in_c, out_channels, act_fn=act_fn))
                up.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            
            up = nn.Sequential(*up)
            self.upsamples.append(up)

    def forward(self, x):
        outputs = self.forward_features(x)
        return aggregate(outputs, self.agg_fn)

    def forward_features(self, x):
        outputs = self.fpn.forward_features(x)
        outputs = [up(out) for up, out in zip(self.upsamples, outputs)]
        return outputs

class PAN(nn.Module):
    # https://arxiv.org/abs/1803.01534
    def __init__(self, in_channels, out_channels=256, fuse_fn="sum", agg_fn="max", act_fn=nn.ReLU):
        assert fuse_fn in ("sum", "concat")
        super().__init__()
        self.fuse_fn = fuse_fn
        self.agg_fn = agg_fn
        self.out_channels = out_channels

        self.fpn = FPN(in_channels, out_channels=out_channels, fuse_fn=fuse_fn, act_fn=act_fn)

        downsample_convs = [ConvBnAct(out_channels, out_channels, stride=2) for _ in range(len(in_channels)-1)]
        self.downsample_convs = nn.ModuleList(downsample_convs)

        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        output_convs = [ConvBnAct(in_c, out_channels) for _ in range(len(in_channels)-1)]
        self.output_convs = nn.ModuleList(output_convs)

    def forward(self, x):
        outputs = self.forward_features(x)
        output_dim = outputs[0].shape[2:]
        for i in range(1, len(outputs)):
            outputs[i] = F.interpolate(outputs[i], output_dim, mode="nearest")
    
        return aggregate(outputs, self.agg_fn)

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

class BiFPNBlock(nn.Module):
    pass

class BiFPN(nn.Module):
    # https://arxiv.org/pdf/1911.09070.pdf
    pass
