from functools import partial
from typing import Callable, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


__all__ = [
    'ConvBnAct', 'SeparableConv2d', 'DeformableConv2d',
    'ESEBlock', 'SPPBlock'
]


# torchvision.ops.misc.ConvNormActivation initializes weights differently
class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=3,
        stride: int=1,
        padding: int=1,
        dilation: int=1,
        groups: int=1,
        bias: Optional[bool]=None,
        norm_layer: Optional[Callable[..., nn.Module]]=nn.BatchNorm2d,
        act_fn: Optional[Callable[..., nn.Module]]=partial(nn.ReLU, inplace=True)
    ):
        super().__init__()
        if bias is None:
            bias = norm_layer is None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if norm_layer is not None:
            self.bn = norm_layer(out_channels)
        if act_fn is not None:
            self.act = act_fn()
            nn.init.kaiming_normal_(self.conv.weight, a=getattr(self.act, "negative_slop", 0), mode="fan_out")


class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, act_fn=nn.ReLU6):
        super().__init__()
        # don't include bn and act?
        self.dw = ConvBnAct(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, act_fn=act_fn)
        self.pw = ConvBnAct(in_channels, out_channels, kernel_size=1, padding=0, act_fn=act_fn)


# https://arxiv.org/abs/1911.06667
class ESEBlock(nn.Module):
    def __init__(self, num_channels, gate_fn=nn.Hardsigmoid):
        # author's code uses Hardsigmoid, although it is not mentioned in the code
        # https://github.com/youngwanLEE/vovnet-detectron2/blob/master/vovnet/vovnet.py
        super().__init__()
        self.linear = nn.Conv2d(num_channels, num_channels, 1)      # use conv so don't need to flatten output
        self.gate = gate_fn()

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1,1))
        out = self.linear(out)
        return x * self.gate(out)


# https://arxiv.org/abs/1703.06211
# https://arxiv.org/abs/1811.11168
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, conv_fn=nn.Conv2d, mask_act=nn.Sigmoid, v2=True, mask_init_bias=0):
        # https://github.com/msracver/Deformable-ConvNets/blob/master/DCNv2_op/example_symbol.py    x2 after sigmoid
        # https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/modulated_deform_conv.py          don't x2 after sigmoid
        super().__init__()
        num_locations = kernel_size**2 if isinstance(kernel_size, int) else kernel_size[0] * kernel_size[1]
        # include groups here also?
        self.conv_offset = conv_fn(in_channels, 2*num_locations, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv_mask = None
        if v2:
            self.conv_mask = nn.Sequential(
                conv_fn(in_channels, num_locations, kernel_size, stride=stride, padding=padding, dilation=dilation),
                mask_act(inplace=True)
            )
        # what is offset groups (torchvision) / deform groups (mmcv) ?
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # how to initalize?

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = self.conv_mask(x) if self.conv_mask is not None else None
        return self.conv(x, offset, mask)


# add source. YOLO?
class SPPBlock(nn.Module):
    def __init__(self, kernel_size=5, repeats=3, pool_fn="max"):
        # https://github.com/ultralytics/yolov5/blob/master/models/common.py    see SPPF
        # equivalent to [5, 9, 13] max pooling
        # any convolution here?
        assert pool_fn in ("max", "avg")
        super().__init__()
        pool_fn = nn.MaxPool2d if pool_fn == "max" else nn.AvgPool2d
        padding = (kernel_size - 1) // 2
        self.pool = pool_fn(kernel_size, stride=1, padding=padding)
        self.repeats = repeats
    
    def forward(self, x):
        outputs = []
        out = x
        for _ in range(self.repeats):
            out = self.pool(out)
            outputs.append(out)
        return torch.cat(outputs, dim=1)
