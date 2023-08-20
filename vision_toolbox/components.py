import math
from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn
from torchvision.ops import DeformConv2d


__all__ = ["ConvNormAct", "SeparableConv2d", "DeformableConv2d", "SPPBlock"]


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        norm: str = "bn",
        act: str = "relu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=math.ceil((kernel_size - stride) / 2),
            dilation=dilation,
            groups=groups,
            bias=norm == "none",
        )
        self.norm = dict(none=nn.Identity, bn=nn.BatchNorm2d)[norm](out_channels)
        self.act = dict(
            none=nn.Identity,
            relu=partial(nn.ReLU, True),
            leaky_relu=partial(nn.LeakyReLU, 0.2, True),
            swish=partial(nn.SiLU, True),
            silu=partial(nn.SiLU, True),
            gelu=nn.GELU,
        )[act]()
        if act in ("relu", "leaky_relu"):
            nn.init.kaiming_normal_(self.conv.weight, 0.2, "fan_out", act)


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        act_fn: Callable[[], nn.Module] = partial(nn.ReLU6, inplace=True),
    ):
        super().__init__()
        # don't include bn and act?
        self.dw = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            act_fn=act_fn,
        )
        self.pw = ConvNormAct(in_channels, out_channels, kernel_size=1, padding=0, act_fn=act_fn)


# https://arxiv.org/abs/1703.06211
# https://arxiv.org/abs/1811.11168
class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_fn: Callable[[int, int], nn.Module] = nn.Conv2d,
        mask_act: Callable[[], nn.Module] = nn.Sigmoid,
        v2: bool = True,
        mask_init_bias: float = 0,
    ):
        # https://github.com/msracver/Deformable-ConvNets/blob/master/DCNv2_op/example_symbol.py    x2 after sigmoid
        # https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/modulated_deform_conv.py          don't x2 after sigmoid
        super().__init__()
        num_locations = kernel_size**2 if isinstance(kernel_size, int) else kernel_size[0] * kernel_size[1]
        # include groups here also?
        self.conv_offset = conv_fn(
            in_channels,
            2 * num_locations,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.conv_mask = None
        if v2:
            self.conv_mask = nn.Sequential(
                conv_fn(
                    in_channels,
                    num_locations,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                mask_act(inplace=True),
            )
        # what is offset groups (torchvision) / deform groups (mmcv) ?
        self.conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # how to initalize?

    def forward(self, x: Tensor) -> Tensor:
        offset = self.conv_offset(x)
        mask = self.conv_mask(x) if self.conv_mask is not None else None
        return self.conv(x, offset, mask)


# add source. YOLO?
class SPPBlock(nn.Module):
    def __init__(self, kernel_size: int = 5, repeats: int = 3, pool: str = "max"):
        # https://github.com/ultralytics/yolov5/blob/master/models/common.py    see SPPF
        # equivalent to [5, 9, 13] max pooling
        super().__init__()
        self.pool = dict(avg=nn.AvgPool2d, max=nn.MaxPool2d)[pool](kernel_size, 1, (kernel_size - 1) // 2)
        self.repeats = repeats

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for _ in range(self.repeats):
            x = self.pool(x)
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class Permute(nn.Module):
    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)


# https://arxiv.org/pdf/1603.09382.pdf
class StochasticDepth(nn.Module):
    def __init__(self, p: float) -> None:
        assert 0.0 <= p <= 1.0
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        keep_p = 1.0 - self.p
        return x * x.new_empty(shape).bernoulli_(keep_p).div_(keep_p)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}"
