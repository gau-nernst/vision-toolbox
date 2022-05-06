from abc import ABCMeta, abstractmethod
from typing import Callable, Iterable, List

import torch
from torch import nn
import torch.nn.functional as F

from .components import ConvBnAct, SeparableConv2d


__all__ = ["BaseNeck", "FPN", "PAN", "BiFPN"]


# support torchscript
def aggregate_concat(x: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(x, dim=1)  # torchscript does not support partial


def aggregate_sum(x: List[torch.Tensor]) -> torch.Tensor:
    out = x[0]
    for o in x[1:]:
        out = out + o  # += will do inplace addition
    return out


def aggregate_avg(x: List[torch.Tensor]) -> torch.Tensor:
    return aggregate_sum(x) / len(x)


def aggregate_max(x: List[torch.Tensor]) -> torch.Tensor:
    out = x[0]
    for o in x[1:]:
        out = torch.maximum(out, o)
    return out


_aggregate_functions = {
    "concat": aggregate_concat,
    "sum": aggregate_sum,
    "avg": aggregate_avg,
    "max": aggregate_max,
}


class BaseNeck(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def fuse_feature_maps(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        pass


# https://arxiv.org/abs/1612.03144
class FPN(BaseNeck):
    def __init__(
        self,
        in_channels: Iterable[int],
        out_channels: int = 256,
        fuse_fn: str = "sum",
        block: Callable[[int, int], nn.Module] = ConvBnAct,
        interpolation_mode: str = "nearest",
        top_down: bool = True,
    ):
        super().__init__()
        self.fuse = _aggregate_functions[fuse_fn]
        self.out_channels = out_channels
        self.top_down = top_down

        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_c, out_channels, kernel_size=1)
                if in_c != out_channels
                else nn.Identity()
                for in_c in in_channels
            ]
        )
        self.upsample = nn.Upsample(
            scale_factor=2.0 if top_down else 0.5, mode=interpolation_mode
        )
        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        self.output_convs = nn.ModuleList(
            [block(in_c, out_channels) for _ in range(len(in_channels) - 1)]
        )

    def _fuse_top_down(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for i in range(len(x) - 2, -1, -1):  # 2, 1, 0
            x[i] = self.fuse(x[i], self.upsample(x[i + 1]))
            x[i] = self.output_convs[i](x[i])
        return x

    def _fuse_bottom_up(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for i in range(len(x) - 1):  # 0, 1, 2
            x[i + 1] = self.fuse(x[i + 1], self.upsample(x[i]))
            x[i + 1] = self.output_convs[i](x[i + 1])
        return x

    # input feature maps are ordered from bottom to top
    def fuse_feature_maps(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(x) == len(self.lateral_convs)
        outputs = [l_conv(x[i]) for i, l_conv in enumerate(self.lateral_convs)]
        if self.top_down:
            return self._fuse_top_down(outputs)
        return self._fuse_bottom_up(outputs)


# https://arxiv.org/abs/1803.01534
class PAN(BaseNeck):
    def __init__(
        self,
        in_channels: Iterable[int],
        out_channels: int = 256,
        fuse_fn: str = "sum",
        block: Callable[[int, int], nn.Module] = ConvBnAct,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.top_down = FPN(
            in_channels,
            out_channels,
            fuse_fn=fuse_fn,
            block=block,
            interpolation_mode=interpolation_mode,
        )
        self.bottom_up = FPN(
            [out_channels] * len(in_channels),
            out_channels,
            fuse_fn=fuse_fn,
            block=block,
            interpolation_mode=interpolation_mode,
        )

    def fuse_feature_maps(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = self.top_down(x)
        outputs = self.bottom_up(outputs)
        return outputs


# https://arxiv.org/pdf/1911.09070.pdf
# https://github.com/google/automl/blob/master/efficientdet/efficientdet_arch.py
class BiFPN(BaseNeck):
    def __init__(
        self,
        in_channels: Iterable[int],
        out_channels: int = 64,
        num_layers: int = 1,
        block: Callable[[int, int], nn.Module] = SeparableConv2d,
        interpolation_mode="nearest",
        eps: float = 1e-4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.laterals = nn.ModuleList(
            [nn.Conv2d(in_c, out_channels, kernel_size=1) for in_c in in_channels]
        )
        self.layers = nn.Sequential(
            *[
                BiFPNLayer(
                    len(in_channels),
                    out_channels,
                    block=block,
                    interpolation_mode=interpolation_mode,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

    def fuse_feature_maps(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = [l_conv(x[i]) for i, l_conv in enumerate(self.laterals)]
        outputs = self.layers(outputs)
        return outputs


class BiFPNLayer(nn.Module):
    def __init__(
        self,
        num_levels: int,
        num_channels: int,
        block: Callable[[int, int], nn.Module] = SeparableConv2d,
        interpolation_mode="nearest",
        eps: float = 1e-4,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.td_fuses = nn.ModuleList(
            [
                WeightedFeatureFusion(num_channels, block=block, eps=eps)
                for _ in range(num_levels - 1)
            ]
        )
        self.out_fuses = nn.ModuleList(
            [
                WeightedFeatureFusion(num_channels, num_inputs=3, block=block, eps=eps)
                for _ in range(num_levels - 2)
            ]
        )
        self.out_fuses.append(WeightedFeatureFusion(num_channels, block=block, eps=eps))

        self.upsample = nn.Upsample(scale_factor=2.0, mode=interpolation_mode)
        self.downsample = nn.Upsample(scale_factor=0.5, mode=interpolation_mode)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # top-down
        tds = [None] * self.num_levels
        tds[-1] = x[-1]
        for i in range(self.num_levels - 2, -1, -1):
            tds[i] = self.td_fuses[i](
                [x[i], self.upsample(tds[i + 1])]
            )  # P6td = conv(P6in + resize(P7td))

        # bottom-up
        outs = [None] * self.num_levels
        outs[0] = tds[0]
        for i in range(self.num_levels - 2):
            outs[i + 1] = self.out_fuses[i](
                [x[i + 1], tds[i + 1], self.downsample(tds[i])]
            )  # P4in + P4td + resize(P3td)
        outs[-1] = self.out_fuses[-1](
            [x[-1], self.downsample(tds[-2])]
        )  # P7in + resize(P6td)

        return outs


class WeightedFeatureFusion(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_inputs: int = 2,
        block: Callable[[int, int], nn.Module] = SeparableConv2d,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float))
        self.conv = block(num_channels, num_channels)
        self.eps = eps

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        weights = F.relu(self.weights)
        out = 0
        for i in range(weights):
            out += x[i] * weights[i]
        return self.conv(out / (weights.sum() + self.eps))
