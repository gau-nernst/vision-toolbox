from abc import ABCMeta, abstractmethod
from typing import Callable, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn

from .components import ConvBnAct, SeparableConv2d


__all__ = ["FPN", "PAN", "BiFPN"]


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


# https://arxiv.org/abs/1612.03144
class FPN(nn.Module):
    def __init__(
        self,
        in_channels_list: Iterable[int],
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
                nn.Conv2d(in_c, out_channels, kernel_size=1) if in_c != out_channels else nn.Identity()
                for in_c in in_channels_list
            ]
        )
        self.upsample = nn.Upsample(scale_factor=2.0 if top_down else 0.5, mode=interpolation_mode)
        in_c = out_channels if fuse_fn == "sum" else out_channels * 2
        self.output_convs = nn.ModuleList([block(in_c, out_channels) for _ in range(len(in_channels_list) - 1)])

    def _fuse_top_down(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for i, output_conv in enumerate(self.output_convs):
            x[-2 - i] = self.fuse([x[-2 - i], self.upsample(x[-1 - i])])  # 2, 1, 0
            x[-2 - i] = output_conv(x[-2 - i])
        return x

    def _fuse_bottom_up(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for i, output_conv in enumerate(self.output_convs):
            x[i + 1] = self.fuse([x[i + 1], self.upsample(x[i])])  # 1, 2, 3
            x[i + 1] = output_conv(x[i + 1])
        return x

    # input feature maps are ordered from bottom (largest) to top (smallest)
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(x) == len(self.lateral_convs)
        outputs = [l_conv(x[i]) for i, l_conv in enumerate(self.lateral_convs)]
        if self.top_down:
            return self._fuse_top_down(outputs)
        return self._fuse_bottom_up(outputs)


# https://arxiv.org/abs/1803.01534
class PAN(nn.Module):
    def __init__(
        self,
        in_channels_list: Iterable[int],
        out_channels: int = 256,
        fuse_fn: str = "sum",
        block: Callable[[int, int], nn.Module] = ConvBnAct,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.top_down = FPN(
            in_channels_list,
            out_channels,
            fuse_fn=fuse_fn,
            block=block,
            interpolation_mode=interpolation_mode,
        )
        self.bottom_up = FPN(
            [out_channels] * len(in_channels_list),
            out_channels,
            fuse_fn=fuse_fn,
            block=block,
            interpolation_mode=interpolation_mode,
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = self.top_down(x)
        outputs = self.bottom_up(outputs)
        return outputs


# https://arxiv.org/pdf/1911.09070.pdf
# https://github.com/google/automl/blob/master/efficientdet/efficientdet_arch.py
class BiFPN(nn.Module):
    def __init__(
        self,
        in_channels_list: Iterable[int],
        out_channels: int = 64,
        num_layers: int = 1,
        block: Callable[[int, int], nn.Module] = SeparableConv2d,
        interpolation_mode="nearest",
        eps: float = 1e-4,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.laterals = nn.ModuleList([nn.Conv2d(in_c, out_channels, kernel_size=1) for in_c in in_channels_list])
        self.layers = nn.ModuleList(
            [
                BiFPNLayer(
                    len(in_channels_list),
                    out_channels,
                    block=block,
                    interpolation_mode=interpolation_mode,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(x) == len(self.laterals)
        outputs = [l_conv(x[i]) for i, l_conv in enumerate(self.laterals)]
        for layer in self.layers:
            outputs = layer(outputs)
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
            [WeightedFeatureFusion(num_channels, block=block, eps=eps) for _ in range(num_levels - 1)]
        )
        self.out_fuses = nn.ModuleList(
            [WeightedFeatureFusion(num_channels, num_inputs=3, block=block, eps=eps) for _ in range(num_levels - 2)]
        )
        self.last_out_fuse = WeightedFeatureFusion(num_channels, block=block, eps=eps)

        self.upsample = nn.Upsample(scale_factor=2.0, mode=interpolation_mode)
        self.downsample = nn.Upsample(scale_factor=0.5, mode=interpolation_mode)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # top-down, P6td = conv(P6in + resize(P7td))
        tds = list(x)  # make a copy
        for i, td_fuse in enumerate(self.td_fuses):
            tds[-2 - i] = td_fuse([x[-2 - i], self.upsample(tds[-1 - i])])

        # bottom-up, P4in + P4td + resize(P3td)
        outs = list(tds)
        for i, out_fuse in enumerate(self.out_fuses):
            outs[i + 1] = out_fuse([x[i + 1], tds[i + 1], self.downsample(tds[i])])

        # # P7in + resize(P6td)
        outs[-1] = self.last_out_fuse([x[-1], self.downsample(tds[-2])])
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
        for i in range(weights.shape[0]):
            out += x[i] * weights[i]
        return self.conv(out / (weights.sum() + self.eps))
