# https://arxiv.org/abs/2201.03545
# https://github.com/facebookresearch/ConvNeXt
# https://arxiv.org/abs/2301.00808
# https://github.com/facebookresearch/ConvNeXt-V2

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..components import LayerScale, Permute, StochasticDepth
from .base import BaseBackbone


class GlobalResponseNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: shape (B, H, W, C)
        gx = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True)  # (B, 1, 1, C)
        nx = gx / gx.mean(-1, keepdim=True).add(self.eps)
        return x + x * nx * self.gamma + self.beta


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        expansion_ratio: float = 4.0,
        bias: bool = True,
        layer_scale_init: float | None = 1e-6,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-6,
        v2: bool = False,
    ) -> None:
        if v2:
            layer_scale_init = None
        super().__init__()
        hidden_dim = int(d_model * expansion_ratio)
        self.layers = nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(d_model, d_model, 7, padding=3, groups=d_model, bias=bias),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(d_model, norm_eps),
            nn.Linear(d_model, hidden_dim, bias=bias),
            nn.GELU(),
            GlobalResponseNorm(hidden_dim) if v2 else nn.Identity(),
            nn.Linear(hidden_dim, d_model, bias=bias),
            LayerScale(d_model, layer_scale_init) if layer_scale_init is not None else nn.Identity(),
            StochasticDepth(stochastic_depth),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)


class ConvNeXt(BaseBackbone):
    def __init__(
        self,
        d_model: int,
        depths: tuple[int, ...],
        expansion_ratio: float = 4.0,
        bias: bool = True,
        layer_scale_init: float | None = 1e-6,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-6,
        v2: bool = False,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, d_model, 4, 4), Permute(0, 2, 3, 1), nn.LayerNorm(d_model, norm_eps))

        stochastic_depth_rates = torch.linspace(0, stochastic_depth, sum(depths))
        self.stages = nn.Sequential()

        for stage_idx, depth in enumerate(depths):
            stage = nn.Sequential()
            if stage_idx > 0:
                # equivalent to PatchMerging in SwinTransformer
                downsample = nn.Sequential(
                    nn.LayerNorm(d_model, norm_eps),
                    Permute(0, 3, 1, 2),
                    nn.Conv2d(d_model, d_model * 2, 2, 2),
                    Permute(0, 2, 3, 1),
                )
                d_model *= 2
            else:
                downsample = nn.Identity()
            stage.append(downsample)

            for block_idx in range(depth):
                rate = stochastic_depth_rates[sum(depths[:stage_idx]) + block_idx]
                block = ConvNeXtBlock(d_model, expansion_ratio, bias, layer_scale_init, rate, norm_eps, v2)
                stage.append(block)

            self.stages.append(stage)

        self.norm = nn.LayerNorm(d_model, norm_eps)

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        out = [self.stem(x)]
        for stage in self.stages:
            out.append(stage(out[-1]))
        return out[-1:]

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.get_feature_maps(x)[-1].mean((1, 2)))

    @staticmethod
    def from_config(variant: str, v2: bool = False, pretrained: bool = False) -> ConvNeXt:
        d_model, depths = dict(
            A=(40, (2, 2, 6, 2)),
            F=(48, (2, 2, 6, 2)),
            P=(64, (2, 2, 6, 2)),
            N=(80, (2, 2, 8, 2)),
            T=(96, (3, 3, 9, 3)),
            S=(96, (3, 3, 27, 3)),
            B=(128, (3, 3, 27, 3)),
            L=(192, (3, 3, 27, 3)),
            XL=(256, (3, 3, 27, 3)),
            H=(352, (3, 3, 27, 3)),
        )[variant]
        m = ConvNeXt(d_model, depths, v2=v2)

        if pretrained:
            # TODO: also add torchvision checkpoints?
            if not v2:
                ckpt = dict(
                    T="convnext_tiny_22k_224.pth",
                    S="convnext_small_22k_224.pth",
                    B="convnext_base_22k_224.pth",
                    L="convnext_large_22k_224.pth",
                    XL="convnext_xlarge_22k_224.pth",
                )[variant]
                base_url = "https://dl.fbaipublicfiles.com/convnext/"
            else:
                ckpt = dict(
                    A="convnextv2_atto_1k_224_fcmae.pt",
                    F="convnextv2_femto_1k_224_fcmae.pt",
                    P="convnextv2_pico_1k_224_fcmae.pt",
                    N="convnextv2_nano_1k_224_fcmae.pt",
                    T="convnextv2_tiny_1k_224_fcmae.pt",
                    B="convnextv2_base_1k_224_fcmae.pt",
                    L="convnextv2_large_1k_224_fcmae.pt",
                    H="convnextv2_huge_1k_224_fcmae.pt",
                )[variant]
                base_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/"
            state_dict = torch.hub.load_state_dict_from_url(base_url + ckpt)["model"]
            m.load_official_ckpt(state_dict)

        return m

    @torch.no_grad()
    def load_official_ckpt(self, state_dict: dict[str, Tensor]) -> None:
        def copy_(m: nn.Conv2d | nn.Linear | nn.LayerNorm, prefix: str):
            m.weight.copy_(state_dict.pop(prefix + ".weight"))
            m.bias.copy_(state_dict.pop(prefix + ".bias"))

        copy_(self.stem[0], "downsample_layers.0.0")  # Conv2d
        copy_(self.stem[2], "downsample_layers.0.1")  # LayerNorm

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx > 0:
                copy_(stage[0][0], f"downsample_layers.{stage_idx}.0")  # LayerNorm
                copy_(stage[0][2], f"downsample_layers.{stage_idx}.1")  # Conv2d

            for block_idx in range(1, len(stage)):
                block: ConvNeXtBlock = stage[block_idx]
                prefix = f"stages.{stage_idx}.{block_idx - 1}."

                copy_(block.layers[1], prefix + "dwconv")
                copy_(block.layers[3], prefix + "norm")
                copy_(block.layers[4], prefix + "pwconv1")

                if isinstance(block.layers[6], GlobalResponseNorm):  # v2
                    block.layers[6].gamma.copy_(state_dict.pop(prefix + "grn.gamma").squeeze())
                    block.layers[6].beta.copy_(state_dict.pop(prefix + "grn.beta").squeeze())

                copy_(block.layers[7], prefix + "pwconv2")
                if isinstance(block.layers[8], LayerScale):
                    block.layers[8].gamma.copy_(state_dict.pop(prefix + "gamma"))

        # FCMAE checkpoints don't contain head norm
        if "norm.weight" in state_dict:
            copy_(self.norm, "norm")
            assert len(state_dict) == 2
        else:
            assert len(state_dict) == 0
