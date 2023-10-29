# https://arxiv.org/abs/2103.14030
# https://github.com/microsoft/Swin-Transformer

from __future__ import annotations

import itertools
from functools import partial

import torch
from torch import Tensor, nn

from .base import BaseBackbone
from .vit import MHA, ViTBlock


def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, int, int]:
    B, H, W, C = x.shape
    nH, nW = H // window_size, W // window_size
    x = x.view(B, nH, window_size, nW, window_size, C)
    x = x.transpose(2, 3).reshape(B * nH * nW, window_size * window_size, C)
    return x, nH, nW


def window_unpartition(x: Tensor, window_size: int, nH: int, nW: int) -> Tensor:
    B = x.shape[0] // (nH * nW)
    C = x.shape[2]
    x = x.view(B, nH, nW, window_size, window_size, C)
    x = x.transpose(2, 3).reshape(B, nH * window_size, nW * window_size, C)
    return x


class WindowAttention(MHA):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_heads: int,
        window_size: int = 7,
        shift: bool = False,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(d_model, n_heads, bias, dropout)
        self.input_size = input_size
        self.window_size = window_size

        if shift:
            self.shift = window_size // 2

            img_mask = torch.zeros(1, input_size, input_size, 1)
            slices = (slice(0, -window_size), slice(-window_size, -self.shift), slice(-self.shift, None))
            for i, (h_slice, w_slice) in enumerate(itertools.product(slices, slices)):
                img_mask[:, h_slice, w_slice, :] = i

            windows_mask, _, _ = window_partition(img_mask, window_size)  # (nH * nW, win_size * win_size, 1)
            attn_mask = windows_mask.transpose(1, 2) - windows_mask
            self.register_buffer("attn_mask", (attn_mask != 0) * (-100.0), False)
            self.attn_mask: Tensor

        else:
            self.shift = 0
            self.attn_mask = None

        self.relative_pe_table = nn.Parameter(torch.empty(1, n_heads, (2 * window_size - 1) ** 2))
        nn.init.trunc_normal_(self.relative_pe_table, 0, 0.02)

        xy = torch.cartesian_prod(torch.arange(window_size), torch.arange(window_size))  # all possible (x,y) pairs
        diff = xy.unsqueeze(1) - xy.unsqueeze(0)  # difference between all (x,y) pairs
        index = (diff[:, :, 0] + window_size - 1) * (2 * window_size - 1) + diff[:, :, 1] + window_size - 1
        self.register_buffer("relative_pe_index", index, False)
        self.relative_pe_index: Tensor

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.input_size, (x.shape[1], self.input_size)
        attn_bias = self.relative_pe_table[..., self.relative_pe_index]
        if self.shift > 0:
            x = x.roll((-self.shift, -self.shift), (1, 2))
            attn_bias = attn_bias + self.attn_mask.unsqueeze(1)  # add n_heads dim

        x, nH, nW = window_partition(x, self.window_size)  # (B * nH * nW, win_size * win_size, C)
        x = super().forward(x, attn_bias=attn_bias)
        x = window_unpartition(x, self.window_size, nH, nW)  # (B, H, W, C)

        if self.shift > 0:
            x = x.roll((self.shift, self.shift), (1, 2))
        return x


class SwinBlock(ViTBlock):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_heads: int,
        window_size: int = 7,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        bias: bool = True,
        dropout: float = 0.0,
        layer_scale_init: float | None = None,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        # fmt: off
        super().__init__(
            d_model, n_heads, bias, mlp_ratio, dropout,
            layer_scale_init, stochastic_depth, norm_eps,
            partial(WindowAttention, input_size, d_model, n_heads, window_size, shift, bias, dropout),
        )
        # fmt: on


class PatchMerging(nn.Module):
    def __init__(self, d_model: int, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model * 4, norm_eps)
        self.reduction = nn.Linear(d_model * 4, d_model * 2, False)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // 2, 2, W // 2, 2, C).transpose(2, 3).flatten(-3)
        x = self.reduction(self.norm(x))
        x = x.view(B, H // 2, W // 2, C * 2)
        return x


class SwinTransformer(BaseBackbone):
    def __init__(
        self,
        img_size: int,
        d_model: int,
        n_heads: int,
        depths: tuple[int, ...],
        window_sizes: tuple[int, ...],
        patch_size: int = 4,
        mlp_ratio: float = 4.0,
        bias: bool = True,
        dropout: float = 0.0,
        layer_scale_init: float | None = None,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        assert img_size % patch_size == 0
        assert d_model % n_heads == 0
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.patch_norm = nn.LayerNorm(d_model, norm_eps)
        self.dropout = nn.Dropout(dropout)

        input_size = img_size // patch_size
        self.stages = nn.Sequential()
        for i, (depth, window_size) in enumerate(zip(depths, window_sizes)):
            stage = nn.Sequential()
            if i > 0:
                downsample = PatchMerging(d_model, norm_eps)
                input_size //= 2
                d_model *= 2
                n_heads *= 2
            else:
                downsample = nn.Identity()
            stage.append(downsample)

            for i in range(depth):
                shift = (i % 2) and input_size > window_size
                # fmt: off
                block = SwinBlock(
                    input_size, d_model, n_heads, window_size, shift, mlp_ratio,
                    bias, dropout, layer_scale_init, stochastic_depth, norm_eps,
                )
                # fmt: on
                stage.append(block)

            self.stages.append(stage)

        self.norm = nn.LayerNorm(d_model, norm_eps)

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        out = [self.dropout(self.patch_norm(self.patch_embed(x).permute(0, 2, 3, 1)))]
        for stage in self.stages:
            out.append(stage(out[-1]))
        return out[1:]

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.get_feature_maps(x)[-1]).mean((1, 2))

    def resize_pe(self, img_size: int) -> None:
        raise NotImplementedError()

    @staticmethod
    def from_config(variant: str, img_size: int, pretrained: bool = False) -> SwinTransformer:
        d_model, n_heads, depths, window_sizes, ckpt = {
            # Sub-section 3.3 in https://arxiv.org/pdf/2103.14030.pdf
            "T": (96, 3, (2, 2, 6, 2), (7, 7, 7, 7), "v1.0.8/swin_tiny_patch4_window7_224_22k.pth"),
            "S": (96, 3, (2, 2, 18, 2), (7, 7, 7, 7), "v1.0.8/swin_small_patch4_window7_224_22k.pth"),
            "B": (128, 4, (2, 2, 18, 2), (7, 7, 7, 7), "v1.0.0/swin_base_patch4_window7_224_22k.pth"),
            "L": (192, 6, (2, 2, 18, 2), (7, 7, 7, 7), "v1.0.0/swin_large_patch4_window7_224_22k.pth"),
            # https://github.com/microsoft/Cream/blob/main/AutoFormerV2/configs
            "S3-T": (96, 3, (2, 2, 6, 2), (7, 7, 14, 7), "supernet-tiny.pth"),
            "S3-S": (96, 3, (2, 2, 18, 2), (14, 14, 14, 14), "supernet-small.pth"),
            "S3-B": (96, 3, (2, 2, 30, 2), (7, 7, 14, 7), "supernet-base.pth"),
        }[variant]
        m = SwinTransformer(224 if pretrained else img_size, d_model, n_heads, depths, window_sizes)

        if pretrained:
            if variant.startswith("S3"):
                base_url = "https://github.com/silent-chen/AutoFormer-model-zoo/releases/download/v1.0/"
            else:
                base_url = "https://github.com/SwinTransformer/storage/releases/download/"
            state_dict = torch.hub.load_state_dict_from_url(base_url + ckpt)["model"]
            m.load_official_ckpt(state_dict)
            if img_size != 224:
                m.resize_pe(img_size)

        return m

    @torch.no_grad()
    def load_official_ckpt(self, state_dict: dict[str, Tensor]) -> None:
        def copy_(m: nn.Linear | nn.LayerNorm, prefix: str) -> None:
            m.weight.copy_(state_dict.pop(prefix + ".weight"))
            m.bias.copy_(state_dict.pop(prefix + ".bias"))

        copy_(self.patch_embed, "patch_embed.proj")
        copy_(self.patch_norm, "patch_embed.norm")

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx > 0:
                prefix = f"layers.{stage_idx-1}.downsample."

                def rearrange(p):
                    p1, p2, p3, p4 = p.chunk(4, -1)
                    return torch.cat((p1, p3, p2, p4), -1)

                stage[0].norm.weight.copy_(rearrange(state_dict.pop(prefix + "norm.weight")))
                stage[0].norm.bias.copy_(rearrange(state_dict.pop(prefix + "norm.bias")))
                stage[0].reduction.weight.copy_(rearrange(state_dict.pop(prefix + "reduction.weight")))

            for block_idx in range(1, len(stage)):
                block: SwinBlock = stage[block_idx]
                prefix = f"layers.{stage_idx}.blocks.{block_idx - 1}."
                block_idx += 1

                copy_(block.mha[0], prefix + "norm1")
                if block.mha[1].attn_mask is not None:
                    torch.testing.assert_close(block.mha[1].attn_mask, state_dict.pop(prefix + "attn_mask"))
                torch.testing.assert_close(
                    block.mha[1].relative_pe_index, state_dict.pop(prefix + "attn.relative_position_index")
                )
                q_w, k_w, v_w = state_dict.pop(prefix + "attn.qkv.weight").chunk(3, 0)
                block.mha[1].q_proj.weight.copy_(q_w)
                block.mha[1].k_proj.weight.copy_(k_w)
                block.mha[1].v_proj.weight.copy_(v_w)
                q_b, k_b, v_b = state_dict.pop(prefix + "attn.qkv.bias").chunk(3, 0)
                block.mha[1].q_proj.bias.copy_(q_b)
                block.mha[1].k_proj.bias.copy_(k_b)
                block.mha[1].v_proj.bias.copy_(v_b)
                copy_(block.mha[1].out_proj, prefix + "attn.proj")
                block.mha[1].relative_pe_table.copy_(state_dict.pop(prefix + "attn.relative_position_bias_table").T)
                copy_(block.mlp[0], prefix + "norm2")
                copy_(block.mlp[1].linear1, prefix + "mlp.fc1")
                copy_(block.mlp[1].linear2, prefix + "mlp.fc2")

        copy_(self.norm, "norm")
        assert len(state_dict) == 2  # head.weight and head.bias
