# https://arxiv.org/abs/2012.12877
# https://arxiv.org/abs/2204.07118
# https://github.com/facebookresearch/deit

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..components import LayerScale
from .base import _act, _norm
from .vit import ViTBlock


class DeiT(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int,
        n_heads: int,
        patch_size: int,
        img_size: int,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float | None = None,
        stochastic_depth: float = 0.0,
        norm: _norm = partial(nn.LayerNorm, eps=1e-6),
        act: _act = nn.GELU,
    ) -> None:
        assert img_size % patch_size == 0
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pe = nn.Parameter(torch.empty(1, (img_size // patch_size) ** 2 + 2, d_model))
        nn.init.normal_(self.pe, 0, 0.02)

        self.layers = nn.Sequential()
        for _ in range(depth):
            block = ViTBlock(d_model, n_heads, bias, mlp_ratio, dropout, layer_scale_init, stochastic_depth, norm, act)
            self.layers.append(block)

        self.norm = norm(d_model)

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        out = torch.cat([self.cls_token, self.dist_token, out], 1) + self.pe
        out = self.layers(out)
        return self.norm(out[:, :2]).mean(1)

    @torch.no_grad()
    def resize_pe(self, size: int, interpolation_mode: str = "bicubic") -> None:
        pe = self.pe[:, 2:]
        old_size = int(pe.shape[1] ** 0.5)
        new_size = size // self.patch_embed.weight.shape[2]
        pe = pe.unflatten(1, (old_size, old_size)).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, (new_size, new_size), mode=interpolation_mode)
        pe = pe.permute(0, 2, 3, 1).flatten(1, 2)
        self.pe = nn.Parameter(torch.cat((self.pe[:, :2], pe), 1))

    @staticmethod
    def from_config(variant: str, img_size: int, version: bool = False, pretrained: bool = False) -> DeiT:
        variant, patch_size = variant.split("_")

        d_model, depth, n_heads = dict(
            Ti=(192, 12, 3),
            S=(384, 12, 6),
            M=(512, 12, 8),
            B=(768, 12, 12),
            L=(1024, 24, 16),
            H=(1280, 32, 16),
        )[variant]
        patch_size = int(patch_size)
        m = DeiT(d_model, depth, n_heads, patch_size, img_size)

        if pretrained:
            ckpt = dict(
                Ti_16_224="deit_tiny_distilled_patch16_224-b40b3cf7.pth",
                S_16_224="deit_small_distilled_patch16_224-649709d9.pth",
                B_16_224="deit_base_distilled_patch16_224-df68dfff.pth",
                B_16_384="deit_base_distilled_patch16_384-d0272ac0.pth",
            )[f"{variant}_{patch_size}_{img_size}"]
            base_url = "https://dl.fbaipublicfiles.com/deit/"
            state_dict = torch.hub.load_state_dict_from_url(base_url + ckpt)["model"]
            m.load_official_ckpt(state_dict)

        return m

    @torch.no_grad()
    def load_official_ckpt(self, state_dict: dict[str, Tensor]) -> None:
        def copy_(m: nn.Linear | nn.LayerNorm, prefix: str):
            m.weight.copy_(state_dict.pop(prefix + ".weight").view(m.weight.shape))
            m.bias.copy_(state_dict.pop(prefix + ".bias"))

        copy_(self.patch_embed, "patch_embed.proj")
        self.cls_token.copy_(state_dict.pop("cls_token"))
        if self.dist_token is not None:
            self.dist_token.copy_(state_dict.pop("dist_token"))
            state_dict.pop("head_dist.weight")
            state_dict.pop("head_dist.bias")
        self.pe.copy_(state_dict.pop("pos_embed"))

        for i, block in enumerate(self.layers):
            block: ViTBlock
            prefix = f"blocks.{i}."

            copy_(block.mha[0], prefix + "norm1")
            q_w, k_w, v_w = state_dict.pop(prefix + "attn.qkv.weight").chunk(3, 0)
            block.mha[1].q_proj.weight.copy_(q_w)
            block.mha[1].k_proj.weight.copy_(k_w)
            block.mha[1].v_proj.weight.copy_(v_w)
            q_b, k_b, v_b = state_dict.pop(prefix + "attn.qkv.bias").chunk(3, 0)
            block.mha[1].q_proj.bias.copy_(q_b)
            block.mha[1].k_proj.bias.copy_(k_b)
            block.mha[1].v_proj.bias.copy_(v_b)
            copy_(block.mha[1].out_proj, prefix + "attn.proj")
            if isinstance(block.mha[2], LayerScale):
                block.mha[2].gamma.copy_(state_dict.pop(prefix + "gamma_1"))

            copy_(block.mlp[0], prefix + "norm2")
            copy_(block.mlp[1].linear1, prefix + "mlp.fc1")
            copy_(block.mlp[1].linear2, prefix + "mlp.fc2")
            if isinstance(block.mha[2], LayerScale):
                block.mha[2].gamma.copy_(state_dict.pop(prefix + "gamma_2"))

        copy_(self.norm, "norm")
        assert len(state_dict) == 2, state_dict.keys()


class DeiT3(DeiT):
    def __init__():
        pass

    #                 deit3_S_16_224="deit_3_small_224_21k.pth",
    # deit3_S_16_384="deit_3_small_384_21k.pth",
    # deit3_M_16_224="deit_3_medium_224_21k.pth",
    # deit3_B_16_224="deit_3_base_224_21k.pth",
    # deit3_B_16_384="deit_3_base_384_21k.pth",
    # deit3_L_16_224="deit_3_large_224_21k.pth",
    # deit3_L_16_384="deit_3_large_384_21k.pth",
    # deit3_H_16_224="deit_3_huge_224_21k.pth",
