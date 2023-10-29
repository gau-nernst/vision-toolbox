# https://arxiv.org/abs/2012.12877
# https://arxiv.org/abs/2204.07118
# https://github.com/facebookresearch/deit

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..components import LayerScale
from .vit import ViT, ViTBlock


class DeiT(ViT):
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
        norm_eps: float = 1e-6,
    ) -> None:
        # fmt: off
        super().__init__(
            d_model, depth, n_heads, patch_size, img_size, True, "cls_token", bias,
            mlp_ratio, dropout, layer_scale_init, stochastic_depth, norm_eps,
        )
        # fmt: on
        self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        out = torch.cat([self.cls_token, self.dist_token, out + self.pe], 1)
        out = self.layers(out)
        return self.norm(out[:, :2]).mean(1)

    @staticmethod
    def from_config(variant: str, img_size: int, pretrained: bool = False) -> DeiT:
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
        pe = state_dict.pop("pos_embed")
        self.pe.copy_(pe[:, -self.pe.shape[1] :])

        self.cls_token.copy_(state_dict.pop("cls_token"))
        if pe.shape[1] > self.pe.shape[1]:
            self.cls_token.add_(pe[:, 0])

        if hasattr(self, "dist_token"):
            self.dist_token.copy_(state_dict.pop("dist_token"))
            self.dist_token.add_(pe[:, 1])
            state_dict.pop("head_dist.weight")
            state_dict.pop("head_dist.bias")

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
            if isinstance(block.mlp[2], LayerScale):
                block.mlp[2].gamma.copy_(state_dict.pop(prefix + "gamma_2"))

        copy_(self.norm, "norm")
        assert len(state_dict) == 2, state_dict.keys()


class DeiT3(ViT):
    def __init__(
        self,
        d_model: int,
        depth: int,
        n_heads: int,
        patch_size: int,
        img_size: int,
        cls_token: bool = True,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-6,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        # fmt: off
        super().__init__(
            d_model, depth, n_heads, patch_size, img_size, cls_token, "cls_token", bias,
            mlp_ratio, dropout, layer_scale_init, stochastic_depth, norm_eps,
        )
        # fmt: on

    @staticmethod
    def from_config(variant: str, img_size: int, pretrained: bool = False) -> DeiT:
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
        m = DeiT3(d_model, depth, n_heads, patch_size, img_size)

        if pretrained:
            ckpt = dict(
                S_16_224="deit_3_small_224_21k.pth",
                S_16_384="deit_3_small_384_21k.pth",
                M_16_224="deit_3_medium_224_21k.pth",
                B_16_224="deit_3_base_224_21k.pth",
                B_16_384="deit_3_base_384_21k.pth",
                L_16_224="deit_3_large_224_21k.pth",
                L_16_384="deit_3_large_384_21k.pth",
                H_16_224="deit_3_huge_224_21k.pth",
            )[f"{variant}_{patch_size}_{img_size}"]
            base_url = "https://dl.fbaipublicfiles.com/deit/"
            state_dict = torch.hub.load_state_dict_from_url(base_url + ckpt)["model"]
            m.load_official_ckpt(state_dict)

        return m

    @torch.no_grad()
    def load_official_ckpt(self, state_dict: dict[str, Tensor]) -> None:
        DeiT.load_official_ckpt(self, state_dict)
