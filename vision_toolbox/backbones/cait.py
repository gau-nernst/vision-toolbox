# https://arxiv.org/abs/2103.17239
# https://github.com/facebookresearch/deit

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .vit import MHA, ViT, ViTBlock


# basically attention pooling
class ClassAttention(MHA):
    def forward(self, x: Tensor) -> None:
        q = self.q_proj(x[:, 0]).unflatten(-1, (self.n_heads, -1)).unsqueeze(2)  # (B, n_heads, 1, head_dim)
        k = self.k_proj(x).unflatten(-1, (self.n_heads, -1)).transpose(-2, -3)  # (B, n_heads, L, head_dim)
        v = self.v_proj(x).unflatten(-1, (self.n_heads, -1)).transpose(-2, -3)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, None, self.dropout if self.training else 0.0)
        else:
            attn = (q * self.scale) @ k.transpose(-1, -2)
            out = F.dropout(torch.softmax(attn, -1), self.dropout, self.training) @ v

        return self.out_proj(out.flatten(1))  # (B, n_heads, 1, head_dim) -> (B, n_heads * head_dim)


# does not support flash attention
class TalkingHeadAttention(MHA):
    def __init__(self, d_model: int, n_heads: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__(d_model, n_heads, bias, dropout)
        self.talking_head_proj = nn.Sequential(
            nn.Conv2d(n_heads, n_heads, 1),  # impl as 1x1 conv to avoid permutating data
            nn.Softmax(-1),
            nn.Conv2d(n_heads, n_heads, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_proj(x).unflatten(-1, (self.n_heads, -1)).transpose(-2, -3)  # (B, n_heads, L, head_dim)
        k = self.k_proj(x).unflatten(-1, (self.n_heads, -1)).transpose(-2, -3)
        v = self.v_proj(x).unflatten(-1, (self.n_heads, -1)).transpose(-2, -3)

        attn = q @ (k * self.scale).transpose(-1, -2)
        out = self.talking_head_proj(attn) @ v
        out = out.transpose(-2, -3).flatten(-2)
        out = self.out_proj(out)
        return out


class CaiTCABlock(ViTBlock):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-6,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        # fmt: off
        super().__init__(
            d_model, n_heads, bias, mlp_ratio, dropout,
            layer_scale_init, stochastic_depth, norm_eps,
            partial(ClassAttention, d_model, n_heads, bias, dropout),
        )
        # fmt: on

    def forward(self, x: Tensor, cls_token: Tensor) -> Tensor:
        cls_token = cls_token + self.mha(torch.cat((cls_token, x), 1))
        cls_token = cls_token + self.mlp(cls_token)
        return cls_token


class CaiTSABlock(ViTBlock):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-6,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        # fmt: off
        super().__init__(
            d_model, n_heads, bias, mlp_ratio, dropout,
            layer_scale_init, stochastic_depth, norm_eps,
            partial(TalkingHeadAttention, d_model, n_heads, bias, dropout),
        )
        # fmt: on


class CaiT(nn.Module):
    def __init__(
        self,
        d_model: int,
        sa_depth: int,
        ca_depth: int,
        n_heads: int,
        patch_size: int,
        img_size: int,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1e-6,
        stochastic_depth: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        assert img_size % patch_size == 0
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pe = nn.Parameter(torch.empty(1, (img_size // patch_size) ** 2, d_model))
        nn.init.normal_(self.pe, 0, 0.02)

        self.sa_layers = nn.Sequential()
        for _ in range(sa_depth):
            blk = CaiTSABlock(d_model, n_heads, bias, mlp_ratio, dropout, layer_scale_init, stochastic_depth, norm_eps)
            self.sa_layers.append(blk)

        self.ca_layers = nn.ModuleList()
        for _ in range(ca_depth):
            blk = CaiTCABlock(d_model, n_heads, bias, mlp_ratio, dropout, layer_scale_init, stochastic_depth, norm_eps)
            self.ca_layers.append(blk)

        self.norm = nn.LayerNorm(d_model, norm_eps)

    def forward(self, imgs: Tensor) -> Tensor:
        patches = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (N, C, H, W) -> (N, H*W, C)
        patches = self.sa_layers(patches + self.pe)

        cls_token = self.cls_token
        for block in self.ca_layers:
            cls_token = block(patches, cls_token)
        return self.norm(cls_token.squeeze(1))

    @torch.no_grad()
    def resize_pe(self, size: int, interpolation_mode: str = "bicubic") -> None:
        ViT.resize_pe(self, size, interpolation_mode)

    @staticmethod
    def from_config(variant: str, img_size: int, pretrained: bool = False) -> CaiT:
        variant, sa_depth = variant.split("_")

        d_model = dict(xxs=192, xs=288, s=384, m=768)[variant]
        sa_depth = int(sa_depth)
        ca_depth = 2
        n_heads = d_model // 48
        patch_size = 16
        m = CaiT(d_model, sa_depth, ca_depth, n_heads, patch_size, img_size)

        if pretrained:
            ckpt = dict(
                xxs_24_224="XXS24_224.pth",
                xxs_24_384="XXS24_384.pth",
                xxs_36_224="XXS36_224.pth",
                xxs_36_384="XXS36_384.pth",
                xs_24_384="XS24_384.pth",
                s_24_224="S24_224.pth",
                s_24_384="S24_384.pth",
                s_36_384="S36_384.pth",
                m_36_384="M36_384.pth",
                m_48_448="M48_448.pth",
            )[f"{variant}_{sa_depth}_{img_size}"]
            base_url = "https://dl.fbaipublicfiles.com/deit/"
            state_dict = torch.hub.load_state_dict_from_url(base_url + ckpt)["model"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            m.load_official_ckpt(state_dict)

        return m

    @torch.no_grad()
    def load_official_ckpt(self, state_dict: dict[str, Tensor]) -> None:
        def copy_(m: nn.Linear | nn.LayerNorm, prefix: str):
            m.weight.copy_(state_dict.pop(prefix + ".weight").view(m.weight.shape))
            m.bias.copy_(state_dict.pop(prefix + ".bias"))

        copy_(self.patch_embed, "patch_embed.proj")
        self.cls_token.copy_(state_dict.pop("cls_token"))
        self.pe.copy_(state_dict.pop("pos_embed"))

        for i, sa_block in enumerate(self.sa_layers):
            sa_block: CaiTSABlock
            prefix = f"blocks.{i}."

            copy_(sa_block.mha[0], prefix + "norm1")
            q_w, k_w, v_w = state_dict.pop(prefix + "attn.qkv.weight").chunk(3, 0)
            sa_block.mha[1].q_proj.weight.copy_(q_w)
            sa_block.mha[1].k_proj.weight.copy_(k_w)
            sa_block.mha[1].v_proj.weight.copy_(v_w)
            q_b, k_b, v_b = state_dict.pop(prefix + "attn.qkv.bias").chunk(3, 0)
            sa_block.mha[1].q_proj.bias.copy_(q_b)
            sa_block.mha[1].k_proj.bias.copy_(k_b)
            sa_block.mha[1].v_proj.bias.copy_(v_b)
            copy_(sa_block.mha[1].out_proj, prefix + "attn.proj")
            copy_(sa_block.mha[1].talking_head_proj[0], prefix + "attn.proj_l")
            copy_(sa_block.mha[1].talking_head_proj[2], prefix + "attn.proj_w")
            sa_block.mha[2].gamma.copy_(state_dict.pop(prefix + "gamma_1"))

            copy_(sa_block.mlp[0], prefix + "norm2")
            copy_(sa_block.mlp[1].linear1, prefix + "mlp.fc1")
            copy_(sa_block.mlp[1].linear2, prefix + "mlp.fc2")
            sa_block.mlp[2].gamma.copy_(state_dict.pop(prefix + "gamma_2"))

        for i, ca_block in enumerate(self.ca_layers):
            ca_block: CaiTCABlock
            prefix = f"blocks_token_only.{i}."

            copy_(ca_block.mha[0], prefix + "norm1")
            copy_(ca_block.mha[1].q_proj, prefix + "attn.q")
            copy_(ca_block.mha[1].k_proj, prefix + "attn.k")
            copy_(ca_block.mha[1].v_proj, prefix + "attn.v")
            copy_(ca_block.mha[1].out_proj, prefix + "attn.proj")
            ca_block.mha[2].gamma.copy_(state_dict.pop(prefix + "gamma_1"))

            copy_(ca_block.mlp[0], prefix + "norm2")
            copy_(ca_block.mlp[1].linear1, prefix + "mlp.fc1")
            copy_(ca_block.mlp[1].linear2, prefix + "mlp.fc2")
            ca_block.mlp[2].gamma.copy_(state_dict.pop(prefix + "gamma_2"))

        copy_(self.norm, "norm")
        assert len(state_dict) == 2
