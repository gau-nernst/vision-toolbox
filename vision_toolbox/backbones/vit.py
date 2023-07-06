# https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

import torch
from torch import Tensor, nn


configs = dict(
    Ti=dict(n_layers=12, d_model=192, n_heads=3),
    S=dict(n_layers=12, d_model=384, n_heads=6),
    B=dict(n_layers=12, d_model=768, n_heads=12),
    L=dict(n_layers=24, d_model=1024, n_heads=16),
    H=dict(n_layers=32, d_model=1280, n_heads=16),
)


class ViT(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        patch_size: int,
        img_size: int | tuple[int, int],
        mlp_dim: int | None = None,
        cls_token: bool = True,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model)) if cls_token else None

        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        pe_size = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        if cls_token:
            pe_size += 1
        self.pe = nn.Parameter(torch.empty(1, pe_size, d_model))

        mlp_dim = mlp_dim or d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            mlp_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, nn.LayerNorm(d_model))

        self.reset_parameters()

    def reset_parameters(self):
        if self.cls_token is not None:
            nn.init.zeros_(self.cls_token)
        nn.init.normal_(self.pe, 0, 0.02)

    def forward(self, imgs: Tensor) -> Tensor:
        out = self.patch_embed(imgs)
        out = out.flatten(-2).transpose(-1, -2)  # (N, C, H, W) -> (N, H*W, C)
        if self.cls_token is not None:
            out = torch.cat([self.cls_token.expand(out.shape[0], -1, -1), out], 1)
        out = out + self.pe
        out = self.encoder(out)
        out = out[:, 0] if self.cls_token is not None else out.mean(1)
        return out

    @staticmethod
    def from_config(variant: str, patch_size: int, img_size: int | tuple[int, int]) -> "ViT":
        return ViT(**configs[variant], patch_size=patch_size, img_size=img_size)
