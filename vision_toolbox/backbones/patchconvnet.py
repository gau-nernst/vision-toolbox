import torch
from torch import nn
from torchvision.ops.misc import SqueezeExcitation

from .base import BaseBackbone


_S_width = 384
_mlp_ratio = 3
configs = {
    "S60": {
        "width": _S_width,
        "depth": 60,
        "mlp_ratio": _mlp_ratio
    },
    "S120": {
        "width": _S_width,
        "depth": 120,
        "mlp_ratio": _mlp_ratio
    }
}


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchConvBlock(nn.Module):
    def __init__(self, width, layer_scale_init=1e-6):
        super().__init__()
        self.layers = nn.Sequential(
            LayerNorm2d(width),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.GELU(),
            SqueezeExcitation(width, width // 4),
            nn.Conv2d(width, width, 1)
        )
        self.layer_scale = nn.Parameter(torch.full((1,width,1,1), layer_scale_init))

    def forward(self, x: torch.Tensor):
        # (N, C, H, W)
        return x + self.layers(x) * self.layer_scale


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, layer_scale_init=1e-6):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(embed_dim))
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.layer_scale_1 = nn.Parameter(torch.full((1,1,embed_dim), layer_scale_init))

        mlp_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.layer_scale_2 = nn.Parameter(torch.full((1,1,embed_dim), layer_scale_init))
        self.norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # (N, HW, C)
        cls_token = self.cls_token.expand(x.shape[0], 1, -1)
        out = torch.concat((cls_token, x), dim=1)
        out = self.norm_1(out)
        out, _ = self.attn(out[:,:1], out, out)
        
        cls_token = cls_token + out * self.layer_scale_1
        cls_token = cls_token + cls_token * self.layer_scale_2
        cls_token = self.norm_2(cls_token)
        return cls_token


class PatchConvNet(BaseBackbone):
    def __init__(self, embed_dim, depth, mlp_ratio):
        super().__init__()
        stem_layers = []
        in_c, out_c = 3, embed_dim // 8
        for _ in range(4):
            # original code uses bias=False even though there is no norm layer
            stem_layers.append(nn.Conv2d(in_c, out_c, 3, stride=2, padding=1))
            stem_layers.append(nn.GELU())
            in_c, out_c = out_c, out_c * 2
        self.stem = nn.Sequential(*stem_layers)

        self.blocks = nn.Sequential(*[PatchConvBlock(embed_dim) for _ in range(depth)])
        self.pool = AttentionPooling(embed_dim, mlp_ratio)

    def forward_features(self, x):
        out = self.stem(x)
        out = self.blocks(out)
        out = out.flatten(-2).transpose(-1, -2)         # (N, C, H, W) -> (N, HW, C)
        out = self.pool(out)
        return out

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)


def S60(pretrained=False, **kwargs): return PatchConvNet.from_config(configs["S60"], pretrained=pretrained, **kwargs)
def S120(pretrained=False, **kwargs): return PatchConvNet.from_config(configs["S120"], pretrained=pretrained, **kwargs)
