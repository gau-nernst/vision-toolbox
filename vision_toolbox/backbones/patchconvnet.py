# https://arxiv.org/abs/2112.13692
# https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py
import warnings

import torch
from torch import nn
try:
    from torchvision.ops.misc import SqueezeExcitation
    from torchvision.ops import StochasticDepth
except ImportError:
    warnings.warn('torchvision.ops.misc.SqueezeExcitation is not available. Please update your torchvision')
    SqueezeExcitation = None

from .base import BaseBackbone


__all__ = [
    'AttentionPooling', 'PatchConvNet',
    'PatchConvNet_S60', 'PatchConvNet_S120',
    'PatchConvNet_B60', 'PatchConvNet_B120',
    'PatchConvNet_L60', 'PatchConvNet_L120'
]


_base = {
    'mlp_ratio': 3,
    'drop_path': 0.3
}
_S_embed_dim = 384
_B_embed_dim = 768
_L_embed_dim = 1024
configs = {
    'PatchConvNet-S60': {
        **_base,
        'embed_dim': _S_embed_dim,
        'depth': 60,
    },
    'PatchConvNet-S120': {
        **_base,
        'embed_dim': _S_embed_dim,
        'depth': 120,
    },
    'PatchConvNet-B60': {
        **_base,
        'embed_dim': _B_embed_dim,
        'depth': 60,
    },
    'PatchConvNet-B120': {
        **_base,
        'embed_dim': _B_embed_dim,
        'depth': 120,
    },
    'PatchConvNet-L60': {
        **_base,
        'embed_dim': _L_embed_dim,
        'depth': 60,
    },
    'PatchConvNet-L120': {
        **_base,
        'embed_dim': _L_embed_dim,
        'depth': 120,
    }
}


# https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py#L31
# this is very slow
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchConvBlock(nn.Module):
    def __init__(self, embed_dim, layer_scale_init=1e-6, drop_path=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            # LayerNorm2d(embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
            nn.GELU(),
            SqueezeExcitation(embed_dim, embed_dim // 4),
            nn.Conv2d(embed_dim, embed_dim, 1)
        )
        self.layer_scale = nn.Parameter(torch.ones((embed_dim,1,1)) * layer_scale_init)
        self.drop_path = StochasticDepth(drop_path, 'row') if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        # (N, C, H, W)
        return x + self.drop_path(self.layers(x) * self.layer_scale)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, layer_scale_init=1e-6, drop_path=0.3):
        super().__init__()
        self.drop_path = StochasticDepth(drop_path, 'row') if drop_path > 0 else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(embed_dim))
        self.norm_1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.layer_scale_1 = nn.Parameter(torch.ones(embed_dim) * layer_scale_init)
        self.norm_2 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.layer_scale_2 = nn.Parameter(torch.ones(embed_dim) * layer_scale_init)
        self.norm_3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # (N, HW, C)
        cls_token = self.cls_token.expand(x.shape[0], 1, -1)
        combined = torch.cat((cls_token, x), dim=1)
        combined = self.norm_1(combined)

        cls_token, _ = self.attn(combined[:,:1], combined, combined, need_weights=False)
        cls_token = torch.flatten(cls_token, 1)                     # (N, 1, C) -> (N, C)
        cls_token = cls_token + self.drop_path(cls_token * self.layer_scale_1)
        cls_token = self.norm_2(cls_token)

        cls_token = self.mlp(cls_token)
        cls_token = cls_token + self.drop_path(cls_token * self.layer_scale_2)
        cls_token = self.norm_3(cls_token)

        return cls_token


class PatchConvNet(BaseBackbone):
    def __init__(self, embed_dim, depth, mlp_ratio, drop_path=0.3):
        super().__init__()
        self.out_channels = (embed_dim,)

        stem_layers = []
        in_c, out_c = 3, embed_dim // 8
        for _ in range(4):
            # original code uses bias=False even though there is no norm layer
            stem_layers.append(nn.Conv2d(in_c, out_c, 3, stride=2, padding=1))
            stem_layers.append(nn.GELU())
            in_c, out_c = out_c, out_c * 2
        self.stem = nn.Sequential(*stem_layers)

        self.trunk = nn.Sequential(*[PatchConvBlock(embed_dim, drop_path=drop_path) for _ in range(depth)])
        self.pool = AttentionPooling(embed_dim, mlp_ratio, drop_path=drop_path)

        nn.init.trunc_normal_(self.pool.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pool.attn.in_proj_weight, std=0.02)
        nn.init.trunc_normal_(self.pool.attn.out_proj.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        out = self.stem(x)
        out = self.trunk(out)
        out = out.flatten(-2).transpose(-1, -2)         # (N, C, H, W) -> (N, HW, C)
        out = self.pool(out)
        return out

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)


def PatchConvNet_S60(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-S60'], pretrained=pretrained, **kwargs)
def PatchConvNet_S120(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-S120'], pretrained=pretrained, **kwargs)
def PatchConvNet_B60(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-B60'], pretrained=pretrained, **kwargs)
def PatchConvNet_B120(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-B120'], pretrained=pretrained, **kwargs)
def PatchConvNet_L60(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-L60'], pretrained=pretrained, **kwargs)
def PatchConvNet_L120(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-L120'], pretrained=pretrained, **kwargs)
