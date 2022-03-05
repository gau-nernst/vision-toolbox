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
    'patchconvnet_s60', 'patchconvnet_s120',
    'patchconvnet_b60', 'patchconvnet_b120',
    'patchconvnet_l60', 'patchconvnet_l120'
]


_base = {
    'mlp_ratio': 3,
    'drop_path': 0.3,
    'layer_scale_init': 1e-6
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


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class PatchConvBlock(nn.Module):
    def __init__(self, embed_dim, drop_path=0.3, layer_scale_init=1e-6, norm_type='bn'):
        assert norm_type in ('bn', 'ln')
        super().__init__()
        if norm_type == 'ln':
            # LayerNorm version. Primary format is (N, H, W, C)
            # follow this approach https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
            self.layers = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                Permute(0, 3, 1, 2),        # (N, H, W, C) -> (N, C, H, W)
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),    # dw-conv
                nn.GELU(),
                SqueezeExcitation(embed_dim, embed_dim // 4),
                Permute(0, 2, 3, 1),        # (N, C, H, W) -> (N, H, W, C)
                nn.Linear(embed_dim, embed_dim)
            )
            self.layer_scale = nn.Parameter(torch.ones(embed_dim) * layer_scale_init)

        else:
            # BatchNorm version. Primary format is (N, C, H, W)
            self.layers = nn.Sequential(
                nn.BatchNorm2d(embed_dim),
                nn.Conv2d(embed_dim, embed_dim, 1),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
                nn.GELU(),
                SqueezeExcitation(embed_dim, embed_dim // 4),
                nn.Conv2d(embed_dim, embed_dim, 1)
            )
            self.layer_scale = nn.Parameter(torch.ones(embed_dim, 1, 1) * layer_scale_init)
        
        self.drop_path = StochasticDepth(drop_path, 'row') if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        return x + self.drop_path(self.layers(x) * self.layer_scale)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, drop_path=0.3, layer_scale_init=1e-6):
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
        out = torch.cat((cls_token, x), dim=1)

        # attention pooling. q = cls_token. k = v = (cls_token, x)
        out = self.norm_1(out)
        out = self.attn(out[:,:1], out, out, need_weights=False)[0]
        cls_token = cls_token + self.drop_path(out * self.layer_scale_1)    # residual + layer scale + dropout

        # mlp
        out = self.norm_2(cls_token)
        out = self.mlp(out)
        cls_token = cls_token + self.drop_path(out * self.layer_scale_2)
        
        out = self.norm_3(cls_token).squeeze(1)     # (N, 1, C) -> (N, C)
        return out


class PatchConvNet(BaseBackbone):
    def __init__(self, embed_dim, depth, mlp_ratio, drop_path, layer_scale_init, norm_type='bn'):
        assert norm_type in ('bn', 'ln')
        super().__init__()
        self.norm_type = norm_type
        self.out_channels = (embed_dim,)
        self.stem = nn.Sequential(
            nn.Conv2d(3, embed_dim // 8, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim//8, embed_dim // 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1),
        )

        kwargs = dict(drop_path=drop_path, layer_scale_init=layer_scale_init)
        self.trunk = nn.Sequential(*[PatchConvBlock(embed_dim, norm_type=norm_type, **kwargs) for _ in range(depth)])
        self.pool = AttentionPooling(embed_dim, mlp_ratio, **kwargs)

        # weight initialization
        nn.init.trunc_normal_(self.pool.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pool.attn.in_proj_weight, std=0.02)
        nn.init.trunc_normal_(self.pool.attn.out_proj.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor):
        out = self.stem(x)

        if self.norm_type == 'ln':
            # layer norm
            out = torch.permute(out, (0, 2, 3, 1))      # (N, C, H, W) -> (N, H, W, C)
            out = self.trunk(out)
            out = torch.flatten(out, 1, 2)              # (N, H, W, C) -> (N, HW, C)
        else:
            # batch norm
            out = self.trunk(out)
            out = out.flatten(2).transpose(1, 2)        # (N, C, H, W) -> (N, HW, C)
        
        out = self.pool(out)
        return out

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)


def patchconvnet_s60(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-S60'], pretrained=pretrained, **kwargs)
def patchconvnet_s120(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-S120'], pretrained=pretrained, **kwargs)
def patchconvnet_b60(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-B60'], pretrained=pretrained, **kwargs)
def patchconvnet_b120(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-B120'], pretrained=pretrained, **kwargs)
def patchconvnet_l60(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-L60'], pretrained=pretrained, **kwargs)
def patchconvnet_l120(pretrained=False, **kwargs): return PatchConvNet.from_config(configs['PatchConvNet-L120'], pretrained=pretrained, **kwargs)
