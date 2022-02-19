import torch
from torch import nn
from torchvision.ops.misc import SqueezeExcitation

from .base import BaseBackbone


_S_width = 384
configs = {
    "S60": {
        "width": _S_width,
        "depth": 60,
    },
    "S120": {
        "width": _S_width,
        "depth": 120
    }
}


class PatchConvBlock(nn.Module):
    def __init__(self, width, norm_layer=nn.LayerNorm, act_fn=nn.GELU, layer_scale_init=1e-6):
        super().__init__()
        self.layers = nn.Sequential(
            norm_layer(width),
            nn.Conv2d(width, width, 1),
            act_fn(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            act_fn(),
            SqueezeExcitation(width, width // 4),
            nn.Conv2d(width, width, 1)
        )
        self.layer_scale = nn.Parameter(torch.full(width, layer_scale_init))

    def forward(self, x):
        return x + self.layers(x) * self.layer_scale


class PatchConvNet(BaseBackbone):
    def __init__(self, width, depth, act_fn=nn.GELU):
        super().__init__()
        stem_layers = []
        in_c, out_c = 3, width // 8
        for _ in range(4):
            stem_layers.append(nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False))    # bias=False even though there is no norm layer
            stem_layers.append(act_fn)
            in_c, out_c = out_c, out_c * 2
        self.stem = nn.Sequential(*stem_layers)

        self.blocks = nn.Sequential(*[PatchConvBlock(width) for _ in range(depth)])

        self.cls_token = nn.Parameter(torch.zeros(width))
        self.attn = nn.MultiheadAttention(width, 1)
        

    def forward(self, x):
        out = self.stem(x)
        out = self.blocks(x)


def S60():
    pass
