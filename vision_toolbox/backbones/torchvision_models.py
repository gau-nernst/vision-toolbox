from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import efficientnet, mobilenet, regnet, resnet
from torchvision.models.feature_extraction import create_feature_extractor

from .base import BaseBackbone


class _ExtractorBackbone(BaseBackbone):
    def __init__(self, backbone: nn.Module, node_names: list[str]):
        super().__init__()
        self.feat_extractor = create_feature_extractor(backbone, node_names)
        with torch.no_grad():
            self.out_channels_list = tuple(x.shape[1] for x in self.feat_extractor(torch.rand(1, 3, 224, 224)).values())
        self.stride = 32

    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        return list(self.feat_extractor(x).values())


class ResNetExtractor(_ExtractorBackbone):
    def __init__(self, name: str, pretrained: bool = False):
        backbone = getattr(resnet, name)(pretrained=pretrained, progress=False)
        node_names = ["relu"] + [f"layer{i+1}" for i in range(4)]
        super().__init__(backbone, node_names)


class RegNetExtractor(_ExtractorBackbone):
    def __init__(self, name: str, pretrained: bool = False):
        backbone = getattr(regnet, name)(pretrained=pretrained)
        node_names = ["stem"] + [f"trunk_output.block{i+1}" for i in range(4)]
        super().__init__(backbone, node_names)


class MobileNetExtractor(_ExtractorBackbone):
    def __init__(self, name: str, pretrained: bool = False):
        backbone = getattr(mobilenet, name)(pretrained=pretrained, progress=False)
        block_name = "conv" if name == "mobilenet_v2" else "block"

        # take output at expansion 1x1 conv
        stage_indices = [i for i, b in enumerate(backbone.features) if getattr(b, "_is_cn", False)]
        node_names = [f"features.{i}.{block_name}.0" for i in stage_indices] + [f"features.{len(backbone.features)-1}"]
        super().__init__(backbone, node_names)


class EfficientNetExtractor(_ExtractorBackbone):
    def __init__(self, name: str, pretrained: bool = False):
        backbone = getattr(efficientnet, name)(pretrained=pretrained, progress=False)

        # take output at expansion 1x1 conv
        stage_indices = [2, 3, 4, 6]
        node_names = [f"features.{i}.0.block.0" for i in stage_indices] + [f"features.{len(backbone.features)-1}"]
        super().__init__(backbone, node_names)
