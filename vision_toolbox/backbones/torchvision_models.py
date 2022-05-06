import warnings
from typing import List

import torch
from torch import nn
from torchvision.models import mobilenet, resnet

try:
    from torchvision.models import efficientnet, regnet
    from torchvision.models.feature_extraction import create_feature_extractor
except ImportError:
    warnings.warn("torchvision < 0.11.0. torchvision models won't be available")
    regnet = efficientnet = create_feature_extractor = None

from .base import BaseBackbone

__all__ = [
    "ResNetExtractor",
    "RegNetExtractor",
    "MobileNetExtractor",
    "EfficientNetExtractor",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_1_6gf",
    "regnet_x_3_2gf",
    "regnet_x_8gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
    "regnet_y_3_2gf",
    "regnet_y_8gf",
    "regnet_y_16gf",
    "regnet_y_32gf",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]


class _ExtractorBackbone(BaseBackbone):
    def __init__(self, backbone: nn.Module, node_names: List[str]):
        super().__init__()
        self.feat_extractor = create_feature_extractor(backbone, node_names)
        with torch.no_grad():
            self.out_channels_list = tuple(
                x.shape[1]
                for x in self.feat_extractor(torch.rand(1, 3, 224, 224)).values()
            )
        self.stride = 32

    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
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
        stage_indices = [
            i for i, b in enumerate(backbone.features) if getattr(b, "_is_cn", False)
        ]
        node_names = [f"features.{i}.{block_name}.0" for i in stage_indices] + [
            f"features.{len(backbone.features)-1}"
        ]
        super().__init__(backbone, node_names)


class EfficientNetExtractor(_ExtractorBackbone):
    def __init__(self, name: str, pretrained: bool = False):
        backbone = getattr(efficientnet, name)(pretrained=pretrained, progress=False)

        # take output at expansion 1x1 conv
        stage_indices = [2, 3, 4, 6]
        node_names = [f"features.{i}.0.block.0" for i in stage_indices] + [
            f"features.{len(backbone.features)-1}"
        ]
        super().__init__(backbone, node_names)


def resnet18(pretrained=False, **kwargs):
    return ResNetExtractor("resnet18", pretrained=pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    return ResNetExtractor("resnet34", pretrained=pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
    return ResNetExtractor("resnet50", pretrained=pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    return ResNetExtractor("resnet101", pretrained=pretrained, **kwargs)


def resnet152(pretrained=False, **kwargs):
    return ResNetExtractor("resnet152", pretrained=pretrained, **kwargs)


def resnext50_32x4d(pretrained=False, **kwargs):
    return ResNetExtractor("resnext50_32x4d", pretrained=pretrained, **kwargs)


def resnext101_32x8d(pretrained=False, **kwargs):
    return ResNetExtractor("resnext101_32x8d", pretrained=pretrained, **kwargs)


def wide_resnet50_2(pretrained=False, **kwargs):
    return ResNetExtractor("wide_resnet50_2", pretrained=pretrained, **kwargs)


def wide_resnet101_2(pretrained=False, **kwargs):
    return ResNetExtractor("wide_resnet101_2", pretrained=pretrained, **kwargs)


def mobilenet_v2(pretrained=False, **kwargs):
    return MobileNetExtractor("mobilenet_v2", pretrained=pretrained, **kwargs)


def mobilenet_v3_large(pretrained=False, **kwargs):
    return MobileNetExtractor("mobilenet_v3_large", pretrained=pretrained, **kwargs)


def mobilenet_v3_small(pretrained=False, **kwargs):
    return MobileNetExtractor("mobilenet_v3_small", pretrained=pretrained, **kwargs)


def efficientnet_b0(pretrained=False, **kwargs):
    return EfficientNetExtractor("efficientnet_b0", pretrained=pretrained, **kwargs)


def efficientnet_b1(pretrained=False, **kwargs):
    return EfficientNetExtractor("efficientnet_b1", pretrained=pretrained, **kwargs)


def efficientnet_b2(pretrained=False, **kwargs):
    return EfficientNetExtractor("efficientnet_b2", pretrained=pretrained, **kwargs)


def efficientnet_b3(pretrained=False, **kwargs):
    return EfficientNetExtractor("efficientnet_b3", pretrained=pretrained, **kwargs)


def efficientnet_b4(pretrained=False, **kwargs):
    return EfficientNetExtractor("efficientnet_b4", pretrained=pretrained, **kwargs)


def efficientnet_b5(pretrained=False, **kwargs):
    return EfficientNetExtractor("efficientnet_b5", pretrained=pretrained, **kwargs)


def efficientnet_b6(pretrained=False, **kwargs):
    return EfficientNetExtractor("efficientnet_b6", pretrained=pretrained, **kwargs)


def efficientnet_b7(pretrained=False, **kwargs):
    return EfficientNetExtractor("efficientnet_b7", pretrained=pretrained, **kwargs)


def regnet_x_400mf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_x_400mf", pretrained=pretrained, **kwargs)


def regnet_x_800mf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_x_800mf", pretrained=pretrained, **kwargs)


def regnet_x_1_6gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_x_1_6gf", pretrained=pretrained, **kwargs)


def regnet_x_3_2gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_x_3_2gf", pretrained=pretrained, **kwargs)


def regnet_x_8gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_x_8gf", pretrained=pretrained, **kwargs)


def regnet_x_16gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_x_16gf", pretrained=pretrained, **kwargs)


def regnet_x_32gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_x_32gf", pretrained=pretrained, **kwargs)


def regnet_y_400mf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_y_400mf", pretrained=pretrained, **kwargs)


def regnet_y_800mf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_y_800mf", pretrained=pretrained, **kwargs)


def regnet_y_1_6gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_y_1_6gf", pretrained=pretrained, **kwargs)


def regnet_y_3_2gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_y_3_2gf", pretrained=pretrained, **kwargs)


def regnet_y_8gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_y_8gf", pretrained=pretrained, **kwargs)


def regnet_y_16gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_y_16gf", pretrained=pretrained, **kwargs)


def regnet_y_32gf(pretrained=False, **kwargs):
    return RegNetExtractor("regnet_y_32gf", pretrained=pretrained, **kwargs)
