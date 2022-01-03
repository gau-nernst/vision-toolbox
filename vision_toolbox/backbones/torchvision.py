import torch
from torchvision.models import resnet, mobilenet, efficientnet
from torchvision.models.feature_extraction import create_feature_extractor

from .base import BaseBackbone


__all__ = [
    "ResNetExtractor", "MobileNetExtractor", "EfficientNetExtractor",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2",
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"
]


class _ExtractorBackbone(BaseBackbone):
    def __init__(self, backbone, node_names):
        super().__init__()
        self.feat_extractor = create_feature_extractor(backbone, node_names)

        self.feat_extractor.eval()
        with torch.no_grad():
            out_channels = [x.shape[1] for x in self.feat_extractor(torch.rand(1,3,224,224)).values()]
        self.out_channels = tuple(out_channels)

    def forward_features(self, x):
        return list(self.feat_extractor(x).values())


class ResNetExtractor(_ExtractorBackbone):
    def __init__(self, name, pretrained=False):
        backbone = resnet.__dict__[name](pretrained=pretrained, progress=False)
        node_names = ["relu", "layer1", "layer2", "layer3", "layer4"]
        super().__init__(backbone, node_names)


class MobileNetExtractor(_ExtractorBackbone):
    def __init__(self, name, pretrained=False):
        backbone = mobilenet.__dict__[name](pretrained=pretrained, progress=False)
        
        # take output at expansion 1x1 conv
        stage_indices = [i for i, b in enumerate(backbone.features) if getattr(b, "_is_cn", False)]
        block_name = "conv" if name == "mobilenet_v2" else "block"
        node_names = [f"features.{i}.{block_name}.0" for i in stage_indices] + [f"features.{len(backbone.features)-1}"]

        super().__init__(backbone, node_names)


class EfficientNetExtractor(_ExtractorBackbone):
    def __init__(self, name, pretrained=False):
        backbone = efficientnet.__dict__[name](pretrained=pretrained, progress=False)

        # take output at expansion 1x1 conv
        stage_indices = [2, 3, 4, 6]
        node_names = [f"features.{i}.0.block.0" for i in stage_indices] + [f"features.{len(backbone.features)-1}"]

        super().__init__(backbone, node_names)

def resnet18(**kwargs): return ResNetExtractor("resnet18", **kwargs)
def resnet34(**kwargs): return ResNetExtractor("resnet34", **kwargs)
def resnet50(**kwargs): return ResNetExtractor("resnet50", **kwargs)
def resnet101(**kwargs): return ResNetExtractor("resnet101", **kwargs)
def resnet152(**kwargs): return ResNetExtractor("resnet152", **kwargs)
def resnext50_32x4d(**kwargs): return ResNetExtractor("resnext50_32x4d", **kwargs)
def resnext101_32x8d(**kwargs): return ResNetExtractor("resnext101_32x8d", **kwargs)
def wide_resnet50_2(**kwargs): return ResNetExtractor("wide_resnet50_2", **kwargs)
def wide_resnet101_2(**kwargs): return ResNetExtractor("wide_resnet101_2", **kwargs)

def mobilenet_v2(**kwargs): return MobileNetExtractor("mobilenet_v2", **kwargs)
def mobilenet_v3_large(**kwargs): return MobileNetExtractor("mobilenet_v3_large", **kwargs)
def mobilenet_v3_small(**kwargs): return MobileNetExtractor("mobilenet_v3_small", **kwargs)

def efficientnet_b0(**kwargs): return EfficientNetExtractor("efficientnet_b0", **kwargs)
def efficientnet_b1(**kwargs): return EfficientNetExtractor("efficientnet_b1", **kwargs)
def efficientnet_b2(**kwargs): return EfficientNetExtractor("efficientnet_b2", **kwargs)
def efficientnet_b3(**kwargs): return EfficientNetExtractor("efficientnet_b3", **kwargs)
def efficientnet_b4(**kwargs): return EfficientNetExtractor("efficientnet_b4", **kwargs)
def efficientnet_b5(**kwargs): return EfficientNetExtractor("efficientnet_b5", **kwargs)
def efficientnet_b6(**kwargs): return EfficientNetExtractor("efficientnet_b6", **kwargs)
def efficientnet_b7(**kwargs): return EfficientNetExtractor("efficientnet_b7", **kwargs)
