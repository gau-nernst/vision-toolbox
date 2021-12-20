import torch
from torch import nn
from torchvision.models import resnet, mobilenet, efficientnet

from .base import BaseBackbone


class _TorchVisionBackbone(BaseBackbone):
    def forward_features(self, x):
        outputs = []
        out = x
        for s in self.stages:
            out = s(out)
            outputs.append(out)

        return outputs


class TorchVisionResNet(_TorchVisionBackbone):
    def __init__(self, name, pretrained=False):
        super().__init__()
        model = resnet.__dict__[name](pretrained=pretrained, progress=False)
        model.eval()
        self.stages = nn.ModuleList([
            nn.Sequential(model.conv1, model.bn1, model.relu),
            nn.Sequential(model.maxpool, model.layer1),
            model.layer2,
            model.layer3,
            model.layer4
        ])

        out_channels = [x.shape[1] for x in self.forward_features(torch.rand(1,3,224,224))]
        self.out_channels = tuple(out_channels)


class TorchVisionMobileNet(_TorchVisionBackbone):
    def __init__(self, name, pretrained=False, include_last_conv=True):
        super().__init__()
        model = mobilenet.__dict__[name](pretrained=pretrained, progress=False)
        model.eval()

        self.stages = nn.ModuleList()
        stage = [model.features[0]]
        for module in model.features[1:-1]:
            if module._is_cn:           # stride = 2, start of a new stage
                self.stages.append(nn.Sequential(*stage))
                stage = [module]
            else:
                stage.append(module)
        
        if include_last_conv:
            stage.append(model.features[-1])
        self.stages.append(nn.Sequential(*stage))

        out_channels = [x.shape[1] for x in self.forward_features(torch.rand(1,3,224,224))]
        self.out_channels = tuple(out_channels)


class TorchVisionEfficientNet(_TorchVisionBackbone):
    def __init__(self, name, pretrained=False, include_last_conv=True):
        super().__init__()
        model = efficientnet.__dict__[name](pretrained=pretrained, progress=False)
        model.eval()

        self.stages = nn.ModuleList()
        stage = []
        sample = torch.rand(1,3,224,224)
        size = 224
        for module in model.features:
            sample = module(sample)
            if sample.shape[-1] == size // 2 and stage:         # stride = 2, start of a new stage
                self.stages.append(nn.Sequential(*stage))                
                stage = [module]
                size = sample.shape[-1]
            else:
                stage.append(module)
        
        if not include_last_conv:
            stage.pop()
        self.stages.append(nn.Sequential(*stage))

        out_channels = [x.shape[1] for x in self.forward_features(torch.rand(1,3,224,224))]
        self.out_channels = tuple(out_channels)

def torchvision_backbone(name, **kwargs):
    if name in resnet.__all__:
        return TorchVisionResNet(name, **kwargs)

    if name in mobilenet.__all__:
        return TorchVisionMobileNet(name, **kwargs)

    if name in efficientnet.__all__:
        return TorchVisionEfficientNet(name, **kwargs)

    raise ValueError


def resnet18(**kwargs): return TorchVisionResNet("resnet18", **kwargs)
def resnet34(**kwargs): return TorchVisionResNet("resnet34", **kwargs)
def resnet50(**kwargs): return TorchVisionResNet("resnet50", **kwargs)
def resnet101(**kwargs): return TorchVisionResNet("resnet101", **kwargs)
def resnet152(**kwargs): return TorchVisionResNet("resnet152", **kwargs)
def resnext50_32x4d(**kwargs): return TorchVisionResNet("resnext50_32x4d", **kwargs)
def resnext101_32x8d(**kwargs): return TorchVisionResNet("resnext101_32x8d", **kwargs)
def wide_resnet50_2(**kwargs): return TorchVisionResNet("wide_resnet50_2", **kwargs)
def wide_resnet101_2(**kwargs): return TorchVisionResNet("wide_resnet101_2", **kwargs)

def mobilenet_v2(**kwargs): return TorchVisionMobileNet("mobilenet_v2", **kwargs)
def mobilenet_v3_large(**kwargs): return TorchVisionMobileNet("mobilenet_v3_large", **kwargs)
def mobilenet_v3_small(**kwargs): return TorchVisionMobileNet("mobilenet_v3_small", **kwargs)

def efficientnet_b0(**kwargs): return TorchVisionEfficientNet("efficientnet_b0", **kwargs)
def efficientnet_b1(**kwargs): return TorchVisionEfficientNet("efficientnet_b1", **kwargs)
def efficientnet_b2(**kwargs): return TorchVisionEfficientNet("efficientnet_b2", **kwargs)
def efficientnet_b3(**kwargs): return TorchVisionEfficientNet("efficientnet_b3", **kwargs)
def efficientnet_b4(**kwargs): return TorchVisionEfficientNet("efficientnet_b4", **kwargs)
def efficientnet_b5(**kwargs): return TorchVisionEfficientNet("efficientnet_b5", **kwargs)
def efficientnet_b6(**kwargs): return TorchVisionEfficientNet("efficientnet_b6", **kwargs)
def efficientnet_b7(**kwargs): return TorchVisionEfficientNet("efficientnet_b7", **kwargs)
