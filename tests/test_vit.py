import torch

from vision_toolbox.backbones import ViT


def test_resize_pe():
    m = ViT.from_config("Ti", 16, 224)
    m(torch.randn(1, 3, 224, 224))
    m.resize_pe(256)
    m(torch.randn(1, 3, 256, 256))
