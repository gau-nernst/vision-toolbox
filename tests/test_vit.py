import timm
import torch

from vision_toolbox.backbones import ViT


def test_resize_pe():
    m = ViT.from_config("Ti", 16, 224)
    m(torch.randn(1, 3, 224, 224))
    m.resize_pe(256)
    m(torch.randn(1, 3, 256, 256))


def test_from_pretrained():
    m = ViT.from_config("Ti", 16, 224, True).eval()
    x = torch.randn(1, 3, 224, 224)
    out = m(x)

    m_timm = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=0).eval()
    out_timm = m_timm(x)

    torch.testing.assert_close(out, out_timm)
