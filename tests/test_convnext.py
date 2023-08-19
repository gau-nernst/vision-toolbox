import timm
import torch

from vision_toolbox.backbones import ConvNeXt


def test_forward():
    m = ConvNeXt.from_config("T")
    m(torch.randn(1, 3, 224, 224))


def test_from_pretrained():
    m = ConvNeXt.from_config("T", True).eval()
    x = torch.randn(1, 3, 224, 224)
    out = m(x)

    m_timm = timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=0).eval()
    out_timm = m_timm(x)

    torch.testing.assert_close(out, out_timm)
