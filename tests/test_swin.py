import timm
import torch

from vision_toolbox.backbones import SwinTransformer


def test_forward():
    m = SwinTransformer.from_config("T", 224)
    m(torch.randn(1, 3, 224, 224))


def test_from_pretrained():
    m = SwinTransformer.from_config("T", 224, True).eval()
    x = torch.randn(1, 3, 224, 224)
    out = m(x)

    m_timm = timm.create_model("swin_tiny_patch4_window7_224.ms_in22k", pretrained=True, num_classes=0).eval()
    out_timm = m_timm(x)

    torch.testing.assert_close(out, out_timm, rtol=2e-5, atol=2e-5)
