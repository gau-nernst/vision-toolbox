import timm
import torch

from vision_toolbox.backbones import CaiT


def test_forward():
    m = CaiT.from_config("xxs_24", 224)
    m(torch.randn(1, 3, 224, 224))


def test_from_pretrained():
    m = CaiT.from_config("xxs_24", 224, True).eval()
    x = torch.randn(1, 3, 224, 224)
    out = m(x)

    m_timm = timm.create_model("cait_xxs24_224.fb_dist_in1k", pretrained=True, num_classes=0).eval()
    out_timm = m_timm(x)

    torch.testing.assert_close(out, out_timm, rtol=2e-5, atol=2e-5)
