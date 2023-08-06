import timm
import torch

from vision_toolbox.backbones import MLPMixer


def test_forward():
    m = MLPMixer.from_config("Ti", 16, 224)
    m(torch.randn(1, 3, 224, 224))


# def test_from_pretrained():
#     m = MLPMixer.from_config("Ti", 16, 224, True).eval()
#     x = torch.randn(1, 3, 224, 224)
#     out = m(x)

#     m_timm = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=0).eval()
#     out_timm = m_timm(x)

#     torch.testing.assert_close(out, out_timm, rtol=2e-5, atol=2e-5)
