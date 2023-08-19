import pytest
import timm
import torch

from vision_toolbox.backbones import ConvNeXt


@pytest.mark.parametrize("v2", [False, True])
def test_forward(v2):
    m = ConvNeXt.from_config("T", v2)
    m(torch.randn(1, 3, 224, 224))


@pytest.mark.parametrize("v2", [False, True])
def test_from_pretrained(v2):
    m = ConvNeXt.from_config("T", v2, True).eval()
    x = torch.randn(1, 3, 224, 224)
    out = m(x)

    model_name = "convnextv2_tiny.fcmae" if v2 else "convnext_tiny.fb_in22k"
    m_timm = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
    out_timm = m_timm(x)

    torch.testing.assert_close(out, out_timm)
