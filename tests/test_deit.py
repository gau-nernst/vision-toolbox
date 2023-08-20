import pytest
import timm
import torch

from vision_toolbox.backbones import DeiT


@pytest.mark.parametrize("cls", (DeiT,))
def test_forward(cls):
    m = cls.from_config("Ti_16", 224)
    m(torch.randn(1, 3, 224, 224))


@pytest.mark.parametrize("cls", (DeiT,))
def test_resize_pe(cls):
    m = cls.from_config("Ti_16", 224)
    m(torch.randn(1, 3, 224, 224))
    m.resize_pe(256)
    m(torch.randn(1, 3, 256, 256))


@pytest.mark.parametrize(
    "cls,variant,timm_name",
    (
        (DeiT, "Ti_16", "deit_tiny_distilled_patch16_224.fb_in1k"),
        # ("deit3_S_16", "deit3_small_patch16_224.fb_in22k_ft_in1k"),
    ),
)
def test_from_pretrained(cls, variant, timm_name):
    m = cls.from_config(variant, 224, True, True).eval()
    x = torch.randn(1, 3, 224, 224)
    out = m(x)

    m_timm = timm.create_model(timm_name, pretrained=True, num_classes=0).eval()
    out_timm = m_timm(x)

    torch.testing.assert_close(out, out_timm, rtol=2e-5, atol=2e-5)
