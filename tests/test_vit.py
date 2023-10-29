import pytest
import timm
import torch

from vision_toolbox.backbones import ViT


def test_forward():
    m = ViT.from_config("Ti_16", 224)
    m(torch.randn(1, 3, 224, 224))


def test_resize_pe():
    m = ViT.from_config("Ti_16", 224)
    m(torch.randn(1, 3, 224, 224))
    m.resize_pe(256)
    m(torch.randn(1, 3, 256, 256))


@pytest.mark.parametrize(
    "config,timm_name",
    [
        (dict(variant="Ti_16", img_size=224, weights="augreg"), "vit_tiny_patch16_224.augreg_in21k"),
        (dict(variant="B_16", img_size=224, weights="siglip"), "vit_base_patch16_siglip_224"),
    ],
)
def test_from_pretrained(config, timm_name):
    m = ViT.from_config(**config).eval()
    x = torch.randn(1, 3, 224, 224)
    out = m(x)

    m_timm = timm.create_model(timm_name, pretrained=True, num_classes=0).eval()
    out_timm = m_timm(x)

    torch.testing.assert_close(out, out_timm, rtol=2e-5, atol=2e-5)
