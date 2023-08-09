import torch

from vision_toolbox.backbones import SwinTransformer
from vision_toolbox.backbones.swin import window_partition, window_unpartition


def test_window_partition():
    img = torch.randn(1, 224, 280, 3)
    windows, nH, nW = window_partition(img, 7)
    _img = window_unpartition(windows, 7, nH, nW)
    torch.testing.assert_close(img, _img)


def test_forward():
    m = SwinTransformer.from_config("T", 224)
    m(torch.randn(1, 3, 224, 224))


# def test_from_pretrained():
#     m = ViT.from_config("Ti", 16, 224, True).eval()
#     x = torch.randn(1, 3, 224, 224)
#     out = m(x)

#     m_timm = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=0).eval()
#     out_timm = m_timm(x)

#     torch.testing.assert_close(out, out_timm, rtol=2e-5, atol=2e-5)
