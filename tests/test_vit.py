import torch

from vision_toolbox.backbones import ViT
from vision_toolbox.utils import torch_hub_download


def test_resize_pe():
    m = ViT.from_config("Ti", 16, 224)
    m(torch.randn(1, 3, 224, 224))
    m.resize_pe(256)
    m(torch.randn(1, 3, 256, 256))


def test_from_jax():
    url = (
        "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz"
    )
    m = ViT.from_jax_weights(torch_hub_download(url))
    m(torch.randn(1, 3, 224, 224))
