from functools import partial

import pytest
import torch
from torch import Tensor

from vision_toolbox.backbones import (
    Darknet,
    DarknetYOLOv5,
    EfficientNetExtractor,
    MobileNetExtractor,
    RegNetExtractor,
    ResNetExtractor,
    VoVNet,
)


@pytest.fixture
def inputs():
    return torch.rand(1, 3, 224, 224)


factory_list = [
    *[partial(Darknet.from_config, x) for x in ("darknet19", "cspdarknet53")],
    *[partial(DarknetYOLOv5.from_config, x) for x in ("n", "l")],
    *[
        partial(VoVNet.from_config, x, y, z)
        for x, y, z in ((27, True, False), (39, False, False), (19, True, True), (57, False, True))
    ],
    partial(ResNetExtractor, "resnet18"),
    partial(RegNetExtractor, "regnet_x_400mf"),
    partial(MobileNetExtractor, "mobilenet_v2"),
    partial(EfficientNetExtractor, "efficientnet_b0"),
]


@pytest.mark.parametrize("factory", factory_list)
class TestBackbone:
    def test_attributes(self, factory):
        m = factory()

        assert hasattr(m, "out_channels_list")
        assert isinstance(m.out_channels_list, tuple)
        for c in m.out_channels_list:
            assert isinstance(c, int)

        assert hasattr(m, "stride")
        assert isinstance(m.stride, int)

        assert hasattr(m, "get_feature_maps")
        assert callable(m.get_feature_maps)

    def test_forward(self, factory, inputs):
        m = factory()
        outputs = m(inputs)

        assert isinstance(outputs, Tensor)
        assert len(outputs.shape) == 4

    def test_get_feature_maps(self, factory, inputs):
        m = factory()
        outputs = m.get_feature_maps(inputs)

        assert isinstance(outputs, list)
        assert len(outputs) == len(m.out_channels_list)
        for out, out_c in zip(outputs, m.out_channels_list):
            assert isinstance(out, Tensor)
            assert len(out.shape) == 4
            assert out.shape[1] == out_c

    def test_pretrained(self, factory):
        factory(pretrained=True)

    def test_jit_trace(self, factory, inputs):
        m = factory()
        torch.jit.trace(m, inputs)


# @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile() is not available")
# @pytest.mark.parametrize("name", ["vovnet39", "vovnet19_ese", "darknet19", "cspdarknet53", "darknet_yolov5n"])
# def test_compile(name: str, inputs: Tensor):
#     m = getattr(backbones, name)()
#     m_compiled = torch.compile(m)
#     m_compiled(inputs)
