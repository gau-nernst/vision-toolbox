import pytest
import torch
from torch import Tensor, nn

from vision_toolbox import backbones


@pytest.fixture
def inputs():
    return torch.rand(1, 3, 224, 224)


vovnet_v1_models = [f"vovnet{x}" for x in ["27_slim", 39, 57]]
vovnet_v2_models = [f"vovnet{x}_ese" for x in ["19_slim", 19, 39, 57, 99]]
darknet_models = ["darknet19", "darknet53", "cspdarknet53"]
darknet_yolov5_models = [f"darknet_yolov5{x}" for x in ("n", "s", "m", "l", "x")]
torchvision_models = ["resnet18", "mobilenet_v2", "efficientnet_b0", "regnet_x_400mf"]

all_models = vovnet_v1_models + vovnet_v2_models + darknet_models + darknet_yolov5_models + torchvision_models


@pytest.mark.parametrize("name", all_models)
class TestBackbone:
    def test_model_creation(self, name: str):
        assert hasattr(backbones, name)
        m = getattr(backbones, name)()
        assert isinstance(m, nn.Module)
        assert isinstance(m, backbones.BaseBackbone)

    def test_pretrained_weights(self, name: str):
        getattr(backbones, name)(pretrained=True)

    def test_attributes(self, name: str):
        m = getattr(backbones, name)()

        assert hasattr(m, "out_channels_list")
        assert isinstance(m.out_channels_list, tuple)
        for c in m.out_channels_list:
            assert isinstance(c, int)

        assert hasattr(m, "stride")
        assert isinstance(m.stride, int)

        assert hasattr(m, "get_feature_maps")
        assert callable(m.get_feature_maps)

    def test_forward(self, name: str, inputs: Tensor):
        m = getattr(backbones, name)()
        outputs = m(inputs)

        assert isinstance(outputs, Tensor)
        assert len(outputs.shape) == 4

    def test_get_feature_maps(self, name: str, inputs: Tensor):
        m = getattr(backbones, name)()
        outputs = m.get_feature_maps(inputs)

        assert isinstance(outputs, list)
        assert len(outputs) == len(m.out_channels_list)
        for out, out_c in zip(outputs, m.out_channels_list):
            assert isinstance(out, Tensor)
            assert len(out.shape) == 4
            assert out.shape[1] == out_c

    def test_jit_trace(self, name: str, inputs: Tensor):
        m = getattr(backbones, name)()
        torch.jit.trace(m, inputs)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile() is not available")
@pytest.mark.parametrize("name", ["vovnet39", "vovnet19_ese", "darknet19", "cspdarknet53", "darknet_yolov5n"])
def test_compile(name: str, inputs: Tensor):
    m = getattr(backbones, name)()
    m_compiled = torch.compile(m)
    m_compiled(inputs)
