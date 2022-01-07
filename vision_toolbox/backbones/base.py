from typing import Tuple, Dict
from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.hub import load_state_dict_from_url


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    def forward(self, x):
        return self.forward_features(x)[-1]

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        pass

    def get_out_channels(self) -> Tuple[int]:
        return self.out_channels

    @classmethod
    def from_config(cls, config: Dict, pretrained: bool=False):
        weights = config.pop("weights", None)
        model = cls(**config)
        if pretrained and weights is not None:
            model.load_state_dict(load_state_dict_from_url(weights))
        return model
