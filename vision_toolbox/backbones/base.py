from typing import Tuple, Dict, List
from abc import ABCMeta, abstractmethod
import warnings

import torch
from torch import nn
from torch.hub import load_state_dict_from_url


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass

    def forward(self, x):
        return self.forward_features(x)[-1]

    def get_out_channels(self) -> Tuple[int]:
        return self.out_channels

    def get_stride(self) -> int:
        return self.stride

    @classmethod
    def from_config(cls, config: Dict, pretrained: bool=False, **kwargs):
        weights = config.pop("weights", None)
        model = cls(**config, **kwargs)
        if pretrained:
            if weights is not None:
                state_dict = load_state_dict_from_url(weights)
                model.load_state_dict(state_dict)
            else:
                warnings.warn('No pre-trained weights are available. Skip loading pre-trained weights')

        return model
