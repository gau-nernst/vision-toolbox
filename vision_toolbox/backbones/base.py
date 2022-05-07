from copy import deepcopy
from typing import Dict, List
from abc import ABCMeta, abstractmethod
import warnings

import torch
from torch import nn
from torch.hub import load_state_dict_from_url


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    # subclass only needs to implement this method
    @abstractmethod
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_feature_maps(x)[-1]

    @classmethod
    def from_config(cls, config: Dict, pretrained: bool = False, **kwargs):
        config = deepcopy(config)
        weights = config.pop("weights", None)
        model = cls(**config, **kwargs)
        if pretrained:
            if weights is not None:
                state_dict = load_state_dict_from_url(weights)
                model.load_state_dict(state_dict)
            else:
                msg = "No pre-trained weights are available. Skip loading pre-trained weights"
                warnings.warn(msg)
        return model
