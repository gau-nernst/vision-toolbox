from typing import Tuple, Dict, List
from abc import ABCMeta, abstractmethod
import warnings

import torch
from torch import nn
from torch.hub import load_state_dict_from_url


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    # metadata
    # subclass must have attributes out_channels and stride
    def get_out_channels(self) -> Tuple[int]:
        return self.out_channels

    def get_n_last_out_channels(self, n: int) -> Tuple[int]:
        return self.out_channels[-n:]

    def get_last_out_channels(self) -> int:
        return self.out_channels[-1]

    def get_stride(self) -> int:
        return self.stride

    # subclass only needs to implement this method
    @abstractmethod
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass

    # feature pyramid
    def get_n_last_feature_maps(self, x: torch.Tensor, n: int) -> List[torch.Tensor]:
        return self.get_feature_maps(x)[-n:]

    def get_last_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_feature_maps(x)[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_last_feature_map(x)

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
