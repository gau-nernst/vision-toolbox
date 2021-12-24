from typing import Tuple
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

class BaseBackbone(nn.Module, metaclass=ABCMeta):
    def forward(self, x):
        return self.forward_features(x)[-1]

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        pass

    def get_out_channels(self) -> Tuple[int]:
        return self.out_channels

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))
        return self
