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

    @abstractmethod
    def get_out_channels(self) -> Tuple[int]:
        pass
