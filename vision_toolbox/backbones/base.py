from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable

import torch
from torch import Tensor, nn


_norm = Callable[[int], nn.Module]
_act = Callable[[], nn.Module]


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    # subclass only needs to implement this method
    @abstractmethod
    def get_feature_maps(self, x: Tensor) -> list[Tensor]:
        pass

    def forward(self, x: Tensor) -> Tensor:
        return self.get_feature_maps(x)[-1]

    def _load_state_dict_from_url(self, url: str) -> None:
        state_dict = torch.hub.load_state_dict_from_url(url)
        self.load_state_dict(state_dict)
