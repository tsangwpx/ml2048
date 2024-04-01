from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from torch import nn

from ml2048.stats import TensorStats


class Policy(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def sample_actions(
        self,
        state: torch.LongTensor | torch.ByteTensor,
        valid_actions: torch.BoolTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        raise NotImplementedError


class ProbabilisticPolicy(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def action_logits(
        self,
        state: torch.LongTensor | torch.ByteTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class SupportsEvalValue(metaclass=ABCMeta):
    @abstractmethod
    def eval_value(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class TrainablePolicy(Policy, metaclass=ABCMeta):
    def feed_episodes(
        self,
        episodes: list[dict[str, Any]],
        params: dict[str, Any],
        batch_size: int,
        *,
        tensor_stats: TensorStats,
        seed: int | None = None,
        device: Any = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
