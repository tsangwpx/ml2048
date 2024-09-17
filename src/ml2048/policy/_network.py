import math

import torch
from torch import nn
from torch.nn import functional as F

NUM_CELLS = 16  # 4x4 board
NUM_CLASSES = 16  # EMPTY, 2, 4, 8, ..., up to 32768.
NUM_ACTIONS = 4


class CNNEncoder(nn.Module):
    def __init__(
        self,
        out_features: int,
        multiplier: int = 16,
    ) -> None:
        super().__init__()

        assert out_features >= 1 and out_features % 16 == 0, out_features
        assert multiplier >= 1, multiplier

        out_channels = out_features // 16

        self.out_features = out_features
        self._out_channels = out_channels

        self._depthwise_full = nn.Conv1d(
            NUM_CLASSES,
            NUM_CLASSES * multiplier,
            NUM_CLASSES,
            groups=NUM_CLASSES,
        )
        self._pointwise_full = nn.Conv1d(
            self._depthwise_full.out_channels,
            out_channels * 4,
            1,
        )

        self._depthwise_hori = nn.Conv2d(
            NUM_CLASSES,
            NUM_CLASSES * multiplier,
            (1, 4),
            groups=NUM_CLASSES,
        )
        self._pointwise_hori = nn.Conv2d(
            self._depthwise_hori.out_channels,
            out_channels,
            1,
        )

        self._depthwise_vert = nn.Conv2d(
            NUM_CLASSES,
            NUM_CLASSES * multiplier,
            (4, 1),
            groups=NUM_CLASSES,
        )
        self._pointwise_vert = nn.Conv2d(
            self._depthwise_vert.out_channels,
            out_channels,
            1,
        )

        self._conv_out = nn.Conv1d(
            out_channels,
            out_features,
            12,
        )

        self.reset_parameters()

    def reset_parameters(self):
        sqrt2 = math.sqrt(2)

        # nn.init.orthogonal_(self._depthwise_full.weight, sqrt2)
        nn.init.zeros_(self._depthwise_full.bias)

        # nn.init.orthogonal_(self._depthwise_hori.weight, sqrt2)
        nn.init.zeros_(self._depthwise_hori.bias)

        # nn.init.orthogonal_(self._depthwise_vert.weight, sqrt2)
        nn.init.zeros_(self._depthwise_vert.bias)

        nn.init.zeros_(self._conv_out.bias)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        assert x.dtype == torch.long
        assert x.ndim == 2 and x.shape[1] == 16, x.shape

        # (N, 16) -> (N, 16, NUM_CLASSES)
        x = F.one_hot(x, NUM_CLASSES)
        x = x.float()

        # -> (N, NUM_CLASSES, 16)
        x = torch.permute(x, (0, 2, 1))

        # -> (N, NUM_CLASSES * m, 1)
        x_full = self._depthwise_full(x)
        x_full = F.leaky_relu(x_full)
        # -> (N, out * 4, 1)
        x_full = self._pointwise_full(x_full)
        x_full = F.leaky_relu(x_full)

        # -> (N, NUM_CLASSES, 4, 4)
        board = torch.reshape(x, (-1, NUM_CLASSES, 4, 4))

        # -> (N, NUM_CLASSES * m, 4, 1)
        x_hori = self._depthwise_hori(board)
        x_hori = F.leaky_relu(x_hori)
        # -> (N, out, 4, 1)
        x_hori = self._pointwise_hori(x_hori)
        x_hori = F.leaky_relu(x_hori)

        # -> (N, NUM_CLASSES * m, 1, 4)
        x_vert = self._depthwise_vert(board)
        x_vert = F.leaky_relu(x_vert)
        # -> (N, out, 1, 4)
        x_vert = self._pointwise_vert(x_vert)
        x_vert = F.leaky_relu(x_vert)

        x = torch.cat(
            (
                torch.reshape(x_full, (-1, self._out_channels, 4)),
                torch.flatten(x_hori, 2),
                torch.flatten(x_vert, 2),
            ),
            dim=2,
        )
        x = self._conv_out(x)
        x = F.leaky_relu(x)

        x = torch.flatten(x, 1)
        return x.to(torch.float)


class CNNActorNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_hidden: int,
        num_hidden2: int,
    ) -> None:
        super().__init__()

        self._fc1 = nn.Linear(in_features, num_hidden)
        self._fc2 = nn.Linear(self._fc1.out_features, num_hidden2)

        # logits output
        self._out = nn.Linear(self._fc2.out_features, NUM_ACTIONS)

        self.reset_parameters()

    def reset_parameters(self):
        sqrt2 = math.sqrt(2)

        nn.init.orthogonal_(self._fc1.weight, sqrt2)
        nn.init.zeros_(self._fc1.bias)

        nn.init.orthogonal_(self._fc2.weight, sqrt2)
        nn.init.zeros_(self._fc2.bias)

        nn.init.orthogonal_(self._out.weight, 0.01)
        nn.init.zeros_(self._out.bias)

    def forward(
        self,
        x: torch.FloatTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        # -> (N, num_hidden)
        x = self._fc1(x)
        x = F.relu(x)

        # -> (N, num_hidden2)
        x = self._fc2(x)
        x = F.relu(x)

        # -> (N, 4)
        logits = self._out(x)

        # translation such that logits <= 0
        # note that logit_max is a constant to the graph (detached)
        logit_max, _ = torch.max(logits.detach(), dim=-1, keepdim=True)
        logits = logits - logit_max

        return logits


class CNNCriticNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_hidden: int,
        num_hidden2: int,
    ) -> None:
        super().__init__()

        self._fc1 = nn.Linear(in_features, num_hidden)

        self._fc2 = nn.Linear(self._fc1.out_features, num_hidden2)

        # value output
        self._out = nn.Linear(self._fc2.out_features, 1)

        self.reset_parameters()

    def reset_parameters(self):
        sqrt2 = math.sqrt(2)

        nn.init.orthogonal_(self._fc1.weight, sqrt2)
        nn.init.zeros_(self._fc1.bias)

        nn.init.orthogonal_(self._fc2.weight, sqrt2)
        nn.init.zeros_(self._fc2.bias)

        nn.init.orthogonal_(self._out.weight, 1)
        nn.init.zeros_(self._out.bias)

    def forward(
        self,
        x: torch.FloatTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        # -> (N, num_hidden)
        x = self._fc1(x)
        x = F.relu(x)

        # -> (N, num_hidden2)
        x = self._fc2(x)
        x = F.relu(x)

        # -> (N, 1)
        x = self._out(x)
        x = torch.squeeze(x, dim=-1)

        return x
