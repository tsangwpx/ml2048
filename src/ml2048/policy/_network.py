import math

import torch
from torch import nn
from torch.nn import functional as F

NUM_CELLS = 16
NUM_CLASSES = 16
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


class Conv2DActorNetwork(nn.Module):
    """legacy"""

    def __init__(
        self,
        num_conv1: int,
        num_conv2: int,
        num_hidden: int,
        num_hidden2: int,
    ) -> None:
        super().__init__()

        self._num_conv1 = num_conv1
        self._num_conv2 = num_conv2

        self._conv1 = nn.Conv2d(NUM_CLASSES, num_conv1, kernel_size=2)
        self._conv2 = nn.Conv2d(num_conv1, num_conv2, kernel_size=3)

        self._fc1 = nn.Linear(num_conv2 + NUM_ACTIONS, num_hidden)
        nn.init.orthogonal_(self._fc1.weight)
        nn.init.constant_(self._fc1.bias, 0)

        self._fc2 = nn.Linear(self._fc1.out_features, num_hidden2)
        nn.init.orthogonal_(self._fc2.weight)
        nn.init.constant_(self._fc2.bias, 0)

        # logits output
        self._out = nn.Linear(self._fc2.out_features, NUM_ACTIONS)

    def forward(
        self,
        x: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        assert x.shape[-1] == 16, x.shape

        if x.ndim >= 2:
            batch_size = x.shape[0]
            action_shape = torch.Size((batch_size, 4))
        else:
            action_shape = torch.Size((4,))

        # -> (N, 4, 4)
        x = x.view((-1, 4, 4))

        # -> (N, 4, 4, NUM_CLASSES)
        x = F.one_hot(x, NUM_CLASSES).float()

        # -> (N, NUM_CLASSES, 4, 4)
        x = x.permute(0, 3, 1, 2)

        # -> (N, num_conv1, 3, 3)
        x = self._conv1(x)
        x = F.relu(x)

        # -> (N, num_conv2, 1, 1)
        x = self._conv2(x)
        x = F.relu(x)

        # -> (N, num_conv2)
        x = x.reshape(-1, self._num_conv2)

        # valid_actions -> (N, 4)
        valid_actions = valid_actions.float()
        valid_actions = valid_actions.reshape((-1, 4))

        # -> (N, num_conv2 + 4)
        x = torch.cat((x, valid_actions), dim=1)

        # -> (N, num_hidden)
        x = self._fc1(x)
        x = F.relu(x)
        # x = self._dropout1(x)

        # -> (N, num_hidden)
        x = self._fc2(x)
        x = F.relu(x)
        # x = self._dropout2(x)

        # -> (N, 4)
        logits = self._out(x)

        # translation such that logits <= 0
        # note that logit_max is a constant to the graph (detached)
        logit_max, _ = torch.max(logits.detach(), dim=1, keepdim=True)
        logits = logits - logit_max

        # output
        logits = logits.reshape(action_shape)

        return logits


class Conv2DActorNetworkNew(nn.Module):
    """legacy"""

    def __init__(
        self,
        num_conv1: int,
        num_conv2: int,
        num_hidden: int,
        num_hidden2: int,
    ) -> None:
        super().__init__()

        # (16,)
        # -> (16,16) one hot
        # -> (16,4,4) reshape

        # -> (16,3,3) 16-deepwise 2x2 (groups=16)
        # -> (256,3,3)

        # -> (16, 1, 1) Conv2d(16, 16, 4, groups=16)
        # -> (1, 16) Conv2d(16, 16, 4, groups=16)
        # -> (256,) Conv1d(1, 256, 4)

        self._num_conv1 = num_conv1
        self._num_conv2 = num_conv2

        # self._zero_affine = nn.Linear(16, num_conv2)

        self._class_neigh = 5
        self._class_conv = nn.Conv2d(2, num_conv1, (self._class_neigh, NUM_CELLS))
        self._conv_dim = num_conv1 * (NUM_CLASSES - self._class_neigh)

        self._batch_normalize = nn.BatchNorm2d(num_conv1)

        self._fc_input_dim = self._conv_dim + NUM_ACTIONS

        self._fc1 = nn.Linear(self._fc_input_dim, num_hidden)
        nn.init.orthogonal_(self._fc1.weight)
        nn.init.constant_(self._fc1.bias, 0)

        self._fc2 = nn.Linear(self._fc1.out_features, num_hidden2)
        nn.init.orthogonal_(self._fc2.weight)
        nn.init.constant_(self._fc2.bias, 0)

        # logits output
        self._out = nn.Linear(self._fc2.out_features, NUM_ACTIONS)

    def forward(
        self,
        x: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        assert x.ndim == 2 and x.shape[1] == 16, x.shape
        assert (
            valid_actions.ndim == 2 and valid_actions.shape[1] == 4
        ), valid_actions.shape
        assert valid_actions.shape[0] == x.shape[0], (valid_actions.shape, x.shape)

        action_shape = x.shape[:-1] + (4,)

        valid_actions = valid_actions.float()

        # -> (N, 16, NUM_CLASSES)
        x = F.one_hot(x, NUM_CLASSES).float()

        # -> (N, NUM_CLASSES, 16)
        x = x.permute(0, 2, 1)

        x_zero = x[:, 0:1, :]  # (N, 1, 16)
        x_zero = torch.unsqueeze(x_zero, 1)  # (N, 1, 1, 16)

        x_nonzero = x[:, 1:, :]  # (N, NUM_CLASSES - 1, 16)
        x_nonzero = torch.unsqueeze(x_nonzero, 1)  # (N, 1, NUM_CLASSES - 1, 16)

        # x_zero: (N, 1, NUM_CLASSES - 1, 16)
        # x_nonzero: (N, 1, NUM_CLASSES - 1, 16)
        x_zero, x_nonzero = torch.broadcast_tensors(x_zero, x_nonzero)

        # -> (N, 2, NUM_CLASSES - 1, 16)
        x = torch.cat((x_zero, x_nonzero), dim=1)
        x = self._class_conv(x)  # (N, num_conv1, NUM_CLASSES - num_neigh, 1)
        x = self._batch_normalize(x)
        x = F.relu(x)
        x = torch.flatten(x, 1, 3)  # (N, num_conv1 * (NUM_CLASSES - num_neigh))

        # -> (N, fc_input_dim)
        x = torch.cat((valid_actions, x), dim=1)

        # -> (N, num_hidden)
        x = self._fc1(x)
        x = F.relu(x)
        # x = self._dropout1(x)

        # -> (N, num_hidden2)
        x = self._fc2(x)
        x = F.relu(x)
        # x = self._dropout2(x)

        # -> (N, 4)
        logits = self._out(x)

        # translation such that logits <= 0
        # note that logit_max is a constant to the graph (detached)
        logit_max, _ = torch.max(logits.detach(), dim=1, keepdim=True)
        logits = logits - logit_max

        # output
        logits = logits.reshape(action_shape)

        return logits


class CriticNetwork(nn.Module):
    """
    Given a state, return its state value

    legacy
    """

    def __init__(self, num1: int, num2: int) -> None:
        super().__init__()

        self._num1 = num1
        self._num2 = num2

        self._affine1 = nn.Linear(NUM_CELLS * NUM_CLASSES, self._num1)
        # self._dropout1 = nn.Dropout(0.1)

        # self._affine2 = nn.Linear(self._num1 + 4, num2)
        # # self._dropout2 = nn.Dropout(0.1)
        #
        # self._affine3 = nn.Linear(num2, num2)

        self._out = nn.Linear(num1, 1)

    def forward(
        self,
        x: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        assert x.shape[-1] == 16, x.shape

        if x.ndim >= 2:
            value_shape = x.shape[0:-1]
        else:
            value_shape = torch.Size(())

        x = x.reshape(-1, 16)

        x = F.one_hot(x, NUM_CLASSES).float()
        x = x.view(-1, NUM_CELLS * NUM_CLASSES)

        x = self._affine1(x)
        x = F.relu(x)
        # x = self._dropout1(x)

        # # valid_actions -> (N, 4)
        # valid_actions = valid_actions.float()
        # valid_actions = valid_actions.reshape((-1, 4))

        # x = torch.concatenate((x, valid_actions), dim=1)
        #
        # x = self._affine2(x)
        # x = F.relu(x)
        # # x = self._dropout2(x)
        #
        # x = self._affine3(x)
        # x = F.relu(x)

        x = self._out(x)

        x = x.reshape(value_shape)

        return x


class ConvCriticNetwork(nn.Module):
    """
    Given a state, return its state value

    legacy
    """

    def __init__(self, num_conv1: int, num_hidden: int) -> None:
        super().__init__()

        self._num_hidden = num_hidden
        self._num_conv1 = num_conv1

        self._conv1 = nn.Conv1d(
            NUM_CLASSES, num_conv1, kernel_size=16, groups=NUM_CLASSES
        )
        # self._conv1_dropout = nn.Dropout(0.1)

        self._affine1 = nn.Linear(num_conv1, self._num_hidden)

        self._out = nn.Linear(self._num_hidden, 1)

    def forward(
        self,
        x: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        assert x.shape[-1] == 16, x.shape

        output_shape = x.shape[:-1]

        x = x.reshape((-1, 16))
        x = F.one_hot(x, NUM_CLASSES).float()

        # -> (N, NUM_CLASSES, 16)
        x = x.permute(0, 2, 1)

        # -> (N, num_conv1, 1)
        x = self._conv1(x)
        x = F.relu(x)

        x = torch.squeeze(x, -1)

        # -> (N, num_hidden)
        x = self._affine1(x)
        x = F.relu(x)

        x = self._out(x)

        x = x.reshape(output_shape)

        return x
