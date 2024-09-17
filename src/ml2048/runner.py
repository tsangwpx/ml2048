from typing import Any, Callable, Self, Sequence, TypedDict

import numpy as np
import torch
from numba import njit

from ml2048.event import EventEmitter
from ml2048.game_numba import VecGame, VecStepResult
from ml2048.policy import Policy


class VecRunnerResult(TypedDict, total=False):
    # bookkeeping purpose
    game_id: torch.LongTensor
    step_id: torch.LongTensor

    state: torch.ByteTensor
    valid_actions: torch.BoolTensor
    action: torch.ByteTensor
    action_log_prob: torch.FloatTensor
    next_state: torch.ByteTensor
    next_valid_actions: torch.BoolTensor
    reward: torch.FloatTensor  # scalar rewards / vector of forthcoming rewards
    terminated: torch.BoolTensor
    adv: torch.FloatTensor


class VecRunner:
    """
    Run multiple games together.

    Memorize the last N steps
    """

    EVENT_PREPARED: str = "prepared"
    """
    args: (reset_indices,)
    """

    EVENT_STEPPED: str = "stepped"
    """
    args: (step_result,)
    """

    def __init__(
        self,
        env: VecGame,
        capacity: int,
        *,
        sample_device: torch.device | str | None = None,
    ):
        self.env = env
        self.sample_device = sample_device

        self._emitter = EventEmitter()

        self._vec_size = self.env._size
        self._capacity = capacity

        self._game_count = self._vec_size
        self._games: dict[int, Any] = {}

        # Mapping slot id to game id
        self._game_id = np.arange(self._vec_size, dtype=np.int64)
        self._step_ids = np.zeros((self._vec_size,), dtype=np.int64)

        self._shape = batch_size = (capacity, self._vec_size)

    def add_callback(self, event: str, fn: Callable[..., Any]):
        assert event in {self.EVENT_STEPPED, self.EVENT_PREPARED}

        self._emitter.add_listener(event, fn)

    def step_once(
        self,
        policy: Policy,
    ):
        (new_indices,) = self.env.prepare()

        self._emitter.emit(
            self.EVENT_PREPARED,
            (
                self.env,
                new_indices,
            ),
        )
        del new_indices

        state, valid_actions = self.env.observations()

        state = torch.from_numpy(state)
        state = state.to(self.sample_device, torch.long)

        valid_actions = torch.from_numpy(valid_actions)
        valid_actions = valid_actions.to(self.sample_device, torch.bool)

        with torch.no_grad():
            sample_actions, sample_log_probs = policy.sample_actions(
                state,
                valid_actions,
            )
            del state, valid_actions

        result = self.env.step(sample_actions.cpu().numpy())

        self._emitter.emit(
            self.EVENT_STEPPED,
            (self.env, result, sample_actions, sample_log_probs),
        )

    def step_many(
        self,
        policy: Policy,
        count: int,
    ):
        for _ in range(count):
            self.step_once(policy)


@njit
def _update_count(
    counter: np.ndarray,
    state: np.ndarray,
    terminated: np.ndarray,
):
    assert state.ndim == 2 and state.shape[-1] == 16, "Bad state shape"
    assert terminated.ndim == 1, "Bad terminated shape"
    assert state.shape[0] == terminated.shape[0], "bad length"

    size = terminated.shape[0]
    for idx in range(size):
        if not terminated[idx]:
            continue

        maxcell = state[idx, :].max()
        counter[maxcell] += 1


class RunnerStats:
    """
    Despite the name, this class records the stats of terminated games.

    Note that VecGame reset a game when it is terminated.
    The output stats is shifted towards games with smaller steps.
    Use game_id to track a fixed number of games instead.
    """

    terminated_count: int

    def __init__(self):
        self.counts = np.zeros((20,), dtype=np.int32)
        self.terminated_count = 0

    def reset(self):
        self.counts.fill(0)
        self.terminated_count = 0

    def on_stepped(
        self,
        game: VecGame,
        result: VecStepResult,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
    ):
        _update_count(self.counts, result["state"], result["terminated"])
        self.terminated_count += np.sum(result["terminated"])

    def summary(self) -> list[tuple[int, int, ...]]:
        total = self.counts.sum()
        entries = []
        for power in range(16, 0, -1):
            count = self.counts[power].item()
            if count == 0:
                continue

            maxcell = 2**power
            entries.append((maxcell, count, count / total))

        return entries

    @classmethod
    def combine(cls, seq: Sequence[Self]) -> Self:
        counts = np.sum([s.counts for s in seq], axis=0)
        terminated_count = sum([s.terminated_count for s in seq])
        result = cls()
        result.counts = counts
        result.terminated_count = terminated_count

        return result
