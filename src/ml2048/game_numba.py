"""
2048 implemented with numpy and numba
"""

from typing import Any, Callable, NamedTuple, Optional, Sequence, TypedDict

import numba
import numpy as np
from numba import njit

from ml2048.game import STEP_DOWN, STEP_LEFT, STEP_RIGHT, STEP_UP, Board

"""
+----+----+----+----+
|  0 |  1 |  2 |  3 |
|  4 |  5 |  6 |  7 |
|  8 |  9 | 10 | 11 |
| 12 | 13 | 14 | 15 |
+----+----+----+----+
"""

# Map item to its rendered value.
ITEM_VALUES = np.array(
    [
        0,  # 0
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,  # 8
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,  # 16
        131072,  # 17
    ],
    dtype=np.int32,
)


@njit(inline="always")
def _mirror(cells: Board):
    # left-right flipping
    for j in range(4):
        s = j * 4

        # swap cells in first and last column
        cells[s], cells[s + 3] = cells[s + 3], cells[s]

        # Swap cells in second and third column
        cells[s + 1], cells[s + 2] = cells[s + 2], cells[s + 1]


@njit(inline="always")
def _transpose(cells: Board):
    # transpose, swap along diagonal

    # (i, j) in the strictly lower triangular part
    for j in range(1, 4):
        for i in range(0, j):
            u = j * 4 + i
            v = i * 4 + j
            cells[u], cells[v] = cells[v], cells[u]


@njit(inline="always")
def _anti_transpose(cells: Board):
    # anti-transpose, swap along anti-diagonal

    # (i, j) in the top-left triangular part
    for j in range(0, 3):
        for i in range(0, 3 - j):
            u = j * 4 + i
            v = 15 - 4 * i - j  # v = (3 - i) * 4 + (3 - j)
            cells[u], cells[v] = cells[v], cells[u]


@njit(inline="always")
def _push_row(
    board: np.ndarray,
    offset: int,
    stride: int,
    buckets: Optional[np.ndarray],
):
    memo = 0  # current  number
    w = 0  # write pointer

    for r in range(4):  # read pointer
        num = board[offset + r * stride]

        if num == 0:
            # skip empty cell
            continue
        elif memo == 0:
            memo = num  # save this cell
        elif memo == num:
            # combine two identical numbers
            memo = 0
            board[offset + w * stride] = num + 1
            w += 1

            if buckets is not None:
                # because 1 <= num <= 16
                # buckets[0] is associated with num = 1
                #
                buckets[num] += 1
        else:
            # write the saved number, and save this number
            board[offset + w * stride] = memo
            w += 1
            memo = num

    if memo != 0:
        board[offset + w * stride] = memo
        w += 1

    # Fill the remaining row
    while w < 4:
        board[offset + w * stride] = 0
        w += 1


@njit
def _step_left(board: np.ndarray, buckets: Optional[np.ndarray]):
    _push_row(board, 0, 1, buckets)
    _push_row(board, 4, 1, buckets)
    _push_row(board, 8, 1, buckets)
    _push_row(board, 12, 1, buckets)


@njit
def _step_right(board: np.ndarray, buckets: Optional[np.ndarray]):
    _push_row(board, 3, -1, buckets)
    _push_row(board, 7, -1, buckets)
    _push_row(board, 11, -1, buckets)
    _push_row(board, 15, -1, buckets)


@njit
def _step_up(board: np.ndarray, buckets: Optional[np.ndarray]):
    _push_row(board, 0, 4, buckets)
    _push_row(board, 1, 4, buckets)
    _push_row(board, 2, 4, buckets)
    _push_row(board, 3, 4, buckets)


@njit
def _step_down(board: np.ndarray, buckets: Optional[np.ndarray]):
    _push_row(board, 12, -4, buckets)
    _push_row(board, 13, -4, buckets)
    _push_row(board, 14, -4, buckets)
    _push_row(board, 15, -4, buckets)


@njit
def _step_kernel(board: np.ndarray, merged: np.ndarray, action: int | np.ndarray):
    if action == STEP_LEFT:
        _step_left(board, merged)
    elif action == STEP_RIGHT:
        _step_right(board, merged)
    elif action == STEP_UP:
        _step_up(board, merged)
    elif action == STEP_DOWN:
        _step_down(board, merged)


# @njit
def _spawn(
    board: np.ndarray,
    prob: float,
    rand: np.random.Generator,
) -> bool:
    """
    Spawn one number in empty cells

    Due to a memory leak in numba random generator,
    this function cannot be jitted.

    :param prob: probability to spawn 2. otherwise 4.
    """

    empty_indices = np.flatnonzero(board == 0)
    if empty_indices.size == 0:
        return False

    if empty_indices.size == 1:
        idx = empty_indices[0]
    else:
        idx = rand.choice(empty_indices)
        # target = rand.integers(0, empty_indices.size)
        # idx = empty_indices[target]

    chance = rand.uniform()
    if chance <= prob:
        board[idx] = 1
    else:
        board[idx] = 2

    return True


@njit
def _spawn2(
    randperm: np.ndarray,
    randfloat: np.ndarray,
    rand_idx: int,
    board: np.ndarray,
    two_prob: float,
    count: int,
) -> bool:
    """
    Alternative implementation of _spawn()

    The implementation utilize pregenerated random arrays.

    Spawn one number in empty cells

    randperm = (N, 16). Each row is a permutation of np.arange(16).
    randfloat = (N,). Each item samples from [0, 1].

    :param two_prob: probability to spawn 2. otherwise 4.
    """

    size = randperm.shape[0]
    while rand_idx >= size:
        rand_idx -= size

    i = 0
    while count > 0 and i < 16:
        idx = randperm[rand_idx, i]
        i += 1

        if board[idx] != 0:
            continue

        count -= 1
        if randfloat[idx] < two_prob:
            board[idx] = 1
        else:
            board[idx] = 2

    return count == 0


@njit
def _line_movable(n1: int, n2: int, n3: int, n4: int) -> tuple[bool, bool]:
    """Check whether a line can be moved forward or backward"""

    z1 = n1 == 0
    z2 = n2 == 0
    z3 = n3 == 0
    z4 = n4 == 0

    # whether it can move position 4 toward position 1
    m41 = z3 & z2 & (z1 | (n1 == n4))
    m14 = z2 & z3 & (z4 | (n1 == n4))

    m31 = z2 & (z1 | (n1 == n3))
    m13 = z2 & (z3 | (n1 == n3))

    m21 = z1 | (n1 == n2)
    m12 = z2 | (n1 == n2)

    m42 = z3 & (z2 | (n2 == n4))
    m24 = z3 & (z4 | (n2 == n4))

    m32 = z2 | (n2 == n3)
    m23 = z3 | (n2 == n3)

    m43 = z3 | (n3 == n4)
    m34 = z4 | (n3 == n4)

    # forward move of position 2
    f2 = (n2 >= 1) and m21 != 0
    f3 = (n3 >= 1) and (m31 | m32) != 0
    f4 = (n4 >= 1) and (m41 | m42 | m43) != 0

    # backward move of position 3
    b3 = (n3 >= 1) and m34 != 0
    b2 = (n2 >= 1) and (m23 | m24) != 0
    b1 = (n1 >= 1) and (m12 | m13 | m14) != 0

    forward = (f2 | f3 | f4) != 0
    backward = (b3 | b2 | b1) != 0

    return forward, backward


@njit
def _compute_valid_actions(
    board: np.ndarray | Sequence[int],
    result: np.ndarray,
) -> bool:
    """
    Given a board, set the array of valid actions.
    Return True if there is any valid action.
    """
    l1, r1 = _line_movable(board[0], board[1], board[2], board[3])
    l2, r2 = _line_movable(board[4], board[5], board[6], board[7])
    l3, r3 = _line_movable(board[8], board[9], board[10], board[11])
    l4, r4 = _line_movable(board[12], board[13], board[14], board[15])

    u1, d1 = _line_movable(board[0], board[4], board[8], board[12])
    u2, d2 = _line_movable(board[1], board[5], board[9], board[13])
    u3, d3 = _line_movable(board[2], board[6], board[10], board[14])
    u4, d4 = _line_movable(board[3], board[7], board[11], board[15])

    left = l1 or l2 or l3 or l4
    right = r1 or r2 or r3 or r4

    up = u1 or u2 or u3 or u4
    down = d1 or d2 or d3 or d4

    result[0] = left
    result[1] = right
    result[2] = up
    result[3] = down

    return bool(left or right or up or down)


_ACTION_TRANSFORMATIONS = {
    STEP_LEFT: _step_left,
    STEP_RIGHT: _step_right,
    STEP_UP: _step_up,
    STEP_DOWN: _step_down,
}

_BOARD_SHAPE = (16,)
_BOARD_DTYPE = np.int8
_MERGED_SHAPE = (16,)
_MERGED_DTYPE = np.int8
_ACTION_SHAPE = (4,)


class Game:
    _rand: np.random.Generator

    _completed: bool
    _board: np.ndarray
    _board_tmp: np.ndarray
    _merged: np.ndarray
    _valid_actions: np.ndarray

    def __init__(
        self,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
        two_prob: float = 0.8,
        seed: Optional[int] = None,
    ):
        if reward_fn is None:
            reward_fn = reward_fn_normal

        self._two_prob = two_prob
        self._reward_fn = reward_fn

        self._completed = False
        self._board = np.empty(_BOARD_SHAPE, dtype=_BOARD_DTYPE)
        self._board_tmp = np.empty_like(self._board)
        self._merged = np.empty(_MERGED_SHAPE, dtype=_MERGED_DTYPE)
        self._valid_actions = np.empty(_ACTION_SHAPE, dtype=np.bool_)

        self.reset(seed)

    def reset(self, seed: Optional[int] = None):
        self._rand = np.random.default_rng(seed)

        self._completed = False
        self._board.fill(0)
        self._board_tmp.fill(0)
        self._merged.fill(0)
        self._valid_actions.fill(False)

        _spawn(self._board, self._two_prob, self._rand)
        _compute_valid_actions(self._board, self._valid_actions)

    def _spawn(self):
        """
        Spawn new 2's in empty cells

        returns the number of newly spawned 2's cells
        """

        spawned = _spawn(self._board, self._two_prob, self._rand)
        return int(spawned)

    def render(self, output: np.ndarray):
        output[:] = ITEM_VALUES[self._board]

    def step(self, direction: int) -> tuple[bool, bool, float]:
        """

        A tuple:

        1. Is the step valid?
        2. Is the game over?
        """

        if self._completed:
            raise RuntimeError("Game over")

        if not self._valid_actions[direction]:
            return False, self._completed, 0

        # save the board state
        self._board_tmp[:] = self._board
        self._merged.fill(0)

        transform = _ACTION_TRANSFORMATIONS[direction]
        transform(self._board, self._merged)
        reward = self._reward_fn(self._board, self._board_tmp, self._merged)

        if np.array_equal(self._board, self._board_tmp):
            return False, self._completed, 0

        spawned = _spawn(self._board, self._two_prob, self._rand)
        assert spawned, (self._board, self._board_tmp)

        _compute_valid_actions(self._board, self._valid_actions)

        self._completed = not self._valid_actions.any()

        return (
            True,
            self._completed,
            reward,
        )


@njit
def reward_fn_normal(
    state: np.ndarray,
    prev_state: np.ndarray,
    merged: np.ndarray,
) -> float:
    """
    Reward the value of merged cell
    """
    reward = 0

    # merged[0] should be always 0
    # but compiler probably find this pattern useful?
    reward += merged[0] * 2
    reward += merged[1] * 4
    reward += merged[2] * 8
    reward += merged[3] * 16
    reward += merged[4] * 32
    reward += merged[5] * 64
    reward += merged[6] * 128
    reward += merged[7] * 256
    reward += merged[8] * 512
    reward += merged[9] * 1024
    reward += merged[10] * 2048
    reward += merged[11] * 4096
    reward += merged[12] * 8192
    reward += merged[13] * 16384
    reward += merged[14] * 32768
    reward += merged[15] * 65536

    return float(reward)


@njit
def reward_fn_improved(
    state: np.ndarray,
    prev_state: np.ndarray,
    merged: np.ndarray,
) -> float:
    """
    Extra reward is assigned to the top left cell.

    This is potential-based reward shaping
    because it is like a potential function: divergence = 0.
    """
    reward = reward_fn_normal(state, prev_state, merged)

    extra = 0
    factor = 64

    # first subtract the prev value
    if prev_state[0] != 0:
        extra -= factor * (1 << prev_state[0])

    # second add the new value
    if state[0] != 0:
        extra += factor * (1 << state[0])

    return reward + float(extra)


@njit
def reward_fn_rank(
    state,
    prev_state: np.ndarray,
    merged: np.ndarray,
) -> float:
    """
    Reward log2(value) of merged cell
    """
    reward = 0.0

    for idx, count in enumerate(merged):
        rank = idx + 1
        reward += rank * count

    return reward


@njit
def reward_fn_maxcell(
    state: np.ndarray, prev_state: np.ndarray, merged: np.ndarray
) -> float:
    """
    Reward 1 for each merged cell and the value of unseen merged value.
    """
    reward = 0.0

    curr_max = np.max(state)
    prev_max = np.max(prev_state)
    if curr_max > prev_max:
        point = 2**curr_max
        reward += point

    reward += merged.sum()

    return reward


class VecStepResult(TypedDict):
    state: np.ndarray
    valid_actions: np.ndarray
    step: np.ndarray
    merged: np.ndarray
    reward: np.ndarray
    score: np.ndarray

    terminated: np.ndarray
    invalid: np.ndarray

    prev_state: np.ndarray
    prev_valid_actions: np.ndarray


class _VecGameLegacy:
    """Legacy implementation of VecGame"""

    _RAND_SIZE: int = 1024

    _rand: np.random.Generator

    _state: np.ndarray
    _valid_actions: np.ndarray
    _merged: np.ndarray
    _reward: np.ndarray
    _terminated: np.ndarray
    _invalid: np.ndarray

    def __init__(
        self,
        size: int,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
        *,
        two_prob: float = 0.8,
        reuse_state: bool = False,
    ):
        if size <= 0:
            raise ValueError(f"size={size}")

        if reward_fn is None:
            reward_fn = reward_fn_normal

        self._size = size
        self._two_prob = two_prob
        self._reuse_state = reuse_state

        self._reward_fn = reward_fn

        self._state = np.empty((size,) + _BOARD_SHAPE, dtype=np.int8)
        self._valid_actions = np.empty((size,) + _ACTION_SHAPE, np.bool_)
        self._merged = np.empty((size, 16), dtype=np.int8)
        self._step = np.empty((size,), dtype=np.int32)
        self._reward = np.empty((size,), dtype=np.float32)
        self._score = np.empty((size,), dtype=np.float32)
        self._terminated = np.empty((size,), dtype=np.bool_)
        self._invalid = np.empty((size,), dtype=np.bool_)

        self._fresh_ids = set()

        self._prev_state = np.empty_like(self._state)
        self._prev_valid_actions = np.empty_like(self._valid_actions)

        self._rand_step = 0
        rand_shape = (self._RAND_SIZE,)
        self._randperm = np.empty(rand_shape + _BOARD_SHAPE, dtype=np.uint8)
        self._randbool = np.empty(rand_shape, dtype=np.bool_)
        self._randfloat = np.empty(rand_shape, dtype=np.float32)

        self.reset()

    def observations(self) -> tuple[np.ndarray, np.ndarray]:
        return self._state, self._valid_actions

    def _reset_rand(self):
        self._rand.permuted(self._randperm, axis=1, out=self._randperm)
        self._rand.random(dtype=self._randfloat.dtype, out=self._randfloat)
        np.less(self._randfloat, self._two_prob, out=self._randbool)

    def summary(self) -> list[Any]:
        maxcell = np.max(self._state, axis=1)
        values, counts = np.unique(maxcell, return_counts=True)
        total = counts.sum()

        entries = [
            (2 ** maximum.item(), count, count / total)
            for maximum, count in zip(values, counts)
        ]

        entries.sort(key=lambda s: s[0], reverse=True)
        return entries

    def reset(self, seed: Optional[int] = None):
        self._rand = np.random.default_rng(seed)

        self._rand_step = 0
        self._randperm[:, :] = np.arange(16).reshape((1, 16))
        self._reset_rand()

        self._state.fill(0)
        self._valid_actions.fill(False)
        self._step.fill(0)
        self._merged.fill(0)
        self._reward.fill(0)
        self._score.fill(0)
        self._terminated.fill(True)
        self._invalid.fill(False)

        self._prev_state.fill(0)
        self._prev_valid_actions.fill(False)

    def prepare(self) -> tuple[np.ndarray]:
        """ "So that state and valid_actions are available"""

        if self._rand_step >= self._RAND_SIZE:
            self._rand_step = 0
            self._reset_rand()

        # prepare terminated game to a new game
        indices = np.flatnonzero(self._terminated)

        if indices.size == 0:
            return (indices,)

        sample_count = 0
        if self._reuse_state:
            reuse_indices = np.flatnonzero(~self._terminated)
            self._rand.shuffle(reuse_indices)
            sample_count = min(indices.size, reuse_indices.size) // 2
            reuse_switches = self._rand.random((sample_count,), dtype=np.float32)
        else:
            reuse_indices = None
            reuse_switches = None

        for i in range(indices.size):
            idx = indices[i]
            copy_others = (
                reuse_indices is not None
                and i < sample_count
                and reuse_switches[i] >= 0.5
            )
            if copy_others:
                src_idx = reuse_indices[i]

                self._step[idx] = self._step[src_idx]
                self._score[idx] = self._score[src_idx]
                self._state[idx, :] = self._state[src_idx, :]
                self._valid_actions[idx, :] = self._valid_actions[src_idx, :]
            else:
                self._step[idx] = 0
                self._score[idx] = 0

                state = self._state[idx, :]
                valid_actions = self._valid_actions[idx, :]

                state.fill(0)

                _spawn2(
                    self._randperm,
                    self._randfloat,
                    self._rand_step + idx,
                    state,
                    self._two_prob,
                    2,
                )

                _compute_valid_actions(state, valid_actions)

        self._terminated.fill(False)
        self._invalid.fill(False)
        self._reward.fill(0)
        self._merged.fill(0)

        return (indices,)

    def step(
        self,
        actions: np.ndarray,
    ) -> VecStepResult:
        assert actions.shape == (self._size,), actions.shape

        np.copyto(self._prev_state, self._state)
        np.copyto(self._prev_valid_actions, self._valid_actions)

        tmp_state = np.empty(_BOARD_SHAPE, dtype=_BOARD_DTYPE)
        # print(self._state.reshape((4, 4)))

        for idx in range(self._size):
            state = self._state[idx, :]
            action = actions[idx]
            valid_actions = self._valid_actions[idx, :]
            merged = self._merged[idx, :]
            prev_state = self._prev_state[idx, :]

            if valid_actions[action]:
                tranform = _ACTION_TRANSFORMATIONS[action]
                tranform(state, self._merged[idx, :])
                reward = self._reward_fn(state, prev_state, merged)
                self._reward[idx] = reward

                _spawn2(
                    self._randperm,
                    self._randfloat,
                    self._rand_step + idx,
                    state,
                    self._two_prob,
                    1,
                )
                # reuse tmp_state allocation for other purpose
                has_valid_actions = _compute_valid_actions(state, valid_actions)
                self._terminated[idx] = not has_valid_actions
            else:
                self._invalid[idx] = True

        self._step += 1 - self._invalid
        self._rand_step += 1

        return {
            "state": self._state,
            "valid_actions": self._valid_actions,
            "merged": self._merged,
            "step": self._step,
            "reward": self._reward,
            "terminated": self._terminated,
            "invalid": self._invalid,
            "prev_state": self._prev_state,
            "prev_valid_actions": self._valid_actions,
        }


class VecGame:
    """
    Vectorized 2048 environment

    Game states are stored in structured arrays to benefit spatial locality.

    Random numbers are generated in batch.

    step() is profiled and accelerated with numba.
    """

    _RAND_SIZE: int = 1024

    _rand: np.random.Generator

    _DATA_SPEC = [
        ("id", np.int32, ()),
        ("step", np.int32, ()),
        ("score", np.float32, ()),
        ("reward", np.float32, ()),
        ("board", np.uint8, _BOARD_SHAPE),
        ("merged", np.uint8, _MERGED_SHAPE),
        ("valid_actions", np.uint8, _ACTION_SHAPE),
        # offset 52 here
        ("terminated", np.uint8, ()),
        ("invalid", np.uint8, ()),
        ("_padding", np.uint8, 10),
    ]
    _DATA_DTYPE = np.dtype(_DATA_SPEC, align=True)
    assert _DATA_DTYPE.itemsize == 64, _DATA_DTYPE.itemsize

    def __init__(
        self,
        size: int,
        reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
        *,
        two_prob: float = 0.8,
        reuse_state: bool = False,
    ):
        if size <= 0:
            raise ValueError(f"size={size}")

        if reward_fn is None:
            reward_fn = reward_fn_normal

        self._size = size
        self._two_prob = two_prob
        self._reuse_state = reuse_state

        self._data = np.empty((size,), dtype=self._DATA_DTYPE)
        self._reward_fn = reward_fn

        self._prev_state = np.empty_like(self._data["board"])
        self._prev_valid_actions = np.empty_like(self._data["valid_actions"])

        rand_shape = (self._RAND_SIZE,)
        self._randperm = np.empty(rand_shape + _BOARD_SHAPE, dtype=np.uint8)
        self._randfloat = np.empty(rand_shape, dtype=np.float32)
        self._rand_step = 0

        self._game_count = 0

        self.reset()

    def observations(self) -> tuple[np.ndarray, np.ndarray]:
        return self._data["board"], self._data["valid_actions"]

    def _reset_rand(self):
        self._rand.permuted(self._randperm, axis=1, out=self._randperm)
        self._rand.random(dtype=self._randfloat.dtype, out=self._randfloat)

    def summary(self) -> list[Any]:
        maxcell = np.max(self._data["board"], axis=1)
        values, counts = np.unique(maxcell, return_counts=True)
        total = counts.sum()

        entries = [
            (2 ** maximum.item(), count, count / total)
            for maximum, count in zip(values, counts)
        ]

        entries.sort(key=lambda s: s[0], reverse=True)
        return entries

    def reset(self, seed: Optional[int] = None):
        self._rand = np.random.default_rng(seed)

        self._rand_step = 0
        self._randperm[:, :] = np.arange(16).reshape((1, 16))
        self._reset_rand()

        self._data.fill(0)
        self._data["terminated"] = True

        self._prev_state.fill(0)
        self._prev_valid_actions.fill(False)

    def prepare(self) -> tuple[np.ndarray]:
        """So that state and valid_actions are ready"""

        if self._rand.random() >= 0.9 or self._rand_step >= self._RAND_SIZE:
            self._rand_step = 0
            self._reset_rand()

        rand_offset = self._rand.integers(0, self._RAND_SIZE)

        # prepare terminated game to a new game
        indices = np.flatnonzero(self._data["terminated"])

        if indices.size == 0:
            return (indices,)

        for i in range(indices.size):
            idx = indices[i]

            if True:
                entry = self._data[idx]
                entry.fill(0)

                game_id = self._game_count
                self._game_count += 1

                entry["id"] = game_id
                state = entry["board"]
                valid_actions = entry["valid_actions"]

                _spawn2(
                    self._randperm,
                    self._randfloat,
                    self._rand_step + rand_offset + idx,
                    state,
                    self._two_prob,
                    2,
                )
                _compute_valid_actions(state, valid_actions)

        return (indices,)

    def step(
        self,
        actions: np.ndarray,
    ) -> VecStepResult:
        """
        Call prepare() before this.
        """

        assert actions.shape == (self._size,), actions.shape

        rand_offset = self._rand.integers(0, self._RAND_SIZE)

        np.copyto(self._prev_state, self._data["board"])
        np.copyto(self._prev_valid_actions, self._data["valid_actions"])

        _vec_step(
            self._data,
            actions,
            self._prev_state,
            self._reward_fn,
            self._two_prob,
            self._rand_step + rand_offset,
            self._randperm,
            self._randfloat,
        )
        self._rand_step += 1

        return {
            "state": self._data["board"],
            "valid_actions": self._data["valid_actions"],
            "merged": self._data["merged"],
            "step": self._data["step"],
            "reward": self._data["reward"],
            "score": self._data["score"],
            "terminated": self._data["terminated"],
            "invalid": self._data["invalid"],
            "prev_state": self._prev_state,
            "prev_valid_actions": self._prev_valid_actions,
        }


@njit
def _find_terminated(data: np.ndarray, out: np.ndarray) -> int:
    count = 0
    size = data.shape[0]
    assert size <= out.size

    for idx in range(size):
        if data[idx]["terminated"]:
            out[count] = idx
            count += 1

    return count


@njit(parallel=True)
def _vec_step(
    data: np.ndarray,
    actions: np.ndarray,
    prev_boards: np.ndarray,
    reward_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    two_prob: float,
    rand_seed: int,
    randperm: np.ndarray,
    randfloat: np.ndarray,
):
    assert data.ndim == 1, "Bad shape"
    size = data.shape[0]

    for idx in numba.prange(size):
        valid_actions = data[idx]["valid_actions"]

        if valid_actions[actions[idx]]:
            data[idx]["step"] += 1

            board = data[idx]["board"]
            merged = data[idx]["merged"]
            merged[0:16] = 0

            _step_kernel(board, merged, actions[idx])

            prev_board = prev_boards[idx, :]
            reward = reward_fn(board, prev_board, merged)
            ds = reward_fn_normal(board, prev_board, merged)
            data[idx]["reward"] = reward
            data[idx]["score"] += ds

            _spawn2(randperm, randfloat, rand_seed + idx, board, two_prob, 1)
            playable = _compute_valid_actions(board, valid_actions)
            data[idx]["terminated"] = not playable
            data[idx]["invalid"] = False
        else:
            data[idx]["invalid"] = True


class Summary(NamedTuple):
    maxcell: int
    score: int


class Episode:
    def __init__(self) -> None:
        self._completed = False

        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.extras: list[dict[str, Any] | None] = []
        self.returns: np.ndarray | None = None

        self._score: int = 0
        self._summary: Summary | None = None

    def total_score(self, index: int = -1) -> int:
        state = self.states[index].astype(dtype=np.int32)
        return (2**state).sum().item()

    def append(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        extra: dict[str, Any] | None = None,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.extras.append(extra)

        if reward >= 0:
            self._score += reward

    def finish(self, state: np.ndarray):
        self.states.append(state)

    def summary(self) -> Summary:
        state = self.states[-1]
        max_power = state.max().item()
        maxcell = 2**max_power if max_power else 0

        return Summary(
            maxcell=maxcell,
            score=self._score,
        )


@njit
def _unused_lookup(row: np.ndarray, out: np.ndarray):
    """
    Alternative idea for probably fast line stepping by lookup a precomputed table.

    Sketch:
    This roughly implements push_left or something similar based on hash table.
    1. Assign a number to each unique item in row
    2. Compute the hash of the row based on the assigned number
    3. Get the result from the precomputed table
    4. Decode the result according to the table of assigned numbers

    Stat is lacking:
    What item is merged? How many times is merged?
    Such metadata may be supplied as well.

    Each key is 3bits * 4 = 12bits, fit in int16.
    There is 2**12 = 4096 entries.
    The table is 4096 * 2 bytes = 8KiB
    It may be smaller since the leading slot is either 0 or 1
    """

    # map a cell number to its id in [1, 4]. id=0 is used for empty cell.
    # the table at most consists of 4 non-zero items. namely, 1, 2, 3, 4.
    table = [0] * 18
    nassigned = 0
    key = 0

    for i in range(4):
        key <<= 3
        item = row[i]
        if item == 0:
            continue

        if table[item] == 0:
            nassigned += 1
            table[item] = nassigned

        key |= table[item]

    result = LOOKUP_TABLE[key]

    idx = (result >> 9) & 7
    out[0] = table[idx]

    idx = (result >> 6) & 7
    out[1] = table[idx]

    idx = (result >> 3) & 7
    out[2] = table[idx]

    idx = (result >> 0) & 7
    out[3] = table[idx]
