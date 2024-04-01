import collections
import dataclasses
from typing import Any, Iterator, NamedTuple, TypedDict, cast

import numpy as np
import torch
from numba import njit

from ml2048.game_numba import Game, VecGame, VecStepResult
from ml2048.policy import Policy
from ml2048.runner import EpisodeRunner
from ml2048.util import check_tensors, new_tensors

# Trajectory variables and their shapes
REPLAY_SPEC = {
    "state": ((16,), torch.int8),
    "valid_actions": ((4,), torch.bool),
    "action": ((), torch.int8),
    "action_log_prob": ((), torch.float32),
    "reward": ((), torch.float32),
    "next_state": ((16,), torch.int8),
    "next_valid_actions": ((4,), torch.bool),
    "step": ((), torch.int32),
    "terminated": ((), torch.bool),
}


class ReplayBufferDict(TypedDict, total=False):
    state: torch.ByteTensor
    valid_actions: torch.BoolTensor
    action: torch.ByteTensor
    action_log_prob: torch.FloatTensor
    reward: torch.FloatTensor
    next_state: torch.ByteTensor
    next_valid_actions: torch.BoolTensor
    terminated: torch.BoolTensor


@njit
def compute_discounted_returns(
    rewards: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute discounted returns"""
    size = rewards.size
    returns = np.zeros((size,), dtype=np.float32)

    # backward pass, compute discounted total rewards
    running = 0.0

    for i in range(size - 1, -1, -1):
        running *= gamma
        running += rewards[i]
        returns[i] = running

    return returns


class StatEntry(NamedTuple):
    maxcell: int
    count: int
    scores: float
    steps: float


def episode_summary(
    episodes: list[dict[str, Any]],
    *,
    file: Any = None,
) -> list[StatEntry]:
    bins = {}
    for ep in episodes:
        maxcell = ep["maxcell"]

        if maxcell not in bins:
            bins[maxcell] = []

        bins[maxcell].append((ep["score"], ep["steps"]))

    entries = []

    for maxcell, data in sorted(bins.items(), reverse=True):
        scores, steps = tuple(zip(*data))
        count = len(scores)
        score_avg = sum(scores) / count
        step_avg = sum(steps) / count

        entries.append(StatEntry(maxcell, count, score_avg, step_avg))

    return entries


def compile_episode(
    rows: list[dict[str, Any]],
    gamma: float,
) -> dict[str, Any]:
    size = len(rows) - 1

    assert size >= 0, rows

    states = [s["state"] for s in rows]
    valid_actions = [s["valid_actions"] for s in rows]

    # Remove the last entry
    rows = rows[:-1]
    actions = [s["action"] for s in rows]
    log_probs = [s["action_logp"] for s in rows]
    rewards = [s["reward"] for s in rows]

    states = np.stack(states, axis=0)
    valid_actions = np.stack(valid_actions, axis=0)

    actions = np.array(actions, dtype=np.int8)
    log_probs = np.array(log_probs, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    progress = np.linspace(0, 1, size, dtype=np.float32)

    assert states.shape == (size + 1, 16), states.shape
    assert valid_actions.shape == (size + 1, 4), valid_actions.shape
    assert log_probs.shape == (size,), log_probs
    assert rewards.shape == (size,), rewards.shape

    # returns = compute_discounted_returns(rewards, gamma)
    score = rewards.sum()
    maxcell = states[-1, :].max().item()
    maxcell = 2**maxcell if maxcell >= 1 else 0

    return {
        "states": states,
        "valid_actions": valid_actions,
        "actions": actions,
        "action_log_probs": log_probs,
        "rewards": rewards,
        # "returns": returns,
        "progress": progress,
        "score": score,
        "maxcell": maxcell,
        "steps": actions.shape[0],
    }


def make_batch(batch: dict[str, Any]) -> dict[str, Any]:
    # batch["valid_actions"] = compute_valid_actions(batch["states"])
    # batch["next_valid_actions"] = compute_valid_actions(batch["next_states"])
    return batch


def merge_episodes(
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    state_arrays = []
    valid_action_arrays = []
    action_arrays = []
    action_log_probs = []
    reward_arrays = []
    return_arrays = []
    progress_arrays = []
    next_state_arrays = []
    next_valid_action_arrays = []

    total_len = 0

    for ep in episodes:
        ep_len = ep["actions"].shape[0]
        total_len += ep_len

        state_arrays.append(ep["states"][:-1, ...])
        valid_action_arrays.append(ep["valid_actions"][:-1, ...])

        next_state_arrays.append(ep["states"][1:, ...])
        next_valid_action_arrays.append(ep["valid_actions"][1:, ...])

        action_arrays.append(ep["actions"])
        action_log_probs.append(ep["action_log_probs"])
        reward_arrays.append(ep["rewards"])
        # return_arrays.append(ep["returns"])
        progress_arrays.append(ep["progress"])

    states = np.concatenate(state_arrays)
    valid_actions = np.concatenate(valid_action_arrays)

    next_states = np.concatenate(next_state_arrays)
    next_valid_actions = np.concatenate(next_valid_action_arrays)

    actions = np.concatenate(action_arrays)
    action_log_probs = np.concatenate(action_log_probs)
    rewards = np.concatenate(reward_arrays)
    # returns = np.concatenate(return_arrays)
    progress = np.concatenate(progress_arrays)

    for _idx, _arr in enumerate(
        (
            states,
            valid_actions,
            next_states,
            next_valid_actions,
            actions,
            action_log_probs,
            rewards,
            # returns,
            progress,
        )
    ):
        if _arr.shape[0] != total_len:
            raise RuntimeError(
                f"expects {total_len} rows but shape {_arr.shape} at index {_idx}"
            )

    return {
        "states": states,
        "valid_actions": valid_actions,
        "next_states": next_states,
        "next_valid_actions": next_valid_actions,
        "actions": actions,
        "action_log_probs": action_log_probs,
        "rewards": rewards,
        "progress": progress,
        "size": total_len,
    }


def make_batches_from_data(
    data: dict[str, torch.Tensor],
    batch_size: int,
    *,
    seed: int | None = None,
):
    state = data["state"]
    assert state.ndim == 2 and state.shape[1] == 16, state.shape
    total_size = state.shape[0]

    generator = None
    if seed is not None:
        generator = torch.Generator(state.device)
        generator.manual_seed(seed)

    indices = torch.randperm(
        total_size,
        dtype=torch.long,
        device=state.device,
        generator=generator,
    )

    for start in range(0, total_size, batch_size):
        if start + batch_size > total_size:
            yield_size = total_size - start

            if yield_size < batch_size // 2:
                break
        else:
            yield_size = batch_size

        yield_indices = indices[start : start + yield_size]
        batch = {key: value[yield_indices, ...] for key, value in data.items()}
        yield batch


def make_batches_from_tran_dict(
    trans_dict: dict[str, Any],
    batch_size: int,
    seed: int | None = None,
):
    size = trans_dict["size"]
    indices = np.arange(size, dtype=np.int64)
    rand = np.random.default_rng(seed)
    rand.shuffle(indices)

    fields = (
        "states",
        "valid_actions",
        "next_states",
        "next_valid_actions",
        "actions",
        "action_log_probs",
        "rewards",
        "progress",
    )

    for start in range(0, size, batch_size):
        if start + batch_size > size:
            yield_size = size - start

            if yield_size < batch_size // 2:
                break
        else:
            yield_size = batch_size

        yield_indices = indices[start : start + yield_size]

        batch = {name: trans_dict[name][yield_indices, ...] for name in fields}
        batch = make_batch(batch)
        yield batch


def make_batches(
    episodes: list[dict[str, Any]],
    batch_size: int,
    *,
    cycle: int = 1,
    seed: int | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Output: (state, action, action_logit, reward, next_state)
    """

    state_arrays = []
    valid_action_arrays = []
    action_arrays = []
    action_log_probs = []
    reward_arrays = []
    return_arrays = []
    progress_arrays = []
    next_state_arrays = []
    next_valid_action_arrays = []

    total_len = 0

    for ep in episodes:
        ep_len = ep["actions"].shape[0]
        total_len += ep_len

        state_arrays.append(ep["states"][:-1, ...])
        valid_action_arrays.append(ep["valid_actions"][:-1, ...])

        next_state_arrays.append(ep["states"][1:, ...])
        next_valid_action_arrays.append(ep["valid_actions"][1:, ...])

        action_arrays.append(ep["actions"])
        action_log_probs.append(ep["action_log_probs"])
        reward_arrays.append(ep["rewards"])
        # return_arrays.append(ep["returns"])
        progress_arrays.append(ep["progress"])

    states = np.concatenate(state_arrays)
    valid_actions = np.concatenate(valid_action_arrays)

    next_states = np.concatenate(next_state_arrays)
    next_valid_actions = np.concatenate(next_valid_action_arrays)

    actions = np.concatenate(action_arrays)
    action_log_probs = np.concatenate(action_log_probs)
    rewards = np.concatenate(reward_arrays)
    # returns = np.concatenate(return_arrays)
    progress = np.concatenate(progress_arrays)

    for _idx, _arr in enumerate(
        (
            states,
            valid_actions,
            next_states,
            next_valid_actions,
            actions,
            action_log_probs,
            rewards,
            # returns,
            progress,
        )
    ):
        if _arr.shape[0] != total_len:
            raise RuntimeError(
                f"expects {total_len} rows but shape {_arr.shape} at index {_idx}"
            )

    indices = np.arange(len(states), dtype=np.intp)
    rand = np.random.default_rng(seed)

    for _ in range(cycle):
        rand.shuffle(indices)

        for start in range(0, total_len, batch_size):
            if start + batch_size > total_len:
                yield_size = total_len - start

                if yield_size < batch_size // 2:
                    break
            else:
                yield_size = batch_size

            yield_indices = indices[start : start + yield_size]

            batch = {
                "states": states[yield_indices, :],
                "valid_actions": valid_actions[yield_indices, :],
                "next_states": next_states[yield_indices, :],
                "next_valid_actions": next_valid_actions[yield_indices, :],
                "actions": actions[yield_indices],
                "action_log_probs": action_log_probs[yield_indices],
                "rewards": rewards[yield_indices],
                # "returns": returns[yield_indices],
                "progress": progress[yield_indices],
            }
            batch = make_batch(batch)

            yield batch


def _test_make_batches():
    episodes = 2
    batch_size = 32

    policy = Policy()
    episodes = collect_experience(policy, episodes, 100, batch_size * 2, gamma=0.95)
    for batch in make_batches(episodes, 32, seed=12345):
        assert "states" in batch
        assert "next_states" in batch
        assert "actions" in batch
        assert "action_logits" in batch
        assert "rewards" in batch
        # assert "returns" in batch


class VecReplayBuffer:
    def __init__(
        self,
        capacity: int,
        game_count: int,
        *,
        buffers: dict[str, torch.Tensor] | None = None,
    ):
        assert capacity >= 1, capacity
        assert game_count >= 1, game_count

        self.capacity = capacity
        self.game_count = game_count

        self.offset = 0
        self.length = 0

        batch_shape = (capacity, game_count)

        if buffers is None:
            buffers = new_tensors(REPLAY_SPEC, batch_shape)
        else:
            check_tensors(REPLAY_SPEC, batch_shape, buffers)

        self.buffers: ReplayBufferDict = cast(ReplayBufferDict, buffers)

    def reset(self, zero_: bool | None = None):
        self.offset = 0
        self.length = 0

        if zero_:
            for key in REPLAY_SPEC.keys():
                self.buffers[key].zero_()

    def consume(self, size: int):
        assert self.length >= size, (self.offset, self.length, self.capacity)
        self.length -= size

        if self.length == 0:
            self.offset = 0
        else:
            self.offset += size

    def _rebase(self):
        """
        Move data along the first dimension

        TODO: Performance test against temporary clone and assign.
        """
        if self.offset == 0:
            return

        if self.length >= 1:
            original_offset = self.offset
            original_length = self.length

            for name, tensor in self.buffers.items():
                max_size = min(original_offset, original_length)
                read_offset = original_offset
                write_offset = 0
                done = 0

                while done < original_length:
                    size = min(max_size, original_length - done)
                    src = tensor[read_offset : read_offset + size, ...]
                    dst = tensor[write_offset : write_offset + size, ...]
                    dst[...] = src
                    read_offset += size
                    write_offset += size
                    done += size

        self.offset = 0

    def _try_rebase(self, required_size: int):
        remaining = self.capacity - self.length - self.offset

        if required_size > remaining:
            self._rebase()

    def on_stepped(
        self,
        game: VecGame,
        result: VecStepResult,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
    ):
        self._try_rebase(1)

        index = self.offset + self.length
        assert index < self.capacity, (self.offset, self.length, self.capacity)

        def copy(name: str, src: np.ndarray, dtype: torch.dtype | None = None):
            src_tensor = torch.from_numpy(src).to(dtype=dtype)
            dst = self.buffers[name]
            dst[index, ...].copy_(src_tensor)

        copy("state", result["prev_state"], torch.long)
        copy("valid_actions", result["prev_valid_actions"], torch.bool)
        copy("next_state", result["state"], torch.long)
        copy("next_valid_actions", result["valid_actions"], torch.bool)
        copy("reward", result["reward"], torch.float32)
        copy("terminated", result["terminated"], torch.bool)

        self.buffers["action"][index, ...].copy_(actions.detach())
        self.buffers["action_log_prob"][index, ...].copy_(action_log_probs.detach())
        self.length += 1


def collect_experience(
    runner: EpisodeRunner,
    policy: Policy,
    min_games: int,
    min_steps: int,
    *,
    gamma: float,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    env = Game(seed)

    # (state, action, action_prob, reward, next_state)
    episodes = []
    total = 0

    while len(episodes) < min_games or total < min_steps:
        env.reset()

        rows = runner.run_once(env, policy)
        ep = compile_episode(rows, gamma=gamma)

        episodes.append(ep)
        total += ep["steps"]

    return episodes


_SEGMENT_DTYPE = np.dtype(
    [
        ("state", np.int8, (16,)),
        ("step", np.int32, ()),
        ("action", np.int8, ()),
        # 3 bytes gap here due to alignment
        ("log_prob", np.float32, ()),
    ],
    align=True,
)


@dataclasses.dataclass
class RecordBuffer:
    id: int
    steps: int
    terminated: bool
    maxcell: int = None
    score: float | None = None

    write_index: int = dataclasses.field(repr=False, default=0)
    # (state, action, score) tuples
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = dataclasses.field(
        repr=False,
        default_factory=list,
    )

    def update_stats(self):
        idx = self.write_index
        assert idx >= 1, (len(self.segments), idx)
        idx -= 1

        last_segment = self.segments[-1]
        maxcell = last_segment[0][idx, :].max().item()
        score = last_segment[2][idx].item()
        self.maxcell = maxcell
        self.score = score

    def contiguous_result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        segments = self.segments.copy()
        segment_sizes = []

        for i in range(len(segments) - 1):
            segment_sizes.append(segments[i][0].shape[0])
        segment_sizes.append(self.write_index)
        size = sum(segment_sizes)
        assert size == self.steps + 1, (size, self.steps)

        res_state = np.zeros((size, 16), dtype=np.int8)
        res_action = np.zeros((size,), dtype=np.int8)
        res_score = np.zeros((size,), dtype=np.float32)

        offset = 0
        for (state, action, score), segment_size in zip(segments, segment_sizes):
            res_state[offset : offset + segment_size, :] = state[0:segment_size, :]
            res_action[offset : offset + segment_size] = action[0:segment_size]
            res_score[offset : offset + segment_size] = score[0:segment_size]

            offset += segment_size

        return res_state, res_action, res_score


class ReplayRecorder:
    def __init__(
        self,
        ready_threshold: int,
        recording_threshold: int,
        *,
        segment_size: int = 1024,
    ):
        assert segment_size >= 2, segment_size

        # size of one pre-allocated segment
        self.segment_size = segment_size

        # refuse new recording if ready buffers contain too many items
        self.ready_threshold = ready_threshold

        # recording at most N games at once
        self.recording_threshold = recording_threshold

        # buffers storing completed games
        self.ready_buffers = collections.deque[RecordBuffer]()

        # mapping game_id to (slot_id, ReplayBuffer) for games being recorded
        self._recording: dict[int, tuple[int, RecordBuffer]] = {}

    def on_prepared(self, game: VecGame, new_indices: np.ndarray):
        """VecRunner callback"""

        if len(self.ready_buffers) >= self.ready_threshold:
            # refuse recording new games if ready queue contains enough item.
            return

        free_slots = min(
            new_indices.size,
            self.recording_threshold - len(self._recording),
        )

        data = game._data

        for i in range(free_slots):
            slot_id = new_indices[i].item()
            game_id = data[slot_id]["id"].item()
            buffer = RecordBuffer(
                id=game_id,
                steps=0,
                terminated=False,
                write_index=0,
                segments=[self._new_segment()],
            )
            self._recording[game_id] = (slot_id, buffer)

    def on_stepped(
        self,
        game: VecGame,
        result: VecStepResult,
        action_vec: torch.Tensor,
        action_log_probs: torch.Tensor,
    ):
        """VecRunner callback"""

        prev_state_vec = result["prev_state"]
        state_vec = result["state"]
        score_vec = result["score"]
        terminated = result["terminated"]
        # transfer to CPU domain
        action_vec = action_vec.cpu().numpy()

        completed = []
        for game_id, (slot_id, buffer) in self._recording.items():
            buffer.steps += 1

            self._append_row(
                buffer,
                prev_state_vec[slot_id, :],
                action_vec[slot_id],
                score_vec[slot_id],
            )

            if terminated[slot_id]:
                self._append_row(
                    buffer,
                    state_vec[slot_id, :],
                    0,
                    score_vec[slot_id],
                )
                buffer.terminated = True
                completed.append(buffer)

        for buffer in completed:
            buffer.update_stats()
            self._recording.pop(buffer.id)
            self.ready_buffers.append(buffer)

    def _new_segment(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((self.segment_size, 16), dtype=np.int8),
            np.zeros((self.segment_size,), dtype=np.int8),
            np.zeros((self.segment_size,), dtype=np.float32),
        )

    def _append_row(
        self,
        buffer: RecordBuffer,
        state: np.ndarray,
        action: int,
        score: float,
    ):
        idx = buffer.write_index

        if idx >= self.segment_size:
            segment = self._new_segment()
            buffer.segments.append(segment)
            idx = buffer.write_index = 0
        else:
            segment = buffer.segments[-1]

        seg_state, seg_action, seg_score = segment

        seg_state[idx, :] = state
        seg_action[idx] = action
        seg_score[idx] = score

        buffer.write_index += 1
