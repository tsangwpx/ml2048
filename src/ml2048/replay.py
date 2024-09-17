import collections
import dataclasses

import numpy as np
import torch

from ml2048.game_numba import VecGame, VecStepResult

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
