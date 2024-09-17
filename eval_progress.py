"""
Script to generate data for plotting max tile distribution graph
"""

import dataclasses
import os
import pickle
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict, deque
from functools import partial
from pathlib import Path

import numba
import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml2048.game_numba import VecGame
from ml2048.policy.actor_critic import CNNActorCriticPolicy
from ml2048.replay import ReplayRecorder
from ml2048.runner import VecRunner

_SAVED_MODELS = [
    ("ml2048_20240324_214853", range(0, 9000 + 1, 100)),
    # 9k
    ("ml2048_20240325_060022", range(100, 9000 + 1, 100)),
    # 18k
    ("ml2048_20240325_233133", range(100, 9000 + 1, 100)),
    # 27k
    ("ml2048_20240326_035345", range(100, 9000 + 1, 100)),
    # 36k
    ("ml2048_20240326_182405", range(100, 13000 + 1, 100)),
    # 49k
    ("ml2048_20240327_161822", range(100, 10000 + 1, 100)),
    # 59k
    ("ml2048_20240328_045929", range(100, 10000 + 1, 100)),
    # 69k
    ("ml2048_20240329_192245", range(100, 2500 + 1, 100)),
    ("ml2048_20240329_214739", range(100, 4000 + 1, 100)),
    ("ml2048_20240330_013340", range(100, 2500 + 1, 100)),
    # 78k
]
_TARGETS = [
    (run_id, epoch) for run_id, epoch_range in _SAVED_MODELS for epoch in epoch_range
]
_CACHE_DIR = Path(__file__).parent.joinpath(".cache")


def parser():
    p = ArgumentParser()
    p.add_argument("--most-likely", action="store_true", default=False)
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--device", default=None)
    p.add_argument("save", type=str)
    return p


@dataclasses.dataclass
class StatEntry:
    count: int = 0
    score_sum: float = 0
    step_sum: int = 0


def _remote_execute(fn):
    return fn()


def compute_stats(
    run_id: str,
    epoch: int,
    batch_size: int,
    rounds: int,
) -> list[StatEntry]:
    key = f"{run_id}-{epoch}"
    cache_path = _CACHE_DIR.joinpath(f"{key}-{rounds}.pickle")
    buffer_path = _CACHE_DIR.joinpath(f"{key}-buffer.pickle")

    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    save_path = Path(f"runs/{run_id}/epoch-{epoch}.pt")

    device = torch.device("cuda:0")

    saved_dict = torch.load(save_path)
    policy = CNNActorCriticPolicy(share_encoder=True)
    policy = policy.to(device=device)

    policy.load_state_dict(saved_dict["policy_state"])

    game = VecGame(batch_size)
    recorder = ReplayRecorder(batch_size, batch_size)
    runner = VecRunner(game, batch_size, sample_device=device)
    runner.add_callback(VecRunner.EVENT_PREPARED, recorder.on_prepared)
    runner.add_callback(VecRunner.EVENT_STEPPED, recorder.on_stepped)

    remaining = rounds

    result = [StatEntry() for _ in range(16)]
    buffer_table = defaultdict(lambda: deque(maxlen=64))

    while remaining > 0:
        runner.step_once(policy)

        while recorder.ready_buffers and remaining > 0:
            buffer = recorder.ready_buffers.popleft()
            if buffer.id >= rounds:
                # stop the recorder accept new recording
                recorder.recording_threshold = 0
                continue

            remaining -= 1
            stat_entry = result[buffer.maxcell]
            stat_entry.count += 1
            stat_entry.step_sum += buffer.steps
            stat_entry.score_sum += buffer.score

            if buffer.maxcell >= 13:
                buffer_table[int(buffer.maxcell)].append(buffer)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(result))

        if buffer_table:
            buffer_table = {key: list(value) for key, value in buffer_table.items()}
            buffer_path.parent.mkdir(parents=True, exist_ok=True)
            buffer_path.write_bytes(pickle.dumps(buffer_table))
    except IOError as ex:
        print("IOError: ", key, ex)
    except BaseException:
        print("BaseException", key)
        raise

    return result


def _process_init(threads: int):
    numba.set_num_threads(threads)


def main():
    import multiprocessing.pool

    pool: multiprocessing.pool.Pool

    batch_size = 512
    rounds = 1000
    workers = 2
    threads = max(1, os.cpu_count() // workers // 2)

    fns = [
        partial(compute_stats, run_id, epoch, batch_size, rounds)
        for run_id, epoch in _TARGETS
    ]

    t0 = time.perf_counter()
    with mp.Pool(2, initializer=_process_init, initargs=(threads,)) as pool:
        stats = pool.map(_remote_execute, fns, chunksize=1)

    t1 = time.perf_counter()
    print(f"Completed in {t1 - t0:.3f}")

    length = len(fns)
    data = np.zeros((length, 16), dtype=np.float32)
    for i, entry_list in enumerate(stats):
        for j, stat_entry in enumerate(entry_list):
            data[i, j] = stat_entry.count

    Path("progress.dat").write_bytes(
        pickle.dumps(
            {
                "data": data,
            }
        )
    )


if __name__ == "__main__":
    main()
