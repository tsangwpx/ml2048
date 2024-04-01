"""
Given a save, run N games to obtain the max tile distribution.

"""

import dataclasses
import os
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict

import numba
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml2048.game_numba import VecGame
from ml2048.policy.actor_critic import CNNActorCriticPolicy
from ml2048.policy.max_likely import MaxLikelyPolicy
from ml2048.replay import ReplayRecorder
from ml2048.runner import VecRunner


def parser():
    p = ArgumentParser()
    p.add_argument("--most-likely", action="store_true", default=False)
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("save", type=str)
    return p


@dataclasses.dataclass
class StatEntry:
    count: int = 0
    score_sum: float = 0
    step_sum: int = 0


class GameStats:
    def __init__(self, game_count: int):
        pass


def main():
    numba.set_num_threads(os.cpu_count() // 2)

    t0 = time.perf_counter()
    p = parser()
    ns = p.parse_args()
    assert ns.batch_size >= 1, ns.batch_size

    device = ns.device
    # device = "cuda:0"

    print(f"Loading {ns.save!r}")
    state = torch.load(ns.save)
    policy = CNNActorCriticPolicy(share_encoder=True).to(device=device)
    policy.load_state_dict(state["policy_state"])

    if ns.most_likely:
        policy = MaxLikelyPolicy(policy)

    rounds = ns.rounds
    batch_size = min(rounds, ns.batch_size)
    game = VecGame(batch_size)
    recorder = ReplayRecorder(batch_size, batch_size)
    runner = VecRunner(game, batch_size, sample_device=device)
    runner.add_callback(VecRunner.EVENT_PREPARED, recorder.on_prepared)
    runner.add_callback(VecRunner.EVENT_STEPPED, recorder.on_stepped)

    stats = defaultdict(StatEntry)
    remaining = rounds
    runner_step = 0

    last_time = time.monotonic()

    while remaining > 0:
        runner.step_once(policy)
        runner_step += 1

        now = time.monotonic()
        if now - last_time >= 60:
            # report progress roughly every minute
            last_time = time.monotonic()
            print(f"Progress: {(rounds - remaining) / rounds:.1%}, steps={runner_step}")

        while recorder.ready_buffers and remaining > 0:
            buffer = recorder.ready_buffers.popleft()
            if buffer.id >= rounds:
                # stop the recorder accept new recording
                recorder.recording_threshold = 0
                continue

            remaining -= 1
            key = 2**buffer.maxcell
            stat_entry = stats[key]
            stat_entry.count += 1
            stat_entry.step_sum += buffer.steps
            stat_entry.score_sum += buffer.score

    total = sum((s.count for s in stats.values()))

    for key, stat_entry in sorted(stats.items(), reverse=True):
        heading = f"{key}:"

        print(
            f"{heading:6s}",
            f"{stat_entry.count / total:5.1%}",
            f"count={stat_entry.count},",
            f"steps={stat_entry.step_sum / stat_entry.count:.3f},",
            f"score={stat_entry.score_sum / stat_entry.count:.3f}",
        )

    t1 = time.perf_counter()
    print(f"Completed in {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    main()
