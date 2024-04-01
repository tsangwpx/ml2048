"""
Legacy script to train the model with multiprocessing.

Episode-based training:
In each epoch, a fixed amount of games is simulated.

However, the eval time is affected by the game steps and evaluate
each game individually is slow even though they are parallelly collected
with multiprocessing pool.

"""

import collections
import contextlib
import logging
import math
import multiprocessing.pool
import os.path
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml2048.policy import Policy, TrainablePolicy
from ml2048.policy.actor_critic import ActorCriticPolicy
from ml2048.replay import collect_experience, episode_summary
from ml2048.runner import EpisodeRunner
from ml2048.stats import TensorStats
from ml2048.trainer import BaseTrainer


class Trainer(BaseTrainer):
    _pool: multiprocessing.pool.Pool

    def __init__(
        self,
        *,
        save_dir: Path,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            save_dir=save_dir,
            logger=logger,
        )

        self._workers = 10

        # hyper parameters
        self._gamma = 0.997
        self._epsilon = 0.15  # PPO clip epsilon

        self._maxcell = 0

        self._epoches = 10_000
        self._batch_size = 1024
        self._min_games = 256
        self._min_steps = 32768

        # mini config
        # self._min_games = workers
        # self._min_steps = 1

        self._epoches_per_save: int | None = 20

        self._reuse_times = 3
        self._hist_episodes = collections.deque(maxlen=self._reuse_times)

        self._rolling_stats_batches = int(
            (self._min_steps / self._batch_size) * (self._reuse_times + 2)
        )

        self._epoch = 0

    def _collect_episodes(
        self,
        policy: Policy,
        min_games: int,
        min_steps: int,
    ) -> list[dict[str, Any]]:
        runner = EpisodeRunner(
            0,
            0.0,
        )

        task_games = ceildiv(min_games, self._workers)
        task_steps = ceildiv(min_steps, self._workers)

        eval_results = []

        for _ in range(self._workers):
            res = self._pool.apply_async(
                collect_experience,
                (runner, policy, task_games, task_steps),
                {
                    "gamma": self._gamma,
                },
            )
            eval_results.append(res)

        eval_episodes = []
        for result in eval_results:
            eval_episodes.extend(result.get())

        sum_entries = [
            f"({s.maxcell}, {s.count}, {s.scores:.1f}, {s.steps:.1f})"
            for s in episode_summary(eval_episodes)
        ]
        self.print(f"eval {len(eval_episodes)}:", ", ".join(sum_entries))

        result = list(eval_episodes)
        for old_episodes in self._hist_episodes:
            result.extend(old_episodes)

        self._hist_episodes.append(eval_episodes)
        return result

    def run_inner(
        self,
    ):
        def policy_factory():
            return ActorCriticPolicy()

        params = {
            "gamma": self._gamma,
            "epsilon": self._epsilon,
            "actor_lr": 5.0e-4,
            "critic_lr": 1.0e-3,
            "entropy_coef": 0.0003,
        }
        tensor_stats = TensorStats()

        policy_cuda = policy_factory()
        policy_cuda = policy_cuda.cuda()
        policy_cpu = policy_factory()

        for epoch in range(self._epoches):
            self.print(f"epoch {epoch}")
            self._epoch = epoch

            policy_cpu.load_state_dict(policy_cuda.state_dict())
            policy_cpu.eval()

            t0 = time.perf_counter()

            episodes = self._collect_episodes(
                policy_cpu,
                min_games=self._min_games,
                min_steps=self._min_steps,
            )

            t1 = time.perf_counter()

            if isinstance(policy_cuda, TrainablePolicy):
                params_copy = params.copy()

                # f(x) = 1 / sqrt(x + c ** 2) is decreasing slower than f(x) = 1 / sqrt(x)
                # which also fix issue with x = 0.
                # fix f(0) = 1 by multiplying f(x) with c
                params_copy["actor_lr"] = (
                    params["actor_lr"] / math.sqrt(epoch + 100) * 10
                )
                params_copy["critic_lr"] = (
                    params["critic_lr"] / math.sqrt(epoch + 100) * 10
                )

                policy_cuda.train()
                res = policy_cuda.feed_episodes(
                    episodes=episodes,
                    params=params_copy,
                    batch_size=self._batch_size,
                    device=torch.device("cuda:0"),
                    tensor_stats=tensor_stats,
                )
                losses = res["losses"].sum(dim=0)
                self.print(f"train: games={len(episodes)}, loss={losses}")

            t2 = time.perf_counter()

            self.print(f"profiling {t1 - t0:.3f}, {t2 - t1:.3f}")
            for name, stats in tensor_stats.table.items():
                self.print(name, stats)

            self.print()

            if (
                self._epoches_per_save is not None
                and epoch % self._epoches_per_save == 0
            ):
                state = {
                    "policy_cuda": policy_cuda.state_dict(),
                }
                self.save_state(
                    f"epoch-{epoch}.pt",
                    state,
                )

    def run(self):
        exit_stack = contextlib.ExitStack()

        with exit_stack:
            pool = mp.Pool(self._workers, initializer=_set_num_threads, initargs=(1,))
            exit_stack.enter_context(pool)

            self._pool = pool
            self.run_inner()


def _set_num_threads(nthreads: int):
    torch.set_num_threads(nthreads)


def ceildiv(a, b) -> int:
    return -(a // -b)


if __name__ == "__main__":
    Trainer.main()
