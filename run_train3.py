"""
Train the model without multiprocessing.

This improves the performance by vectorized game environment
and perform a fixed amount of steps in each epoch.
(compared to run_train.py)

"""

import contextlib
import logging
import math
import os.path
import sys
import time
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ml2048.gae import compute_gae
from ml2048.game_numba import VecGame, VecStepResult, reward_fn_improved
from ml2048.policy import Policy
from ml2048.policy.actor_critic import BaseActorCriticPolicy, CNNActorCriticPolicy
from ml2048.replay import REPLAY_SPEC
from ml2048.runner import RunnerStats, VecRunner
from ml2048.stats import TensorStats
from ml2048.trainer import BaseTrainer
from ml2048.util import new_tensors


def policy_factory():
    # return ActorCriticPolicy()
    return CNNActorCriticPolicy(
        share_encoder=True,
    )


ADV_SPEC = {
    "adv": ((), torch.float32),
}


class Trainer(BaseTrainer):
    def __init__(
        self,
        arguments: dict[str, Any],
        *,
        save_dir: Path,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            arguments,
            save_dir=save_dir,
            logger=logger,
        )

        batch_size = 1024
        lr_factor = 1 / 2**4

        # hyper parameters
        self._params_default = {
            "lr_factor": lr_factor,
            "gamma": 0.997,
            "lambda": 0.9,
            "ppo_epsilon": 0.1,  # PPO clip epsilon
            "actor_lr": 4.0e-4 * lr_factor,
            "critic_lr": 1.0e-3 * lr_factor,
            "actor_batch_size": batch_size,
            "critic_batch_size": batch_size * 2,
            "entropy_coef": 0.00025,
            "entropy_period": 50,
            "critic_coef": 1e-5 / 2**10,
        }

        self._tensor_stats = TensorStats()

        self._use_count = 2
        self._step_count = 16
        self._game_count = 4096

        self._eval_device = "cuda:0"
        self._train_device = "cuda:0"

        self._epoches_per_save: int | None = 50

        self._game = VecGame(
            self._game_count,
            reward_fn=reward_fn_improved,
            # reward_fn=reward_fn_normal,
            # reward_fn=reward_fn_rank,
            # reward_fn=reward_fn_maxcell,
        )

        self._runner = VecRunner(
            self._game,
            self._step_count,
            sample_device=self._eval_device,
        )

        self._terminated_stats = RunnerStats()
        self._runner.add_callback(
            VecRunner.EVENT_STEPPED,
            self._terminated_stats.on_stepped,
        )

        # preallocate buffers for storing results
        batch_shape = (
            self._use_count,
            self._step_count,
            self._game_count,
        )

        self._buffers = new_tensors(
            REPLAY_SPEC | ADV_SPEC,
            batch_shape,
            device="cuda",
        )
        self._buffer_step = 0

        def on_stepped(
            game: VecGame,
            result: VecStepResult,
            actions: torch.Tensor,
            action_log_probs: torch.Tensor,
        ):
            # use index
            ui = self._epoch % self._use_count

            # step index
            si = self._buffer_step
            self._buffer_step += 1

            def copy(name: str, src: np.ndarray, dtype: torch.dtype | None = None):
                src_tensor = torch.from_numpy(src).to(dtype=dtype)
                dst = self._buffers[name]
                dst[ui, si, ...].copy_(src_tensor)

            copy("state", result["prev_state"], torch.int8)
            copy("valid_actions", result["prev_valid_actions"], torch.bool)
            copy("next_state", result["state"], torch.int8)
            copy("next_valid_actions", result["valid_actions"], torch.bool)
            copy("reward", result["reward"], torch.float32)
            copy("terminated", result["terminated"], torch.bool)
            copy("step", result["step"], torch.int32)

            # VecRunner should step without grad? detach it anyway.
            self._buffers["action"][ui, si, ...].copy_(actions.detach())
            self._buffers["action_log_prob"][ui, si, ...].copy_(
                action_log_probs.detach()
            )

        self._runner.add_callback(VecRunner.EVENT_STEPPED, on_stepped)

    def _epoch_params(self, epoch: int) -> dict[str, Any]:
        """returns params based on epoch"""
        params = self._params_default.copy()

        params["epoch"] = epoch
        params["epoches"] = self._epoches

        # SUM(lr) = inf and SUM(lr**2) -> 0
        params["actor_lr"] *= 32 / math.sqrt(1024 + epoch)
        params["critic_lr"] *= 32 / math.sqrt(1024 + epoch)

        # loss coef is used to adjust the gradient magnitude
        # so that lr = 1e-3 is good default
        # params["loss_coef"] = 1 / self._game_count
        params["loss_coef"] = 1

        return params

    def loop_once(
        self,
        epoch: int,
        policy_eval: Policy,
        policy_train: Policy,
    ):
        self._terminated_stats.reset()
        self._buffer_step = 0
        self._runner.step_many(policy_eval, self._step_count)

        desc_entries = [
            f"({maxcell}, {count}, {int(count_per * 100)}%)"
            for maxcell, count, count_per in self._game.summary()[:6]
        ]
        self.print("eval", ", ".join(desc_entries))

        desc_entries = [
            f"({maxcell}, {count}, {int(count_per * 100)}%)"
            for maxcell, count, count_per in self._terminated_stats.summary()
        ]
        self.print("terminated", ", ".join(desc_entries))

        def make_view(key: str):
            obj = self._buffers[key]
            obj = obj[0 : self._epoch + 1, ...]
            return obj

        gae_keys = {
            "state",
            "valid_actions",
            "reward",
            "next_state",
            "next_valid_actions",
            "terminated",
            "adv",
        }

        compute_gae(
            policy_eval,
            {k: make_view(k) for k in gae_keys},
            gamma=self._params_default["gamma"],
            lambda_=self._params_default["lambda"],
            tensor_stats=self._tensor_stats,
        )

        data = {k: torch.flatten(make_view(k), 0, 2) for k in REPLAY_SPEC | ADV_SPEC}

        assert isinstance(policy_train, BaseActorCriticPolicy)

        loss_dict = policy_train.learn(
            self._epoch_params(epoch),
            data,
            tensor_stats=self._tensor_stats,
            device=self._train_device,
        )

        self.print(
            "train",
            f"{loss_dict['policy_loss']:.4e}",
            f"{loss_dict['entropy_loss']:.4e}",
            loss_dict["critic_losses"],
        )

        for name, stats in self._tensor_stats.table.items():
            self.print(f"{name:8s}", stats)
            stats.reset()

    def run_inner(
        self,
    ):
        # loop
        policy_train = policy_factory()
        policy_train = policy_train.train(True)
        policy_train = policy_train.cuda()

        policy_eval = policy_factory()
        policy_eval = policy_eval.train(False)
        policy_eval = policy_eval.cuda()
        policy_eval = policy_eval.share_memory()

        self.print("arguments", pformat(self._arguments))
        self.print("params", pformat(self._params_default))
        self.print("model", policy_train)
        self.print(
            "extra",
            pformat(
                {
                    "use_count": self._use_count,
                    "game_count": self._game_count,
                    "step_count": self._step_count,
                }
            ),
        )

        restart_file = self._arguments.get("restart")
        if restart_file:
            self.print(f"Load policy from {restart_file}")
            state_dict = torch.load(restart_file)["policy_state"]
            policy_train.load_state_dict(state_dict)

        warming_steps = self._arguments.get("warming_steps")
        if warming_steps:
            self.print(f"Warming buffer with {warming_steps} steps")
            policy_eval.load_state_dict(policy_train.state_dict())
            tmp_runner = VecRunner(
                self._game, self._game_count, sample_device=self._eval_device
            )
            tmp_runner.step_many(policy_eval, warming_steps)
            del tmp_runner

        for epoch in self._get_epoch_range():
            self.print(f"epoch {epoch}")
            self._epoch = epoch

            policy_eval.load_state_dict(policy_train.state_dict())

            # Send task to workers
            self.loop_once(epoch, policy_eval, policy_train)

            t0 = time.perf_counter()

            if epoch % self._epoches_per_save == 0:
                self.save_state(
                    f"epoch-{epoch}.pt",
                    {
                        "policy_state": policy_train.state_dict(),
                    },
                )

    def run(self):
        self.print(f"save_dir={self._save_dir}")
        exit_stack = contextlib.ExitStack()

        with exit_stack:
            # exit_stack.enter_context(torch.autograd.detect_anomaly())

            self.run_inner()


if __name__ == "__main__":
    Trainer.main()
