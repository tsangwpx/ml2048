from typing import Any, Callable, cast

import torch
import torch.nn as nn

from ml2048.policy import (
    Policy,
    ProbabilisticPolicy,
    SupportsEvalValue,
    TrainablePolicy,
)
from ml2048.policy._network import CNNActorNetwork, CNNCriticNetwork, CNNEncoder
from ml2048.replay import make_batches_from_data
from ml2048.stats import TensorStats, categorical_sample
from ml2048.util import convert_tensors

TRAIN_SPEC = {
    "state": ((16,), torch.long),
    "valid_actions": ((4,), torch.bool),
    "action": ((), torch.int8),
    "action_log_prob": ((), torch.float32),
    "reward": ((), torch.float32),
    "adv": ((), torch.float32),
    "next_state": ((16,), torch.long),
    "next_valid_actions": ((4,), torch.bool),
    "terminated": ((), torch.bool),
    "step": ((), torch.float32),
}


def masked_entropy_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Return entropy of logits with a mask.

    The probability of masked item (mask=0) is 0 and,
    thus, contribute nothing to the total entropy.
    """
    min_real = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, min_real)

    # Use torch's Categorical to normalize logits, probs, and validate args
    dist = torch.distributions.Categorical(logits=masked_logits)

    # p * log p
    p_lop_p = dist.probs * torch.clamp(dist.logits, min=min_real)

    # mask invalid entries
    p_lop_p = torch.where(mask, p_lop_p, 0)

    return -p_lop_p.sum(-1)


def _sample_action(
    actor: nn.Module,
    state: torch.Tensor,
    valid_actions: torch.BoolTensor,
    generator=None,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Sample actions from actor model

    Return a tuple of action indices and their log probs.
    """

    with torch.no_grad():
        logits = actor(state, valid_actions)
        min_real = torch.finfo(logits.dtype).min
        logits = torch.where(valid_actions, logits, min_real)
        dist = torch.distributions.Categorical(logits=logits)
        actions = categorical_sample(dist, generator=generator)
        log_probs = dist.log_prob(actions)

    return cast(tuple[torch.LongTensor, torch.FloatTensor], (actions, log_probs))


class BaseActorCriticPolicy(
    ProbabilisticPolicy,
    SupportsEvalValue,
    TrainablePolicy,
    Policy,
):
    """
    Implement the computations of actor loss and critic loss.
    """

    _critic: nn.Module
    _actor: nn.Module

    def _actor_logits(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def action_logits(
        self,
        state: torch.LongTensor | torch.ByteTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        return self._actor_logits(state, valid_actions)

    def _critic_value(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError

    def _compute_actor_ppo_adv_loss(
        self,
        states: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        actions: torch.LongTensor,
        action_log_probs: torch.FloatTensor,
        adv: torch.FloatTensor,
        *,
        step: torch.FloatTensor,
        epsilon: float,
        entropy_coef: float,
        tensor_stats: TensorStats,
    ):
        """
        Given advantage values, compute the actor loss with PPO, and the entropy loss
        """

        logits = self._actor_logits(states, valid_actions)

        dist = torch.distributions.Categorical(
            logits=logits + torch.where(valid_actions, 0, -10.0e5),
        )
        log_probs = dist.log_prob(actions)

        tensor_stats.update("adv0", adv)

        if False:
            # normalize with the batch mean
            adv_std, adv_mean = torch.std_mean(adv, correction=0)
            adv = (adv - adv_mean) / adv_std / 3
        elif True:
            # normalize with mean 0
            adv_std3 = torch.sqrt(torch.sum(torch.square(adv)) / torch.numel(adv)) * 3
            adv = adv / adv_std3
        else:
            raise AssertionError

        # now adv is usually between [-1, 1]
        # tanh(x) * sqrt(x + c) behave like y=x around [-1, 1] and y=sqrt(x) otherwise
        # so adv is regularized and mostly between [-10, 10]
        # the magic number is computed by minimizing the error in [-1, 1]
        adv = torch.tanh(adv) * torch.sqrt(torch.abs(adv) + 0.6917418778812134)
        tensor_stats.update("adv", adv)

        tensor_stats.update("step", step)
        step_std, step_mean = torch.std_mean(step)
        step95 = step_mean + step_std * 2

        ratio = torch.exp(log_probs - action_log_probs)
        clipped = torch.clip(ratio, 1 - epsilon, 1 + epsilon)
        policy_loss = torch.minimum(
            ratio * adv,
            clipped * adv,
        )
        tensor_stats.update("policy_loss", policy_loss)

        entropy = masked_entropy_from_logits(logits, valid_actions)
        tensor_stats.update("entropy", entropy)

        # coef2 = torch.maximum(
        #     (1 + entropy_min0 / entropy_max0) - entropy.detach() / entropy_max0,
        #     1 - adv.abs(),
        # )
        # tensor_stats.update("coef2", coef2)
        # coef2 = torch.clip(coef2, 0, 1)

        step_z = (step - step_mean) / step_std

        # entropy_coef2 = torch.clip(step / step95, 0.1, 1)
        entropy_c2 = (torch.tanh(step_z * 2 - 1) + 1) * (0.5 * 0.8) + 0.2
        # entropy_c2 = 1
        # entropy_c3 = entropy_coef * torch.clip(64 / step_mean, None, 1)
        entropy_c3 = entropy_coef

        entropy = entropy_c3 * entropy_c2 * entropy
        tensor_stats.update("entropy2", entropy)

        policy_loss = -torch.sum(policy_loss)
        entropy_loss = -torch.sum(entropy)

        return policy_loss, entropy_loss

    def _compute_critic_loss(
        self,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        reward: torch.FloatTensor,
        next_state: torch.LongTensor,
        next_valid_actions: torch.BoolTensor,
        terminated: torch.BoolTensor | None,
        *,
        gamma: float,
        critic_coef: float,
        tensor_stats: TensorStats,
    ) -> tuple[torch.Tensor]:
        """
        Compute the critic loss
        """

        v0 = self._critic_value(state, valid_actions)

        # we do not need the v1 gradient because we focus on updating the v0 only
        with torch.no_grad():
            v1 = self._critic_value(next_state, next_valid_actions)

        if terminated is None:
            v1 = torch.where(torch.any(next_valid_actions, dim=-1), v1, 0)
        else:
            v1 = torch.where(terminated, 0, v1)

        q0 = gamma * v1 + reward
        loss = critic_coef * loss_fn(q0, v0)
        return (loss,)

    def learn(
        self,
        params: dict[str, Any],
        data: dict[str, torch.Tensor],
        *,
        tensor_stats: TensorStats,
        device: Any = None,
        seed: int | None = None,
    ):
        raise NotImplementedError


class CNNActorCriticPolicy(BaseActorCriticPolicy):
    def __init__(
        self,
        encoder_features: int = 1024,
        share_encoder: bool = False,
    ):
        super().__init__()

        # load_state_dict() will load the same parameters twice into the same instance
        # But it is fine.

        self.encoder_shared = share_encoder

        if not share_encoder:
            raise ValueError("Only share encoder is tested")

        if share_encoder:
            self._encoder = CNNEncoder(encoder_features)
            self.add_module("actor_encoder", None)
            self.add_module("critic_encoder", None)
        else:
            self._actor_encoder = CNNEncoder(encoder_features)
            self._critic_encoder = CNNEncoder(encoder_features)
            self.add_module("encoder", None)

        self._actor = CNNActorNetwork(encoder_features, 256, 64)
        self._critic = CNNCriticNetwork(encoder_features, 256, 64)
        self._critic_loss_fn = nn.MSELoss(reduction="mean")

    def _actor_logits(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        batch_shape = state.shape[:-1]
        state = torch.reshape(state, (-1, 16))

        if self.encoder_shared:
            x = self._encoder(state)
        else:
            x = self._actor_encoder(state)

        logits = self._actor(x, valid_actions)
        logits = torch.reshape(logits, batch_shape + (4,))
        return logits

    def _critic_value(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        assert state.shape[-1] == 16, state.shape

        batch_shape = state.shape[:-1]
        state = torch.reshape(state, (-1, 16))

        if self.encoder_shared:
            x = self._encoder(state)
        else:
            x = self._critic_encoder(state)
        value = self._critic(x, valid_actions)
        value = torch.reshape(value, batch_shape)
        return value.float()

    def eval_value(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ) -> torch.FloatTensor:
        return self._critic_value(state, valid_actions)

    def sample_actions(
        self,
        state: torch.LongTensor | torch.ByteTensor,
        valid_actions: torch.BoolTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        if self.encoder_shared:
            x = self._encoder(state)
        else:
            x = self._actor_encoder(state)

        return _sample_action(self._actor, x, valid_actions, generator=generator)

    def _learn_shared(
        self,
        params: dict[str, Any],
        data: dict[str, torch.Tensor],
        *,
        tensor_stats: TensorStats,
        device: Any = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        assert self.encoder_shared

        gamma = params["gamma"]
        epsilon = params["ppo_epsilon"]

        actor_lr = params["actor_lr"]
        critic_lr = params["actor_lr"]
        batch_size = params["actor_batch_size"]

        entropy_coef = params["entropy_coef"]
        critic_coef = params.get("critic_coef", 1.0)

        params = [
            {
                "params": self._encoder.parameters(),
                "lr": min(actor_lr, critic_lr),
            },
            {
                "params": self._actor.parameters(),
                "lr": actor_lr,
            },
            {
                "params": self._critic.parameters(),
                "lr": critic_lr,
            },
        ]

        optimizer = torch.optim.Adam(
            params,
            lr=min(actor_lr, critic_lr),
        )

        keys = {
            "state",
            "valid_actions",
            "action",
            "action_log_prob",
            "reward",
            "next_state",
            "next_valid_actions",
            "adv",
            "terminated",
            "step",
        }
        spec = {k: TRAIN_SPEC[k] for k in keys}

        losses = []

        for idx, batch in enumerate(
            make_batches_from_data(data, batch_size, seed=seed)
        ):
            batch = convert_tensors(spec, batch, device=device)

            state = cast(torch.LongTensor, batch["state"])
            valid_actions = cast(torch.BoolTensor, batch["valid_actions"])
            action = cast(torch.LongTensor, batch["action"])
            action_log_prob = cast(torch.FloatTensor, batch["action_log_prob"])
            reward = cast(torch.FloatTensor, batch["reward"])
            next_state = cast(torch.LongTensor, batch["next_state"])
            next_valid_actions = cast(torch.BoolTensor, batch["next_valid_actions"])
            adv = cast(torch.FloatTensor, batch["adv"])
            terminated = cast(torch.BoolTensor, batch["terminated"])
            step = cast(torch.FloatTensor, batch["step"])

            policy_loss, entropy_loss = self._compute_actor_ppo_adv_loss(
                state,
                valid_actions,
                action,
                action_log_prob,
                adv,
                step=step,
                tensor_stats=tensor_stats,
                epsilon=epsilon,
                entropy_coef=entropy_coef,
            )

            (critic_loss,) = self._compute_critic_loss(
                self._critic_loss_fn,
                state,
                valid_actions,
                reward,
                next_state,
                next_valid_actions,
                terminated,
                tensor_stats=tensor_stats,
                gamma=gamma,
                critic_coef=critic_coef,
            )

            optimizer.zero_grad()
            loss = policy_loss + entropy_loss + critic_loss
            loss.backward()
            optimizer.step()

            losses.append(
                (
                    policy_loss.detach(),
                    entropy_loss.detach(),
                    critic_loss.detach(),
                )
            )

        losses = torch.tensor(losses, device="cpu")
        losses = torch.mean(losses, dim=0)

        return {
            "policy_loss": losses[0],
            "entropy_loss": losses[1],
            "critic_losses": losses[2:3],
        }

    def _learn_separated(
        self,
        params: dict[str, Any],
        data: dict[str, torch.Tensor],
        *,
        tensor_stats: TensorStats,
        device: Any = None,
        seed: int | None = None,
    ):
        raise NotImplementedError

    def learn(
        self,
        params: dict[str, Any],
        data: dict[str, torch.Tensor],
        *,
        tensor_stats: TensorStats,
        device: Any = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        fn = self._learn_shared if self.encoder_shared else self._learn_separated

        return fn(
            params,
            data,
            tensor_stats=tensor_stats,
            device=device,
            seed=seed,
        )
