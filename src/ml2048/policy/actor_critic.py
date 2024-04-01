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


class ActorCriticPolicy(
    BaseActorCriticPolicy,
    SupportsEvalValue,
    TrainablePolicy,
    Policy,
):
    """
    legacy policy implementation used in run_train.py
    """

    def __init__(self):
        super().__init__()

        self._actor = CNNActorNetwork(256, 64)
        self._critic = CNNCriticNetwork(256, 64)
        self._actor._encoder = self._critic._encoder

        self._critic_loss_fn = torch.nn.MSELoss(reduction="sum")

    def sample_actions(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        with torch.no_grad():
            logits = self._actor(state, valid_actions)
            min_real = torch.finfo(logits.dtype).min
            logits = torch.where(valid_actions, logits, min_real)
            dist = torch.distributions.Categorical(logits=logits)
            actions = categorical_sample(dist, generator=generator)
            log_probs = dist.log_prob(actions)

        return actions.long(), log_probs.float()

    def eval_value(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
    ):
        return self._critic(state, valid_actions)

    def learn(
        self,
        params: dict[str, Any],
        data: dict[str, torch.Tensor],
        *,
        tensor_stats: TensorStats,
        device: Any = None,
        seed: int | None = None,
    ):
        gamma = params["gamma"]
        epsilon = params["ppo_epsilon"]

        actor_lr = params["actor_lr"]
        actor_batch_size = params["actor_batch_size"]

        critic_lr = params["actor_lr"]
        critic_batch_size = params["actor_batch_size"]

        entropy_coef = params["entropy_coef"]
        loss_coef = params["loss_coef"]

        actor_optim = torch.optim.Adam(
            self._actor.parameters(),
            lr=actor_lr,
        )
        critic_optim = torch.optim.Adam(
            self._critic.parameters(),
            lr=critic_lr,
        )

        actor_losses = []
        critic_losses = []

        actor_keys = {
            "state",
            "valid_actions",
            "action",
            "action_log_prob",
            "reward",
            # "next_state",
            # "next_valid_actions",
            "adv",
            # "terminated",
            "step",
        }
        actor_spec = {k: TRAIN_SPEC[k] for k in actor_keys}

        for idx, batch in enumerate(
            make_batches_from_data(data, actor_batch_size, seed=seed)
        ):
            batch = convert_tensors(actor_spec, batch, device=device)

            state = cast(torch.LongTensor, batch["state"])
            valid_actions = cast(torch.BoolTensor, batch["valid_actions"])
            action = cast(torch.LongTensor, batch["action"])
            action_log_prob = cast(torch.FloatTensor, batch["action_log_prob"])
            # reward = cast(torch.FloatTensor, batch["reward"])
            # next_state = cast(torch.LongTensor, batch["next_state"])
            # next_valid_actions = cast(torch.BoolTensor, batch["next_valid_actions"])
            adv = cast(torch.FloatTensor, batch["adv"])
            # terminated = cast(torch.BoolTensor, batch["terminated"])
            step = cast(torch.FloatTensor, batch["step"])

            policy_loss, entropy_loss = self._train_actor_adv(
                actor_optim,
                state,
                valid_actions,
                action,
                action_log_prob,
                adv,
                step=step,
                epsilon=epsilon,
                entropy_coef=entropy_coef,
                loss_coef=loss_coef,
                tensor_stats=tensor_stats,
            )

            actor_losses.append((policy_loss, entropy_loss))

        critic_keys = {
            "state",
            "valid_actions",
            "reward",
            "next_state",
            "next_valid_actions",
            "terminated",
        }
        critic_spec = {k: TRAIN_SPEC[k] for k in critic_keys}

        for idx, batch in enumerate(
            make_batches_from_data(data, critic_batch_size, seed=seed)
        ):
            batch = convert_tensors(critic_spec, batch, device=device)

            state = cast(torch.LongTensor, batch["state"])
            valid_actions = cast(torch.BoolTensor, batch["valid_actions"])
            reward = cast(torch.FloatTensor, batch["reward"])
            next_state = cast(torch.LongTensor, batch["next_state"])
            next_valid_actions = cast(torch.BoolTensor, batch["next_valid_actions"])
            terminated = cast(torch.BoolTensor, batch["terminated"])

            (critic_loss,) = self._train_critic(
                critic_optim,
                state,
                valid_actions,
                reward,
                next_state,
                next_valid_actions,
                terminated,
                gamma=gamma,
                tensor_stats=tensor_stats,
            )
            critic_losses.append(critic_loss)

        batch_count = len(actor_losses)
        assert len(critic_losses) % batch_count == 0, (
            len(actor_losses),
            len(critic_losses),
        )

        actor_losses = torch.tensor(actor_losses, device="cpu")
        actor_losses = torch.mean(actor_losses, dim=0)

        critic_losses = torch.tensor(critic_losses, device="cpu")
        critic_losses = torch.reshape(critic_losses, (-1, batch_count))
        critic_losses = torch.mean(critic_losses, dim=1)

        return {
            "policy_loss": actor_losses[0],
            "entropy_loss": actor_losses[1],
            "critic_losses": critic_losses,
        }

    def _train_actor_adv(
        self,
        optimizer: torch.optim.Optimizer,
        states: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        actions: torch.LongTensor,
        action_log_probs: torch.FloatTensor,
        adv: torch.FloatTensor,
        *,
        step: torch.FloatTensor,
        epsilon: float,
        loss_coef: float,
        entropy_coef: float,
        tensor_stats: TensorStats,
    ):
        logits = self._actor(states, valid_actions)

        dist = torch.distributions.Categorical(
            logits=logits + torch.where(valid_actions, 0, -10.0e5),
        )
        log_probs = dist.log_prob(actions)

        tensor_stats.update("adv0", adv)

        if False:
            adv_std, adv_mean = torch.std_mean(adv, correction=0)
            adv = (adv - adv_mean) / adv_std / 3
        elif True or False:
            adv_std3 = torch.sqrt(torch.sum(torch.square(adv)) / torch.numel(adv)) * 3
            adv = adv / adv_std3
        else:
            raise AssertionError

        # adv = torch.clip(adv, -10, 10)
        adv = torch.tanh(adv) * torch.sqrt(torch.abs(adv) + 0.6917418778812134)
        tensor_stats.update("adv", adv)

        importance = True
        ppo = True

        tensor_stats.update("step", step)
        step_std, step_mean = torch.std_mean(step)
        step95 = step_mean + step_std * 2

        if ppo:
            # epsilon2 = torch.clip(step / step95, 0.5, 1) * epsilon
            epsilon2 = epsilon

            ratio = torch.exp(log_probs - action_log_probs)
            clipped = torch.clip(ratio, 1 - epsilon2, 1 + epsilon2)
            policy_loss = torch.minimum(
                ratio * adv,
                clipped * adv,
            )
            tensor_stats.update("policy_loss", policy_loss)
            policy_loss = -torch.sum(policy_loss)
        elif importance:
            ratio = torch.exp(log_probs - action_log_probs)
            tensor_stats.update("ratio", ratio)
            policy_loss = -torch.sum(adv * ratio)
        else:
            policy_loss = -torch.sum(adv * log_probs)

        entropy = masked_entropy_from_logits(logits, valid_actions)
        tensor_stats.update("entropy", entropy)

        # entropy_max0 = 1.3862943611198906  # -log(1/4)
        # entropy_min0 = 0.3250829733914482  # -0.9*log(0.9) - 0.1*log(0.1)

        # coef2 = torch.maximum(
        #     (1 + entropy_min0 / entropy_max0) - entropy.detach() / entropy_max0,
        #     1 - adv.abs(),
        # )
        # tensor_stats.update("coef2", coef2)
        # coef2 = torch.clip(coef2, 0, 1)

        step_z = (step - step_mean) / step_std

        # entropy_coef2 = torch.clip(step / step95, 0.1, 1)
        entropy_coef2 = (torch.tanh(step_z * 2 - 1) + 1) * (0.5 * 0.9) + 0.1

        entropy2 = entropy_coef2 * entropy
        tensor_stats.update("entropy2", entropy2)
        entropy_loss = entropy_coef * -torch.sum(entropy2)

        # entropy_loss = entropy_coef * -torch.sum(entropy)

        actor_loss = policy_loss + entropy_loss

        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

        return policy_loss.detach(), entropy_loss.detach()

    def _train_actor(
        self,
        optimizer: torch.optim.Optimizer,
        states: torch.Tensor,
        valid_actions: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        progress: torch.Tensor,
        next_states: torch.Tensor,
        next_valid_actions: torch.Tensor,
        *,
        gamma: float,
        epsilon: float,
        entropy_coef: float,
        tensor_stats: TensorStats,
    ):
        logits = self._actor(states, valid_actions)
        dist = torch.distributions.Categorical(
            logits=logits + torch.where(valid_actions, 0, -10.0e5),
        )
        log_probs = dist.log_prob(actions)

        with torch.no_grad():
            v0 = self._critic(states)
            v1 = self._critic(next_states)
            assert v0.ndim == 1, v0.shape

        # value(terminal state) = 0
        v1 = torch.where(torch.any(next_valid_actions, dim=1), v1, 0)

        adv = gamma * v1 + rewards - v0
        adv_std, adv_mean = torch.std_mean(adv, correction=0)
        adv = torch.tanh(adv / adv_std / 3) * 3

        tensor_stats.update("adv", adv)

        # adv = (adv - adv_stats.mean) / adv_stats.std
        # adv_limit = 5.0
        #
        # adv = torch.clip(adv, -adv_limit, adv_limit)

        importance = True
        ppo = True

        if ppo:
            ratio = torch.exp(log_probs - action_log_probs)
            clipped = torch.clip(ratio, 1 - epsilon, 1 + epsilon)
            policy_loss = torch.min(
                ratio * adv,
                clipped * adv,
            )
            tensor_stats.update("ratio", ratio)
            tensor_stats.update("policy_loss", policy_loss)
            policy_loss = -torch.sum(policy_loss)
        elif importance:
            ratio = torch.exp(log_probs - action_log_probs)
            tensor_stats.update("ratio", ratio)
            policy_loss = -torch.sum(adv * ratio)
        else:
            policy_loss = -torch.sum(adv * log_probs)

        # Based on advantage, reduce the entropy coefficient
        entropy_coef_fact = 0.1
        # entropy_coef2 = (1 - entropy_coef_fact) * progress + entropy_coef_fact
        # entropy_coef3 = min(abs(adv_stats.mean), 1.0)
        # entropy_loss = entropy_coef * entropy_coef3 * -torch.sum(entropy_coef2 * dist.entropy())

        entropy = masked_entropy_from_logits(logits, valid_actions)
        tensor_stats.update("entropy", entropy)

        # Based on the progress, perturb probability
        entropy = (5.5 + torch.tanh(progress * 6 - 4) * 4.5) / 10 * entropy
        entropy_loss = entropy_coef * -torch.sum(entropy)
        tensor_stats.update("entropy_loss", entropy_loss)

        actor_loss = policy_loss + entropy_loss

        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

        return policy_loss.detach(), entropy_loss.detach()

    def _train_critic(
        self,
        optimizer: torch.optim.Optimizer,
        states: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        reward: torch.FloatTensor,
        next_states: torch.LongTensor,
        next_valid_actions: torch.BoolTensor,
        terminated: torch.BoolTensor | None,
        *,
        gamma: float,
        tensor_stats: TensorStats,
    ):
        loss = self._compute_critic_loss(
            self._critic_loss_fn,
            states,
            valid_actions,
            reward,
            next_states,
            next_valid_actions,
            terminated,
            gamma=gamma,
            tensor_stats=tensor_stats,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return (loss.detach(),)

    def feed_episodes(
        self,
        episodes: list[dict[str, Any]],
        params: dict[str, Any],
        batch_size: int,
        *,
        tensor_stats: TensorStats,
        device: Any = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        from ml2048.replay import make_batches_from_tran_dict, merge_episodes

        gamma = params["gamma"]
        epsilon = params["epsilon"]
        actor_lr = params["actor_lr"]
        critic_lr = params["actor_lr"]
        entropy_coef = params["entropy_coef"]

        actor_optim = torch.optim.Adam(
            self._actor.parameters(),
            lr=actor_lr,
        )
        critic_optim = torch.optim.Adam(
            self._critic.parameters(),
            lr=critic_lr,
        )

        def to_tensor(arr, dtype=None) -> torch.Tensor:
            return torch.from_numpy(arr).to(device=device, dtype=dtype)

        tran_dict = merge_episodes(episodes)
        tensor_stats.update("reward", torch.from_numpy(tran_dict["rewards"]))

        policy_losses = []
        entropy_losses = []
        critic_losses = []

        for idx, bat in enumerate(
            make_batches_from_tran_dict(tran_dict, batch_size, seed=seed)
        ):
            # for idx, bat in enumerate(make_batches(episodes, batch_size, cycle=3, seed=seed)):
            states = to_tensor(bat["states"], torch.long)
            valid_actions = to_tensor(bat["valid_actions"], torch.bool)
            actions = to_tensor(bat["actions"], torch.long)
            action_log_probs = to_tensor(bat["action_log_probs"], torch.float32)
            next_states = to_tensor(bat["next_states"], torch.long)
            next_valid_actions = to_tensor(bat["next_valid_actions"], torch.bool)
            rewards = to_tensor(bat["rewards"], torch.float32)
            progress = to_tensor(bat["progress"], torch.float32)

            policy_loss, entropy_loss = self._train_actor(
                actor_optim,
                states,
                valid_actions,
                actions,
                action_log_probs,
                rewards,
                progress,
                next_states,
                next_valid_actions,
                gamma=gamma,
                epsilon=epsilon,
                entropy_coef=entropy_coef,
                tensor_stats=tensor_stats,
            )
            policy_losses.append(policy_loss)
            entropy_losses.append(entropy_loss)

        for _ in range(3):
            for idx, bat in enumerate(
                make_batches_from_tran_dict(tran_dict, batch_size, seed=seed)
            ):
                states = to_tensor(bat["states"], torch.long)
                valid_actions = to_tensor(bat["valid_actions"], torch.bool)
                actions = to_tensor(bat["actions"], torch.long)
                action_log_probs = to_tensor(bat["action_log_probs"], torch.float32)
                next_states = to_tensor(bat["next_states"], torch.long)
                next_valid_actions = to_tensor(bat["next_valid_actions"], torch.bool)
                rewards = to_tensor(bat["rewards"], torch.float32)
                progress = to_tensor(bat["progress"], torch.float32)

                (critic_loss,) = self._train_critic(
                    critic_optim,
                    states,
                    valid_actions,
                    rewards,
                    next_states,
                    next_valid_actions,
                    None,
                    gamma=gamma,
                    tensor_stats=tensor_stats,
                )
                critic_losses.append(critic_loss)

        losses = list(zip(policy_losses, entropy_losses, critic_losses))

        return {
            "losses": torch.Tensor(losses),
            "policy_loss": torch.tensor(policy_losses).sum().item(),
            "entropy_loss": torch.tensor(entropy_losses).sum().item(),
            "critic_loss": torch.tensor(critic_losses).sum().item(),
        }
