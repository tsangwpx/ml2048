import torch

from ml2048.policy import SupportsEvalValue
from ml2048.stats import TensorStats


def compute_gae(
    policy: SupportsEvalValue,
    data: dict[str, torch.Tensor],
    *,
    gamma: float,
    lambda_: float,
    tensor_stats: TensorStats,
):
    """
    Compute generalized advantage estimation
    """

    # compute GAE over the step dimension
    # N = batch dimension
    # M = step dimension

    # (N, M, step, 16)
    state = data["state"]
    # (N, M, step)
    valid_actions = data["valid_actions"]
    # (N, M, step)
    reward = data["reward"]
    next_state = data["next_state"]
    next_valid_actions = data["next_valid_actions"]
    # (N, M, step)
    terminated = data["terminated"]
    # (N, M, step)
    adv = data["adv"]

    assert adv.ndim == 3, state.shape
    use_count, step_count, game_count = adv.shape

    mask = ~terminated
    with torch.no_grad():
        v0 = policy.eval_value(
            state.to(torch.long),
            valid_actions.to(torch.bool),
        )
        v1 = policy.eval_value(
            next_state.to(torch.long),
            next_valid_actions.to(torch.bool),
        )

    delta = gamma * v1 * mask + reward - v0
    tensor_stats.update("reward", reward)
    tensor_stats.update("state_value", v0)
    tensor_stats.update("delta", delta)

    del v0, v1

    coef = gamma * lambda_

    tmp = torch.zeros(
        (use_count, game_count),
        device=adv.device,
        dtype=adv.dtype,
    )

    for idx in reversed(range(step_count)):
        tmp = tmp * coef
        tmp = delta[:, idx, :] + tmp * mask[:, idx, :]
        adv[:, idx, :] = tmp
