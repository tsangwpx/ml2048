import torch

from ml2048.policy import Policy, ProbabilisticPolicy


class MaxLikelyPolicy(Policy):
    """
    Wrap a ProbabilisticPolicy and sample actions with the highest probabilities
    It is also called greedy sampling.
    """

    def __init__(self, inner: ProbabilisticPolicy):
        super().__init__()

        self.inner = inner

    def sample_actions(
        self,
        state: torch.LongTensor | torch.ByteTensor,
        valid_actions: torch.BoolTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.FloatTensor]:
        logits = self.inner.action_logits(state, valid_actions)

        min_real = torch.finfo(logits.dtype).min
        logits = torch.where(valid_actions, logits, min_real)

        actions = torch.argmax(logits, dim=-1)
        log_probs = torch.zeros_like(actions, dtype=torch.float32).float()

        return actions, log_probs
