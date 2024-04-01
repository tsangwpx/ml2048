import torch

from ml2048.policy import Policy
from ml2048.stats import categorical_sample


class RandomPolicy(Policy):
    def __init__(self, seed: int | None = None):
        super().__init__()

        self._generator = torch.Generator()
        if seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(seed)

    def sample_actions(
        self,
        state: torch.LongTensor,
        valid_actions: torch.BoolTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        dist = torch.distributions.Categorical(probs=valid_actions.float())
        actions = categorical_sample(dist, generator=generator)
        log_probs = dist.log_prob(actions)
        return actions.long(), log_probs.float()


def _test_random_policy():
    policy = RandomPolicy()
    actions, probs = policy.sample_actions(
        torch.ones((16,)).long(),
        torch.ones((4,)).bool(),
    )
    print(actions, probs)
    assert actions.shape == (), actions.shape
    assert probs.shape == (), probs.shape

    for batch_size in (1, 2, 3):
        actions2, probs2 = policy.sample_actions(
            torch.ones((batch_size, 16)).long(),
            torch.ones((batch_size, 4)).bool(),
        )
        print(actions2, probs2)
        assert actions2.shape == (batch_size,), actions2.shape
        assert probs2.shape == (batch_size,), probs2.shape
