import collections
import math

import numpy as np
import torch


class MaskedCategorical(torch.distributions.Categorical):
    def __init__(
        self,
        *,
        probs=None,
        logits=None,
        validate_args=None,
        mask: torch.Tensor | None = None,
    ):
        param = probs if probs is not None else logits

        if mask is None:
            mask = torch.ones_like(param, dtype=torch.bool)
        else:
            if mask.dtype != torch.bool:
                raise ValueError("mask must be bool")
            if mask.shape != param.shape:
                raise ValueError(f"Bad mask shape {mask.shape}")

        super().__init__(probs, logits, validate_args)

        self._mask = mask


def categorical_sample(
    dist: torch.distributions.Categorical,
    sample_shape=torch.Size(),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Impl categorical sample with generator"""
    if not isinstance(sample_shape, torch.Size):
        sample_shape = torch.Size(sample_shape)
    probs_2d = dist.probs.reshape(-1, dist._num_events)
    samples_2d = torch.multinomial(
        probs_2d,
        sample_shape.numel(),
        True,
        generator=generator,
    ).T
    return samples_2d.reshape(dist._extended_shape(sample_shape))


def maskd_entropy(
    dist: torch.distributions.Categorical,
    mask: torch.BoolTensor,
) -> torch.Tensor:
    logits = dist.logits
    probs = dist.probs

    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_lop_p = logits * probs
    p_lop_p = torch.where(mask, p_lop_p, 0)
    return -p_lop_p.sum(-1)


def stat_desc(x: torch.Tensor | np.ndarray) -> tuple[float, float, int, float, float]:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    min_ = torch.min(x)
    max_ = torch.max(x)
    std, mean = torch.std_mean(x, correction=0)

    return mean.item(), std.item(), x.numel(), min_.item(), max_.item()


class RollingStatistics:
    _dtype: torch.dtype

    count: int
    _sum: float
    _sqsum: float

    mean: float
    var: float
    std: float
    min: float
    max: float

    def __init__(self, max_batches: int | None = None) -> None:
        if max_batches is None:
            max_batches = -1

        self._max_batches = max_batches

        self._min_deque = collections.deque()
        self._max_deque = collections.deque()
        self._deque = collections.deque()

        self.reset()

    def reset(self):
        self._min_deque.clear()
        self._max_deque.clear()
        self._deque.clear()

        self.count = 0
        self._sqsum = 0.0
        self._sum = 0.0

        self.mean = 0
        self.var = 0
        self.std = 0
        self.min = math.inf
        self.max = -math.inf

    def update(self, data: torch.Tensor):
        """
        Update the statistics.

        Each update is one batch.
        """
        self._dtype = data.dtype

        data = data.detach()

        if 0 <= self._max_batches <= len(self._deque):
            (r_count, r_sum, r_sqsum, r_min, r_max) = self._deque.popleft()
            self.count -= r_count
            self._sum -= r_sum
            self._sqsum -= r_sqsum

            if self.min == r_min:
                self._min_deque.popleft()

            if self.max == r_max:
                self._max_deque.popleft()

        a_count = data.numel()
        a_sum = data.sum()
        a_sqsum = (data**2).sum()
        a_min = data.min()
        a_max = data.max()
        a_min = a_min.item()
        a_max = a_max.item()

        self.count += a_count
        self._sum += a_sum.item()
        self._sqsum += a_sqsum.item()

        if self.count == 0:
            self.mean = 0
            self.var = 0
            self.std = 0
        elif self.count == 1:
            self.mean = self._sum / self.count
            self.var = 0
            self.std = 0
        else:
            self.mean = self._sum / self.count
            self.var = self._sqsum / self.count - self.mean**2
            if abs(self.var) <= 1.0e-5:
                self.var = 0

            try:
                self.std = math.sqrt(self.var)
            except ValueError:
                # Somehow var is negative but large enough to raise math domain error
                # import pathlib

                # torch.save(
                #     {"__data": data, **vars(self)}, pathlib.Path("debug_dump.bin")
                # )
                print(vars(self))
                raise

        if self._max_batches >= 0:
            self._deque.append((a_count, a_sum, a_sqsum, a_min, a_max))

            while self._min_deque and self._min_deque[-1] > a_min:
                self._min_deque.pop()

            self._min_deque.append(a_min)
            self.min = self._min_deque[0]

            while self._max_deque and self._max_deque[-1] < a_max:
                self._max_deque.pop()

            self._max_deque.append(a_max)
            self.max = self._max_deque[0]
        else:
            self.max = max(self.max, a_max)
            self.min = min(self.min, a_min)

    def __repr__(self) -> str:
        number_fmt = " z.5e"
        extreme_fmt = " z.5e" if self._dtype.is_floating_point else "s"

        return (
            f"<RollingStats"
            f" count={self.count},"
            f" mean={self.mean:{number_fmt}},"
            f" std={self.std:{number_fmt}},"
            f" min={self.min:{extreme_fmt}},"
            f" max={self.max:{extreme_fmt}}>"
        )


class TensorStats:
    def __init__(self):
        self.table: dict[str, RollingStatistics] = collections.defaultdict(
            RollingStatistics
        )

    def update(self, key: str, tensor: torch.Tensor):
        self.table[key].update(tensor)
