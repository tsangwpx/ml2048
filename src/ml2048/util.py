from typing import Mapping, Sequence

import torch

SpecSequence = Sequence[tuple[str, tuple[int, ...], torch.dtype]]
SpecMapping = Mapping[str, tuple[tuple[int, ...], torch.dtype]]


def _normalize_spec(spec: SpecSequence | SpecMapping) -> SpecSequence:
    if isinstance(spec, Mapping):
        return tuple((name, shape, dtype) for name, (shape, dtype) in spec.items())
    else:
        return tuple(spec)


def new_tensors(
    spec: SpecSequence | SpecMapping,
    batch_shape: tuple[int, ...],
    *,
    device: str | None = None,
    pin_memory: bool | None = None,
    empty: bool = False,
) -> dict[str, torch.Tensor]:
    if empty:
        factory = torch.empty
    else:
        factory = torch.zeros

    def make(shape, dtype):
        return factory(
            batch_shape + shape,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )

    result = {}

    for name, shape, dtype in _normalize_spec(spec):
        if name in result:
            raise ValueError(f"Duplicated name {name:r}")

        result[name] = make(shape, dtype)

    return result


def reshape_tensors(
    spec: SpecSequence | SpecMapping,
    batch_shape: tuple[int, ...],
    data: dict[str, torch.Tensor],
    *,
    return_view: bool | None = False,
) -> dict[str, torch.Tensor]:
    sym_diff = set(spec.keys()).symmetric_difference(data.keys())
    if sym_diff:
        raise ValueError(f"sym_diff: {sorted(sym_diff)!r}")

    if return_view:

        def convert(x: torch.Tensor, suffix: tuple[int, ...]):
            return x.view(batch_shape + suffix)

    else:

        def convert(x: torch.Tensor, suffix: tuple[int, ...]):
            return x.reshape(batch_shape + suffix)

    return {
        name: convert(data[name], shape) for name, shape, _ in _normalize_spec(spec)
    }


def convert_tensors(
    spec: SpecSequence | SpecMapping,
    data: dict[str, torch.Tensor],
    *,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """Convert device and dtype, ignore shapes"""

    result = {
        name: data[name].to(device=device, dtype=dtype)
        for name, _, dtype in _normalize_spec(spec)
    }

    return result


def check_tensors(
    spec: SpecSequence | SpecMapping,
    batch_shape: tuple[int, ...],
    tensors: dict[str, torch.Tensor],
    *,
    device: str | None = None,
):
    if device is not None:
        raise NotImplementedError

    for name, shape, dtype in _normalize_spec(spec):
        if name not in tensors:
            raise ValueError(f"Tensor {name!r} is missing")

        ten = tensors[name]
        if not isinstance(ten, torch.Tensor):
            raise ValueError(f"{name!r} is not a tensor: {type(ten)!r}")
        if ten.dtype != dtype:
            raise ValueError(
                f"Tensor {name!r} expects {dtype!r} but {ten.dtype!r} found"
            )

        full_shape = batch_shape + shape
        if ten.shape != full_shape:
            raise ValueError(
                f"Tensor {name!r} expects shape {full_shape} instead of {ten.shape}"
            )
