"""
Legacy module.

2048 implemented in pure python

"""

from collections.abc import Callable
from random import Random
from typing import MutableSequence, Optional, Union

Board = Union[MutableSequence[int], list[int]]

STEP_LEFT = 0
STEP_RIGHT = 1
STEP_UP = 2
STEP_DOWN = 3


def _identity(cells: Board):
    return cells


def _invert(cells: Board):
    # upside down
    return cells[12:16] + cells[8:12] + cells[4:8] + cells[0:4]


def _mirror(cells: Board):
    # left-right flipping
    return [cells[3 - i + j * 4] for j in range(4) for i in range(4)]


def _transpose(cells: Board):
    return [cells[i * 4 + j] for j in range(4) for i in range(4)]


_TRANSFORMATIONS: dict[
    int, tuple[Callable[[Board], Board], Callable[[Board], Board]]
] = {
    STEP_LEFT: (_identity, _identity),
    STEP_RIGHT: (_mirror, _mirror),
    STEP_UP: (_transpose, _transpose),
    STEP_DOWN: (
        lambda cells: _transpose(_invert(cells)),
        lambda cells: _invert(_transpose(cells)),
    ),
}


class Game:
    cells: Board
    random: Random

    def __init__(self, seed: Optional[int] = None):
        self.reset(seed)

    def reset(self, seed: Optional[int] = None):
        self.random = Random(seed)
        self.cells = [0] * 16
        self.spawn()
        self.spawn()

    def spawn(self):
        indices = [i for i, s in enumerate(self.cells) if not s]
        if not indices:
            raise RuntimeError("Game over")

        index = self.random.choice(indices)
        self.cells[index] = 2

    def step(self, direction: int):
        cells = self.cells
        orig = cells.copy()

        prepare, finish = _TRANSFORMATIONS[direction]

        cells = prepare(cells)

        for j in range(0, 4):
            outstanding: int = 0  # current unmerged number
            row = []

            for i in range(0, 4):
                num = cells[j * 4 + i]

                if not num:
                    continue
                elif not outstanding:
                    outstanding = num
                elif outstanding == num:
                    row.append(outstanding * 2)
                    outstanding = 0
                else:
                    row.append(outstanding)
                    outstanding = num

            if outstanding:
                row.append(outstanding)

            for i in range(0, 4):
                num = row[i] if len(row) > i else 0
                cells[j * 4 + i] = num

        cells = finish(cells)

        changed = cells != orig

        if changed:
            self.cells[:] = cells
            self.spawn()
