"""
Legacy module.

2048 implemented with numpy

"""

from collections.abc import Callable
from random import Random
from typing import Any, Optional

import numpy as np
from numba import njit

from ml2048.game import STEP_DOWN, STEP_LEFT, STEP_RIGHT, STEP_UP, Board


@njit
def _identity(src, dst):
    dst[:] = src[:]


@njit
def _invert(src, dst):
    # up-down flipping
    for i in range(0, 16, 4):
        # 0 <--> 12
        # 4 <--> 8
        j = 12 - i
        dst[j : j + 4] = src[i : i + 4]


@njit
def _mirror(src, dst):
    # left-right flipping
    for j in range(4):
        for i in range(4):
            dst[3 - i + j * 4] = src[i + j * 4]


@njit
def _transpose(src, dst):
    # transpose, rotate_left
    for j in range(4):
        for i in range(4):
            p = j
            q = i
            dst[p + q * 4] = src[i + j * 4]


@njit
def _anti_transpose(src, dst):
    # anti_transpose, rotate_right
    for j in range(4):
        for i in range(4):
            p = 3 - j
            q = 3 - i
            dst[p + q * 4] = src[i + j * 4]


_TRANSFORMATIONS: dict[
    int,
    tuple[Callable[[Board, Board], Any], Callable[[Board, Board], Any]],
] = {
    STEP_LEFT: (_identity, _identity),
    STEP_RIGHT: (_mirror, _mirror),
    STEP_UP: (_transpose, _transpose),
    STEP_DOWN: (_anti_transpose, _anti_transpose),
}


class Game:
    cells: Board
    _alt: Board
    _row: np.ndarray
    random: Random

    def __init__(self, seed: Optional[int] = None):
        self.reset(seed)

    def reset(self, seed: Optional[int] = None):
        self.random = Random(seed)
        self.cells = np.zeros((16,), dtype=np.uint32)
        self._alt = np.zeros((16,), dtype=np.uint32)
        self._row = np.zeros((4,), dtype=np.uint32)
        self.spawn()
        self.spawn()

    def spawn(self):
        indices = np.flatnonzero(self.cells == 0)
        if not indices.size:
            raise RuntimeError("Game over")
        k = self.random.randrange(indices.size)
        print("spaw", indices, k, indices.size)
        self.cells[indices[k]] = 2

    def step(self, direction: int):
        src = self.cells
        dst = self._alt
        row = self._row
        orig = src.copy()

        prepare, finish = _TRANSFORMATIONS[direction]

        print("cells", src)
        prepare(src, dst)
        src, dst = dst, src

        print("prep", src)

        for j in range(0, 4):
            outstanding: int = 0  # current unmerged number
            row.fill(0)
            nelem = 0  # number of items in row

            for i in range(0, 4):
                num = src[j * 4 + i]

                if not num:
                    continue
                elif not outstanding:
                    outstanding = num
                elif outstanding == num:
                    row[nelem] = outstanding * 2
                    nelem += 1
                    outstanding = 0
                else:
                    row[nelem] = outstanding
                    nelem += 1
                    outstanding = num

            if outstanding:
                row[nelem] = outstanding
                nelem += 1
                outstanding = 0

            src[j * 4 : j * 4 + 4] = row

        print("step", src)
        finish(src, dst)
        src, dst = dst, src
        print("fini", src)

        changed = (src != orig).any()

        if changed:
            # TODO Check if full then game over
            print(orig, self.cells, sep="\n")
            self.cells = src
            self._alt = dst
            self.spawn()
