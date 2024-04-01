"""
Play the game prototype in command line.

This script may be broken because the main focus is not playing 2048 by human.
"""

import sys

sys.path.insert(0, "src")

import numpy as np

from ml2048.game import STEP_DOWN, STEP_LEFT, STEP_RIGHT, STEP_UP, Board
from ml2048.game_numba import Game


def print_board(cells: Board):
    fmt = "| {:4s} | {:4s} | {:4s} | {:4s} |"

    for k in range(0, 16, 4):
        items = [f"{s:4d}" if s else "" for s in cells[k : k + 4]]
        print(fmt.format(*items))


_STEP_LETTERS = {
    "L": STEP_LEFT,
    "R": STEP_RIGHT,
    "U": STEP_UP,
    "D": STEP_DOWN,
}


def main():
    game = Game()
    board = np.zeros((16,), dtype=np.int32)

    while True:
        game.render(board)
        print_board(board)

        ans = input("Move: ").upper()

        if ans == "Q":
            print("Bye")
            return

        try:
            action = _STEP_LETTERS[ans]
        except KeyError:
            print("Bad action, try again")
            continue

        valid_action, completed = game.step(action)

        if not valid_action:
            print("Bad action, try again")
            continue

        if completed:
            print("Game over")
            return


if __name__ == "__main__":
    main()
