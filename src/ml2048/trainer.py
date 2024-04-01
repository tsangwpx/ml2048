import argparse
import contextlib
import io
import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


class BaseTrainer(metaclass=ABCMeta):

    def __init__(
        self,
        arguments: dict[str, Any],
        *,
        save_dir: Path | None,
        logger: logging.Logger | None = None,
    ):
        self._logger = logger
        self._save_dir = save_dir
        self._exit_stack = contextlib.ExitStack()

        self._arguments = arguments
        self._epoches = arguments["epoches"]
        self._epoch_start = arguments["epoch_start"] or 0
        assert self._epoch_start >= 0
        self._epoch = self._epoch_start

    def _get_epoch_range(self):
        return range(self._epoch_start, self._epoches)

    def save_state(
        self,
        name: str,
        state: dict[str, Any],
    ):
        path = self._save_dir.joinpath(name)
        torch.save(state, path)

    def print(
        self,
        /,
        *args,
        **kwargs,
    ):
        print(*args, **kwargs)

        if self._logger is not None:
            if not args:
                self._logger.info("")
            elif len(args) == 1:
                self._logger.info(args[0])
            else:
                with io.StringIO() as sio:
                    print(*args, **kwargs, file=sio, end="")
                    self._logger.info(sio.getvalue())

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser()
        p.add_argument("--epoches", type=int, default=10_000)
        p.add_argument("--epoch-start", type=int, default=None)
        p.add_argument("--restart", type=str, default=None)
        p.add_argument("--warming-steps", type=int, default=None)
        return p

    @classmethod
    def main(
        cls,
    ):
        # Reduce the size of cuda context after benchmark.
        # Maybe cuda did not free the allocation if benchmark is not performed?
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        now = datetime.now()
        fmt = "%Y%m%d_%H%M%S"
        save_dir = Path("runs", f"ml2048_{now.strftime(fmt)}")
        save_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("ml2048")
        logger.setLevel(logging.DEBUG)

        stream = logging.FileHandler(str(save_dir / "output.log"), encoding="utf-8")
        logger.addHandler(stream)

        p = cls.parser()
        ns = p.parse_args()

        trainer = cls(
            vars(ns),
            save_dir=save_dir,
            logger=logger,
        )
        trainer.run()
