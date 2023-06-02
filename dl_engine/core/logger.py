"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=logging-fstring-interpolation
import logging

from torch import distributed as dist

from dl_engine.utils import Singleton


class Logger(metaclass=Singleton):
    """
    Logger system used in distributed training.
    """
    def __init__(self, name=None) -> None:
        if name is None:
            name = __name__
        self._log = logging.getLogger(__name__)
        self._log.setLevel(logging.INFO)
        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def info(self, msg):
        """
        Log info message.
        """
        self._log.info(f'Rank {self._rank}: {msg}')

    def warning(self, msg):
        """
        Log warning message.
        """
        self._log.warning(f'Rank {self._rank}: {msg}')

    def info_zero_rank(self, msg):
        """
        Log info message from rank 0.
        """
        if self._rank == 0:
            self.info(msg)

    def warning_zero_rank(self, msg):
        """
        Log warning message from rank 0.
        """
        if self._rank == 0:
            self.warning(msg)
