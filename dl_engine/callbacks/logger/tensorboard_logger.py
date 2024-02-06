"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""

from torch.utils.tensorboard.writer import SummaryWriter
from torch.distributed import get_rank, is_initialized


class SingletonWriter:
    """
    Singleton Tensorboard writer instance
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SingletonWriter, cls).__new__(cls)
        return cls.instance

    def initialize(self, log_dir):
        """
        Initialize the writer
        """
        self._iter = 0  # pylint: disable=attribute-defined-outside-init
        self._epoch = 0  # pylint: disable=attribute-defined-outside-init
        if is_initialized() and get_rank() != 0:
            return
        self._writer = SummaryWriter(log_dir=log_dir)  # pylint: disable=attribute-defined-outside-init
        self._writer.add_scalar('epoch', self._epoch, self._iter)

    @property
    def writer(self):
        """
        Get the writer
        """
        return self._writer

    @property
    def iter(self):
        """
        Get the iteration number
        """
        return self._iter

    @property
    def epoch(self):
        """
        Get the epoch number
        """
        return self._epoch

    def update_iter(self, tick=1):
        """
        Update the iteration number
        """
        self._iter += tick

    def update_epoch(self, tick=1):
        """
        Update the epoch number
        """
        self._epoch += tick
        if get_rank() != 0:
            return
        self._writer.add_scalar('epoch', self._epoch, self._iter)
