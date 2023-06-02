"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
from typing import Type

from ..logger.tensorboard_logger import SingletonWriter
from ..base import BaseCallback, BaseIO


class VisualizerIO(BaseIO):
    """
    The class is used to define the interface of visualizer IO.
    """
    def __init__(self):
        super().__init__()
        self.predicted = None
        self.ground_truth = None


class BaseVisualizer(BaseCallback):
    """
    The class is used to define the interface of visualizer.
    """
    def __init__(self, stride, in_descs=None, out_descs=None,
        io_type: Type[BaseIO]=VisualizerIO) -> None:
        super().__init__(in_descs=in_descs, out_descs=out_descs, io_type=io_type)
        self._stride = stride
        self._writer = SingletonWriter()
        self._tensorb = self._writer.writer

        self._iter = 0

    def reset(self) -> None:
        self._iter = 0

    def _run(self, io_proto: VisualizerIO, **extra_kwargs) -> VisualizerIO:
        raise NotImplementedError

    def close(self) -> None:
        self._iter = 0
