"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member
from typing import Dict

import torch
import numpy as np

from dl_engine.core.register import functional_register
from .base import BaseIO, BaseVisualizer


class ScalarIO(BaseIO):
    """
    ScalarIO for scalar visualizer
    """
    def __init__(self):
        super().__init__()
        self.scalar = None


@functional_register.register
class ScalarVisualizer(BaseVisualizer):
    """
    Scalar visualizer to visual the scalar curves
    """
    def __init__(self, stride, tag='', epoch_stat=False, in_descs=None, out_descs=None,
        io_type=ScalarIO) -> None:
        super().__init__(stride, in_descs=in_descs, out_descs=out_descs, io_type=io_type)
        self._tag = tag
        self._epoch_stat = epoch_stat
        self._scalars: dict[str, list] = dict()

    def reset(self) -> None:
        super().reset()
        self._scalars = dict()

    def _run(self, io_proto: ScalarIO, **extra_kwargs):
        assert io_proto.scalar is not None, 'Input scalars should not be None'
        scalars: Dict[str, torch.Tensor] = io_proto.scalar
        if isinstance(scalars, torch.Tensor):
            scalars = dict(default=scalars)
        assert isinstance(scalars, dict), \
            f'Input scalars should be a dict, but got {type(scalars)}'
        for key, value in scalars.items():
            if value.numel() > 1:
                value = torch.mean(value)
            self._scalars.setdefault(key, []).append(value.item())
            if not self._epoch_stat and self._iter % self._stride == 0:
                self._writer.writer.add_scalar(f'{self._tag}/{key}', value, self._writer.iter)
        self._iter += 1

    def close(self) -> None:
        super().close()
        if self._epoch_stat:
            for key, value in self._scalars.items():
                mean_scalar = np.mean(value)
                self._writer.writer.add_scalar(\
                    f'{self._tag}/{key}', mean_scalar, self._writer.iter)
