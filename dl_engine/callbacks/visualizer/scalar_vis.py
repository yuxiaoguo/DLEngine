"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=no-member
from typing import Dict

import torch

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
    def __init__(self, stride, in_descs=None, out_descs=None, io_type=ScalarIO) -> None:
        super().__init__(stride, in_descs=in_descs, out_descs=out_descs, io_type=io_type)

    def _run(self, io_proto: ScalarIO, **extra_kwargs):
        if self._iter % self._stride != 0:
            self._iter += 1
            return

        assert io_proto.scalar is not None, 'Input scalars should not be None'
        scalars: Dict[str, torch.Tensor] = io_proto.scalar
        if isinstance(scalars, torch.Tensor):
            scalars = dict(default=scalars)
        assert isinstance(scalars, dict), f'Input scalars should be a dict, but got {type(scalars)}'
        for key, value in scalars.items():
            self._writer.writer.add_scalar(key, value, self._writer.iter)
        self._iter += 1
