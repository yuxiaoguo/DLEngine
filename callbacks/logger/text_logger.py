"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from typing import Dict, List

from dl_engine.core.logger import Logger
from dl_engine.core.register import functional_register
from ..base import BaseCallback, BaseIO


@functional_register.register
class ProgressTextLogger(BaseCallback):
    """
    Progress bar logger callback.
    """
    def __init__(self, stride, prefix='', in_descs=None, out_descs=None, io_type=BaseIO):
        super().__init__(in_descs=in_descs, out_descs=out_descs, io_type=io_type)
        self._prefix = prefix
        self._stride = stride
        self._iter = 0

        self._step_recorder: Dict[str, List] = dict()

    def _gather_iter_info(self, iter_out: BaseIO):
        for l_name, l_tenor in iter_out.losses.items():
            self._step_recorder.setdefault(l_name, list()).append(l_tenor.item())
        for m_name, m_tenor in iter_out.metrics.items():
            self._step_recorder.setdefault(m_name, list()).append(m_tenor.item())

    def _gen_out_str(self):
        out_str = ' - '.join(
            [f'{k}: {sum(v) / len(v):.5f}' for k, v in self._step_recorder.items()])
        return out_str

    def reset(self) -> None:
        self._iter = 0

    def close(self) -> None:
        self._step_recorder = dict()

    def _run(self, io_proto, **extra_kwargs):
        """
        Run the callback.
        """
        self._gather_iter_info(io_proto)
        if self._iter % self._stride == 0:
            out_msg = self._gen_out_str()
            Logger().info_zero_rank(f'{self._prefix} (iter: {self._iter:07d}) - {out_msg}')
        self._iter += 1
