"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
import os
import zipfile
from io import BytesIO
from typing import Optional

import torch
import numpy as np
from dl_engine.core.network import BaseIO

from dl_engine.core.register import functional_register
from dl_engine.core.config import PipelineConfig

from ..base import BaseCallback, BaseIO


class DataIO(BaseIO):
    """
    DataIO is a data structure that contains the data to be recorded.
    """
    def __init__(self) -> None:
        super().__init__()
        self.data: Optional[torch.Tensor] = None


@functional_register.register
class ZipRecorder(BaseCallback):
    """
    ZipRecorder is a recorder that records the data into zip file.
    """
    def __init__(self, rel_out_dir: str, record_alias: str, last_shape: Optional[list] = None, \
        in_descs=None, out_descs=None, io_type=DataIO) -> None:
        super().__init__(in_descs=in_descs, out_descs=out_descs, io_type=io_type)
        self._zip_file: Optional[zipfile.ZipFile] = None
        self.last_shape = last_shape
        self._rel_out_dir = rel_out_dir
        self._record_alias = record_alias
        self._dump_keys = []

        self._config = PipelineConfig()

        self._iter = 0

    def reset(self):
        """
        Start recording the data. Zip file will be created if not exists.
        """
        assert self._zip_file is None, "ZipRecorder is already recording."
        output_folder = os.path.join(self._config.prof_dir, self._rel_out_dir)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'{self._record_alias}.zip')
        self._zip_file = zipfile.ZipFile(output_path, "w")
        self._iter = 0

    def _run(self, io_proto: DataIO, **extra_kwargs):
        """
        Record the data into zip file.
        """
        assert self._zip_file is not None, "ZipRecorder is not recording."
        assert io_proto.data is not None, "ZipRecorder only supports tensor data."
        data = io_proto.data
        if self.last_shape is not None:
            data = data.reshape([data.shape[0], -1, *self.last_shape])
        if isinstance(data, torch.Tensor):
            for b_idx in range(data.shape[0]):
                b_data = data[b_idx]
                b_np = b_data.detach().cpu().numpy()
                b_stream = BytesIO()
                np.save(b_stream, b_np)
                b_stream.seek(0)
                self._zip_file.writestr(f'{self._iter:07d}.npy', b_stream.read())
                self._iter += 1
        elif isinstance(data, list):
            for g_data in zip(*data):
                assert isinstance(g_data, (list, tuple))
                for h_idx in range(g_data[0].shape[0]):
                    for l_idx, l_data in enumerate(g_data):
                        b_np = l_data[h_idx].detach().cpu().numpy()
                        b_stream = BytesIO()
                        np.save(b_stream, b_np)
                        b_stream.seek(0)
                        out_path = f'{self._iter:07d}_L{l_idx}_H{h_idx}.npy'
                        self._zip_file.writestr(out_path, b_stream.read())
                self._iter += 1
        else:
            raise NotImplementedError

    def close(self):
        """
        Stop recording the data. Zip file will be closed.
        """
        if self._zip_file is not None:
            self._zip_file.close()
        self._zip_file = None
