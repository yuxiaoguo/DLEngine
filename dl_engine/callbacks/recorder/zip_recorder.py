"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
import os
import zipfile
import pickle
from abc import abstractmethod
from io import BytesIO
from typing import Optional, Type

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


class BaseOutStream():
    """
    BaseOutStream is a base class for output stream.
    """
    def __init__(self, out_path_base) -> None:
        self._out_path_base = out_path_base

    @abstractmethod
    def write(self, data: np.ndarray, name: str) -> None:
        """
        Write data into output stream.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Close the output stream.
        """
        raise NotImplementedError


class ZipOutStream(BaseOutStream):
    """
    ZipOutStream is a output stream that writes data into zip file.
    """
    def __init__(self, out_path_base) -> None:
        super().__init__(out_path_base)
        self._zip_file = zipfile.ZipFile(f'{self._out_path_base}.zip', "w")

    def write(self, data: np.ndarray, name: str) -> None:
        """
        Write data into output stream.
        """
        b_stream = BytesIO()
        np.save(b_stream, data)
        b_stream.seek(0)
        self._zip_file.writestr(f'{name}.npy', b_stream.read())

    def close(self):
        """
        Close the output stream.
        """
        self._zip_file.close()


class PickleOutStream(BaseOutStream):
    """
    PickleOutStream is a output stream that writes data into pickle file.
    """
    def __init__(self, out_path_base) -> None:
        super().__init__(out_path_base)
        self._data_dict: dict[str, np.ndarray] = {}

    def write(self, data: np.ndarray, name: str) -> None:
        """
        Write data into output stream.
        """
        self._data_dict[name] = data

    def close(self):
        """
        Close the output stream.
        """
        with open(f'{self._out_path_base}.pkl', 'wb') as pkl_stream:
            pickle.dump(self._data_dict, pkl_stream)
        self._data_dict = {}


@functional_register.register
class ZipRecorder(BaseCallback):
    """
    ZipRecorder is a recorder that records the data into zip file.
    """
    def __init__(self, rel_out_dir: str, record_alias: str, last_shape: Optional[list] = None, \
        in_descs=None, out_descs=None, io_type=DataIO, out_stream_type: Type[BaseOutStream] = \
        ZipOutStream) -> None:
        super().__init__(in_descs=in_descs, out_descs=out_descs, io_type=io_type)
        self._meta_stream: BaseOutStream | None = None
        self._out_stream_type = out_stream_type

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
        assert self._meta_stream is None, "ZipRecorder is already recording."
        output_folder = os.path.join(self._config.prof_dir, self._rel_out_dir)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, self._record_alias)
        self._meta_stream = self._out_stream_type(output_path)
        self._iter = 0

    def _run(self, io_proto: DataIO, **extra_kwargs):
        """
        Record the data into zip file.
        """
        assert self._meta_stream is not None, "ZipRecorder is not recording."
        assert io_proto.data is not None, "ZipRecorder only supports tensor data."
        data = io_proto.data
        if self.last_shape is not None:
            data = data.reshape([data.shape[0], -1, *self.last_shape])
        if isinstance(data, torch.Tensor):
            for b_idx in range(data.shape[0]):
                b_data = data[b_idx]
                b_np = b_data.detach().cpu().numpy()
                self._meta_stream.write(b_np, f'{self._iter:07d}')
                self._iter += 1
        elif isinstance(data, list):
            for g_data in zip(*data):
                assert isinstance(g_data, (list, tuple))
                for h_idx in range(g_data[0].shape[0]):
                    for l_idx, l_data in enumerate(g_data):
                        b_np = l_data[h_idx].detach().cpu().numpy()
                        out_path = f'{self._iter:07d}_L{l_idx}_H{h_idx}'
                        self._meta_stream.write(b_np, out_path)
                self._iter += 1
        else:
            raise NotImplementedError

    def close(self):
        """
        Stop recording the data. Zip file will be closed.
        """
        if self._meta_stream is not None:
            self._meta_stream.close()
        self._meta_stream = None


@functional_register.register
class PickleRecorder(ZipRecorder):
    """
    PickleRecorder is a recorder that records the data into pickle file.
    """
    def __init__(self, rel_out_dir: str, record_alias: str, last_shape: list | None = None,
        in_descs=None, out_descs=None, io_type=DataIO) -> None:
        super().__init__(rel_out_dir, record_alias, last_shape, in_descs, out_descs, \
            io_type, PickleOutStream)
