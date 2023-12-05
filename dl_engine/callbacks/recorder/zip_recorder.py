"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member
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
        self.data_mask: torch.Tensor | None = None
        self.data_name: Optional[torch.Tensor] = None


class BaseOutStream():
    """
    BaseOutStream is a base class for output stream.
    """
    def __init__(self, out_path_base) -> None:
        self._out_path_base = out_path_base

    @abstractmethod
    def write(self, data: np.ndarray | dict, name: str) -> None:
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

    def write(self, data: np.ndarray | dict, name: str) -> None:
        """
        Write data into output stream.
        """
        assert isinstance(data, np.ndarray), \
            "ZipOutStream only supports numpy.ndarray."
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
        self._data_dict: dict[str, np.ndarray | dict] = {}

    def write(self, data: np.ndarray | dict, name: str) -> None:
        """
        Write data into output stream.
        """
        self._data_dict[name] = data

    def close(self):
        """
        Close the output stream.
        """
        sorted_keys = sorted(list(self._data_dict.keys()))
        sorted_data_dict = {_k: self._data_dict[_k] for _k in sorted_keys}
        with open(f'{self._out_path_base}.pkl', 'wb') as pkl_stream:
            pickle.dump(sorted_data_dict, pkl_stream)
        self._data_dict = {}


@functional_register.register
class ArrayRecorder(BaseCallback):
    """
    ZipRecorder is a recorder that records the data into zip file.
    """
    def __init__(self, rel_out_dir: str, record_alias: str, last_shape: Optional[list] = None, \
        mask_dim=1, in_descs=None, out_descs=None, io_type=DataIO, \
        out_stream_type: Type[BaseOutStream] = ZipOutStream) -> None:
        super().__init__(in_descs=in_descs, out_descs=out_descs, io_type=io_type)
        self._meta_stream: BaseOutStream | None = None
        self._out_stream_type = out_stream_type

        self.last_shape = last_shape
        self._rel_out_dir = rel_out_dir
        self._record_alias = record_alias
        self._dump_keys = []

        self._config = PipelineConfig()

        self._iter = 0
        self._mask_dim = mask_dim - 1

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

    def _parse_item(self,
                    data: torch.Tensor,
                    data_mask: torch.Tensor,
                    data_name: torch.Tensor,
                    batch_idx: int):
        b_np: np.ndarray = data[batch_idx].detach().cpu().numpy()
        if data_mask is not None:
            b_mask: np.ndarray = data_mask[batch_idx].detach().cpu().numpy()
            # Select with mask along a given axis
            # b_np = torch.gather(b_np, self._mask_dim, b_mask)
            assert self._mask_dim in [0, 1], "Only support 2D mask."
            if self._mask_dim == 0:
                b_np = b_np[b_mask > 0]
            else:
                b_np = b_np[:, b_mask > 0]
        if data_name is not None:
            native_name = data_name[batch_idx]
            if isinstance(native_name, torch.Tensor):
                name_uint8 = native_name.detach().cpu().numpy()
                b_name = ''.join([chr(_c) for _c in name_uint8])
            elif isinstance(native_name, str):
                b_name = native_name
            else:
                raise NotImplementedError
        else:
            b_name = f'{self._iter:07d}'
        return b_np, b_name

    def _run(self, io_proto: DataIO, **extra_kwargs):
        """
        Record the data into zip file.
        """
        assert self._meta_stream is not None, "ZipRecorder is not recording."
        assert io_proto.data is not None, "ZipRecorder only supports tensor data."
        data = io_proto.data
        assert isinstance(data, (torch.Tensor)), "Recorder only supports tensor data."
        if self.last_shape is not None:
            if data.shape[-1] == self.last_shape[0]:
                rest_dims = self.last_shape
            else:
                rest_dims = [-1, *self.last_shape]
            data = data.reshape([*data.shape[:-1], *rest_dims])
        for b_idx in range(data.shape[0]):
            b_np, b_name = self._parse_item(data, io_proto.data_mask, io_proto.data_name, b_idx)
            self._meta_stream.write(b_np, b_name)
            self._iter += 1

    def close(self):
        """
        Stop recording the data. Zip file will be closed.
        """
        if self._meta_stream is not None:
            self._meta_stream.close()
        self._meta_stream = None


@functional_register.register
class PickleRecorder(ArrayRecorder):
    """
    PickleRecorder is a recorder that records the data into pickle file.
    """
    def __init__(self, rel_out_dir: str, record_alias: str, last_shape: list | None = None,
        mask_dim=1, in_descs=None, out_descs=None, io_type=DataIO) -> None:
        super().__init__(rel_out_dir, record_alias, last_shape, mask_dim=mask_dim, \
            in_descs=in_descs, out_descs=out_descs, io_type=io_type, \
            out_stream_type=PickleOutStream)


@functional_register.register
class DictPickleRecorder(PickleRecorder):
    """
    Using dictionary to record the data
    """
    def _run(self, io_proto: DataIO, **extra_kwargs):
        dict_data = io_proto.data
        assert isinstance(dict_data, dict)
        out_items = dict()
        for d_key, d_value in dict_data.items():
            assert isinstance(d_value, torch.Tensor)
            for b_idx in range(d_value.shape[0]):
                b_np, b_name = self._parse_item(\
                    d_value, io_proto.data_mask, io_proto.data_name, b_idx)
                item = out_items.setdefault(b_name, dict())
                item[d_key] = b_np
        for b_name, b_dict in out_items.items():
            self._meta_stream.write(b_dict, b_name)
            self._iter += 1
        return
