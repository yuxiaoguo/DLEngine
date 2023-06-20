"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member,logging-fstring-interpolation
import os
import json
import pickle
from typing import Iterator
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
from torch import distributed as dist
from torch.utils.data import IterableDataset

from dl_engine.core.register import dataset_register
from dl_engine.core.logger import Logger

from ..protocols import SequentialDataDescV0


class DistDataUtils:
    """
    Large-file-system dataset.
    """
    @staticmethod
    def get_rank_all() -> int:
        """
        Get the number of all ranks.
        """
        if dist.is_available() and dist.is_initialized():
            rank_all = dist.get_world_size()
        else:
            rank_all = 1
        return rank_all

    @staticmethod
    def get_rank_id() -> int:
        """
        Get the rank id.
        """
        if dist.is_available() and dist.is_initialized():
            rank_id = dist.get_rank()
        else:
            rank_id = 0
        return rank_id


@dataset_register.register
class LFSIterableDataset(IterableDataset):
    """
    Large-file-system dataset.
    """
    def __init__(self, data_root, desc_cfg, used_keys: dict[str, str], desc_type) -> None:
        super().__init__()
        self._desc_cfg_path = os.path.join(data_root, desc_cfg)
        with open(self._desc_cfg_path, 'r', encoding='utf-8') as desc_cfg_stream:
            self._desc_cfg = desc_type(**json.load(desc_cfg_stream))
        self._data_root = data_root
        self._used_keys = used_keys

    def __getitem__(self, index):
        Logger().warning('Not implemented')
        raise NotImplementedError

    def __iter__(self) -> Iterator:
        return super().__iter__()


class MetaInstance:
    """
    Meta instance.
    """
    def __init__(self, meta_data: dict[str, np.ndarray] | None, meta_offset, meta_count) -> None:
        self._meta_data = meta_data
        self._meta_offset = meta_offset
        self._meta_count = meta_count

    def empty(self) -> bool:
        """
        Check if the meta instance is empty.
        """
        return self._meta_data is None

    def next(self) -> dict[str, np.ndarray]:
        """
        Get the next meta instance.
        """
        assert self._meta_data is not None
        next_data = {_k: _v[self._meta_offset] for _k, _v in self._meta_data.items()}
        self._meta_offset += 1
        if self._meta_offset >= self._meta_count:
            self._meta_data = None
        return next_data


@dataset_register.register
class LFSSeqIterableDataset(LFSIterableDataset):
    """
    Large-file-system dataset compatible with sequential protocols.
    """
    def __init__(self, data_root, desc_cfg, used_keys: dict[str, str], seq_mode: bool,
        shuffle: bool) -> None:
        super().__init__(data_root, desc_cfg, used_keys, SequentialDataDescV0)
        self._seq_mode = seq_mode
        self._desc_cfg: SequentialDataDescV0 = self._desc_cfg
        self._seq_key = '' if self._seq_mode else '_nonseq'
        self._shuffle = shuffle

        self._data_cfg = self._desc_cfg.props
        self._num_all_samples = self._desc_cfg.total_samples if self._seq_mode else \
            self._desc_cfg.total_nonseq_samples
        self._num_samples = self._num_all_samples // DistDataUtils.get_rank_all() * \
            DistDataUtils.get_rank_all()
        self._num_rank_samples = self._num_samples // DistDataUtils.get_rank_all()
        Logger().info(f'num_samples: {self._num_samples}/{self._num_all_samples}')

        self._write_event = Event()

        self._prefetch_pool = None

        self._assign_meta_files()

        self._prefetch_metas = list()
        self._next_meta_idx = 0

        self._cur_meta = MetaInstance(None, 0, 0)

        self._max_cached_metas = 3
        self._prefetch_event = Event()
        self._prefetch_event.clear()
        self._filled_event = Event()
        self._filled_event.clear()

        self._queue_lock = Lock()

    def _assign_meta_files(self):
        meta_file_offsets = [getattr(_f, f'global{self._seq_key}_offset') \
            for _f in self._desc_cfg.meta_files]
        meta_file_offsets.append(getattr(self._desc_cfg, f'total{self._seq_key}_samples'))
        meta_file_offsets = np.asarray(meta_file_offsets)

        worker_start = DistDataUtils.get_rank_id() * self._num_rank_samples
        worker_end = (DistDataUtils.get_rank_id() + 1) * self._num_rank_samples
        Logger().info(f'worker_start: {worker_start}, worker_end: {worker_end}')

        meta_file_start = np.searchsorted(meta_file_offsets, worker_start, side='right') - 1
        meta_file_end = np.searchsorted(meta_file_offsets, worker_end, side='left')

        self._meta_file_desc = []
        for i in range(meta_file_start, meta_file_end):
            meta_path = os.path.join(self._data_root, self._desc_cfg.meta_files[i].meta_file)
            meta_begin = np.maximum(worker_start - meta_file_offsets[i], 0)
            meta_count = np.minimum(worker_end - meta_file_offsets[i], meta_file_offsets[i + 1] - \
                meta_file_offsets[i])
            self._meta_file_desc.append((meta_path, meta_begin, meta_count))
        if self._shuffle:
            np.random.shuffle(self._meta_file_desc)
        Logger().info(f'meta_file_desc: {self._meta_file_desc}')

    def _load_meta(self, meta_path: str):
        if meta_path.endswith('.pkl'):
            with open(meta_path, 'rb') as meta_stream:
                meta: dict[str, dict] = pickle.load(meta_stream)
        else:
            raise NotImplementedError
        data_dict = {}
        for _, value in meta.items():
            for key, value in value.items():
                if key not in self._used_keys.values():
                    continue
                key_list: list = data_dict.setdefault(key, [])
                key_list.append(value)
        if not self._seq_mode:
            for key, value in data_dict.items():
                data_dict[key] = np.concatenate(value, axis=0)
        if self._shuffle:
            for key, value in data_dict.items():
                perm = np.random.permutation(value.shape[0])
                data_dict[key] = value[perm]
        return data_dict

    def _async_fecth(self, future: Future):
        meta_info = self._meta_file_desc[self._next_meta_idx]
        fetch_meta = MetaInstance(future.result(), *meta_info[1:])
        with self._queue_lock:
            self._prefetch_metas.append(fetch_meta)
            self._next_meta_idx += 1
            self._next_meta_idx %= len(self._meta_file_desc)
            self._filled_event.set()
            self._prefetch_event.clear()

    def _get_item(self):
        if self._prefetch_pool is None:
            self._prefetch_pool = ThreadPoolExecutor(max_workers=1)

        if not self._prefetch_event.is_set():
            with self._queue_lock:
                quene_len = len(self._prefetch_metas)
            if quene_len < self._max_cached_metas:
                self._prefetch_event.set()
                future = self._prefetch_pool.submit(self._load_meta, \
                    self._meta_file_desc[self._next_meta_idx][0])
                future.add_done_callback(self._async_fecth)

        if self._cur_meta.empty():
            self._filled_event.wait()
            with self._queue_lock:
                self._cur_meta = self._prefetch_metas.pop(0)
                if len(self._prefetch_metas) == 0:
                    self._filled_event.clear()

        raw_data = self._cur_meta.next()

        proc_data = dict()
        for key, value in self._used_keys.items():
            proc_data[key] = raw_data[value]
        return proc_data

    def __iter__(self):
        for _ in range(self._num_rank_samples):
            yield self._get_item()
