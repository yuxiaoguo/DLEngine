"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member,logging-fstring-interpolation,eval-used
import os
import json
import pickle
import importlib
import enum
from dataclasses import dataclass, field
from typing import Iterator, Callable
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import IterableDataset

from dl_engine.core.register import dataset_register
from dl_engine.core.logger import Logger

from ..protocols import SequentialDataDescV0, MetaSeqFileDescV0


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


class RankMethod(enum.Enum):
    """
    Rank method for samples from a prefetch file
    """
    ORIGIN = enum.auto()
    RANDOM = enum.auto()
    SEQ_LEN_ASC = enum.auto()
    SEQ_LEN_DESC = enum.auto()


@dataclass
class KeyDataDesc:
    """
    Key data description.
    """
    raw_key: str
    dtype: str = 'float32'
    transforms: list[Callable] = field(default_factory=list)
    fetch_packages: list[list[str]] = field(default_factory=list)
    fetch_transforms: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        self.transforms = [eval(_t) for _t in self.transforms]  # type: ignore
        self.fetch_transforms = [eval(_t) for _t in self.fetch_transforms]  # type: ignore


@dataclass
class LFSMetaDesc:
    """
    Meta description for LFS dataset.

    Args:
        pieces_offset: the start global offset of pieces for each sequence. The
            shape is (num_sequences + 1, ). The last element is the total number
            of pieces.
    """
    pieces_offset: np.ndarray | None = None
    random_range: int = 0
    file_path: str = ''
    seq_begin: int = 0
    seq_count: int = 0


class MetaInstance:
    """
    Meta instance.
    """
    def __init__(self,
                 meta_data: dict[str, np.ndarray] | None,
                 meta_cfg: LFSMetaDesc,
                 shuffle=False,
                 piece_len=0) -> None:
        self._meta_data = meta_data
        self._meta_cfg = meta_cfg
        self._shuffle = shuffle
        self._piece_len = piece_len

        self._cur_idx = 0
        self._index_map = np.arange(meta_cfg.seq_count) + meta_cfg.seq_begin
        if self._shuffle:
            np.random.shuffle(self._index_map)

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
        sel_idx = self._index_map[self._cur_idx]
        seq_idx = np.searchsorted(\
            self._meta_cfg.pieces_offset, sel_idx, side='right') - 1
        if self._piece_len > 0:
            seq_bias = sel_idx - self._meta_cfg.pieces_offset[seq_idx]
            rand_bias = np.random.randint(0, self._meta_cfg.random_range[seq_idx])
            seq_start = seq_bias * self._piece_len + rand_bias
            seq_end = seq_start + self._piece_len
            next_data = {_k: _v[seq_idx][seq_start:seq_end] for _k, _v in self._meta_data.items()}
            next_data['name'] = self._meta_data['name'][seq_idx] + f'_p{seq_bias}'
        else:
            next_data = {_k: _v[seq_idx] for _k, _v in self._meta_data.items()}
        self._cur_idx += 1
        if self._cur_idx >= self._meta_cfg.seq_count:
            self._meta_data = None
        return next_data


@dataset_register.register
class LFSIterableDataset(IterableDataset):
    """
    Large-file-system dataset.
    """
    def __init__(self, data_root, desc_cfg, used_keys: dict[str, str | dict], desc_type) -> None:
        super().__init__()
        self._desc_cfg_path = os.path.join(data_root, desc_cfg)
        with open(self._desc_cfg_path, 'r', encoding='utf-8') as desc_cfg_stream:
            self._desc_cfg = desc_type(**json.load(desc_cfg_stream))
        self._data_root = data_root
        self._used_keys = {
            _k: KeyDataDesc(**_v) if isinstance(_v, dict) else KeyDataDesc(_v) \
                for _k, _v in used_keys.items()
        }

    def __getitem__(self, index):
        Logger().warning('Not implemented')
        raise NotImplementedError

    def __iter__(self) -> Iterator:
        return super().__iter__()


@dataset_register.register
class LFSSeqIterableDataset(LFSIterableDataset):
    """
    Large-file-system dataset compatible with sequential protocols.
    """
    def __init__(self,
                 data_root,
                 desc_cfg,
                 used_keys: dict[str, str | dict],
                 seq_mode: bool = True,
                 seq_len: int = 0,
                 seq_split: int = 0,
                 seq_offset: int = 0,
                 shuffle: bool = False,
                 rank_method: str = 'Origin') -> None:
        super().__init__(data_root, desc_cfg, used_keys, SequentialDataDescV0)
        self._desc_cfg: SequentialDataDescV0 = self._desc_cfg
        self._seq_len = seq_len
        self._seq_split = seq_split
        self._seq_offset = seq_offset
        self._shuffle = shuffle
        self._rank_method = RankMethod[rank_method.upper()]

        # Support legacy settings, will be deprecated in the future
        if self._shuffle:
            self._rank_method = RankMethod.RANDOM
        if not seq_mode:
            assert seq_split <= 1, 'seq_split should be less than 1 in legacy seq mode'
            self._seq_split = 1

        self._data_cfg = self._desc_cfg.props

        # self._num_all_samples = self._calculate_total_samples()
        # self._num_samples = self._num_all_samples // DistDataUtils.get_rank_all() * \
        #     DistDataUtils.get_rank_all()
        # self._num_rank_samples = self._num_samples // DistDataUtils.get_rank_all()
        # self._assign_meta_files()
        self._meta_file_descs, self._num_rank_samples = \
            self._distributed_samples_assignment(self._desc_cfg.meta_files)
        Logger().info(f'num_samples: {self._num_rank_samples}')

        # create an empty class
        self._install_packages()

        self._prefetch_pool = None
        self._prefetch_metas = list()
        self._next_meta_idx = 0

        self._cur_meta = MetaInstance(None, LFSMetaDesc(), self._shuffle)

        self._max_cached_metas = 3
        self._prefetch_event = Event()
        self._prefetch_event.clear()
        self._filled_event = Event()
        self._filled_event.clear()
        self._queue_lock = Lock()

    def _install_packages(self):
        pkgs_def = list()
        for _, _v in self._used_keys.items():
            pkgs_def.extend(_v.fetch_packages)
        # import module from given packages
        # e.g. pkgs = [['torch.nn', 'Module']]
        pkgs = type('CPKGS', (), {})
        for _p_def in pkgs_def:
            pkg_module = importlib.import_module(_p_def[0])
            setattr(pkgs, _p_def[1], getattr(pkg_module, _p_def[1]))
        globals()['cpkgs'] = pkgs

    def _distributed_samples_assignment(self, meta_files_cfg: list[MetaSeqFileDescV0]) -> \
        tuple[list[LFSMetaDesc], int]:
        """
        Assign samples to each rank. 
        The whole dataset is consisted of several meta files.
            For each meta file, it contains a dictionary of sequences. Each sequence has
            variable length of frames. During runtime, the sequence will be split into
            several pieces with fixed length. For each rank, it needs to ensure that the
            number of samples is evenly seen by all ranks. Based on this, the needed 
            meta files and the corresponding number of samples will be assigned to each
            rank.

        Returns:
            meta_file_desc: list of tuple (meta_file_path, meta_begin, meta_count), where
                the meta_file_path is the path of meta file, meta_begin is the begin index
                of the meta file, meta_count is the number of samples in the meta file.
            num_rank_samples: number of samples for each rank.
        """
        # Prerequisite
        assert self._seq_split >= 0, 'seq_split should be greater than 0'

        # Step 1: calculate the number of samples for each meta file
        num_pieces_all_metas: list[int] = list()
        desc_all_metas: list[LFSMetaDesc] = list()
        for meta_file_cfg in meta_files_cfg:
            if self._seq_split == 0:
                # sequence split equals to 0 means no sequence split
                num_pieces_all_seqs = np.ones_like(meta_file_cfg.local_nonseq_offset)
                rand_range_all_seqs = np.zeros_like(meta_file_cfg.local_nonseq_offset)
            else:
                num_frames_all_seqs = np.diff(np.concatenate(\
                    [meta_file_cfg.local_nonseq_offset, [meta_file_cfg.num_nonseq_samples]]))
                num_pieces_all_seqs = np.maximum(\
                    0, (num_frames_all_seqs + int(0.5 * self._seq_split)) // self._seq_split - 1)
                rand_range_all_seqs = np.maximum(0, num_frames_all_seqs - \
                    num_pieces_all_seqs * self._seq_split)
            offset_pieces_all_seqs = np.concatenate([[0], np.cumsum(num_pieces_all_seqs)])

            lfs_mata_desc = LFSMetaDesc(\
                pieces_offset=offset_pieces_all_seqs, random_range=rand_range_all_seqs)
            num_pieces_all_metas.append(offset_pieces_all_seqs[-1])
            desc_all_metas.append(lfs_mata_desc)

        # Step 2: calculate the number of samples for each rank
        num_pieces_all_metas = np.asarray(num_pieces_all_metas)
        num_all_samples = np.sum(num_pieces_all_metas)
        num_rank_samples = num_all_samples // DistDataUtils.get_rank_all()

        # Step 3: assign meta files to each rank
        rank_sample_start = DistDataUtils.get_rank_id() * num_rank_samples
        rank_sample_end = (DistDataUtils.get_rank_id() + 1) * num_rank_samples
        offset_pieces_all_metas = np.concatenate([[0], np.cumsum(num_pieces_all_metas)])
        meta_start_idx = np.searchsorted(\
            offset_pieces_all_metas, rank_sample_start, side='right') - 1
        meta_end_idx = np.searchsorted(\
            offset_pieces_all_metas, rank_sample_end, side='left')
        for idx in range(meta_start_idx, meta_end_idx):
            desc_all_metas[idx].file_path = \
                os.path.join(self._data_root, meta_files_cfg[idx].meta_file)
            desc_all_metas[idx].seq_begin = \
                np.maximum(rank_sample_start - offset_pieces_all_metas[idx], 0)
            desc_all_metas[idx].seq_count = \
                np.minimum(rank_sample_end - offset_pieces_all_metas[idx], \
                    offset_pieces_all_metas[idx + 1] - offset_pieces_all_metas[idx])
        rank_meta_files = [os.path.basename(_m.file_path) for _m in desc_all_metas]
        Logger().info(f'rank_metas: {rank_meta_files}')
        rank_meta_offsets = [_m.seq_begin for _m in desc_all_metas]
        rank_meta_counts = [_m.seq_count for _m in desc_all_metas]
        Logger().info(f'meta_start/count: {rank_meta_offsets}/{rank_meta_counts}')
        return desc_all_metas, num_rank_samples

    def _load_meta(self, meta_path: str):
        if meta_path.endswith('.pkl'):
            with open(meta_path, 'rb') as meta_stream:
                meta: dict[str, dict] = pickle.load(meta_stream)
        else:
            raise NotImplementedError
        data_dict = {}
        data_names = []
        for m_name, m_value in meta.items():
            for u_dst, u_src in self._used_keys.items():
                if u_src.raw_key not in m_value.keys():
                    continue
                t_value = m_value[u_src.raw_key]
                if u_src.fetch_transforms is not None:
                    for f_tran in u_src.fetch_transforms:
                        t_value = f_tran(t_value)
                key_list: list = data_dict.setdefault(u_dst, [])
                key_list.append(t_value)
            data_names.append(m_name)
        data_dict['name'] = data_names

        if self._rank_method == RankMethod.SEQ_LEN_ASC or \
            self._rank_method == RankMethod.SEQ_LEN_DESC:
            assert self._seq_split == 0, 'seq_split should be 0 in resorted ASC/DESC mode'
            key_name = list(self._used_keys.keys())[0]
            seq_len = [len(_s) for _s in data_dict[key_name]]
            perm_ids = np.argsort(seq_len)
            if self._rank_method == RankMethod.SEQ_LEN_DESC:
                perm_ids = perm_ids[::-1]
            for d_key, d_value in data_dict.items():
                if isinstance(d_value, np.ndarray):
                    perm_data = d_value[perm_ids]
                elif isinstance(d_value, list):
                    perm_data = [d_value[_p] for _p in perm_ids]
                else:
                    raise NotImplementedError
                data_dict[d_key] = perm_data
        return data_dict

    def _async_fecth(self, future: Future):
        meta_cfg = self._meta_file_descs[self._next_meta_idx]
        fetch_meta = MetaInstance(future.result(), meta_cfg, self._shuffle, self._seq_split)
        with self._queue_lock:
            self._prefetch_metas.append(fetch_meta)
            self._next_meta_idx += 1
            self._next_meta_idx %= len(self._meta_file_descs)
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
                    self._meta_file_descs[self._next_meta_idx].file_path)
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
            if key not in raw_data:
                proc_data[key] = np.asarray(\
                    self._data_cfg[value.raw_key], dtype=getattr(np, value.dtype))
                continue
            proc_data[key] = raw_data[key]
        for d_key, d_value in self._used_keys.items():
            for trans_op in d_value.transforms:
                proc_data[d_key] = trans_op(proc_data[d_key])
            if isinstance(proc_data[d_key], str):
                continue
            proc_data[d_key] = torch.as_tensor(\
                proc_data[d_key], dtype=getattr(torch, d_value.dtype))
        return proc_data

    def __iter__(self):
        for _ in range(self._num_rank_samples):
            yield self._get_item()
