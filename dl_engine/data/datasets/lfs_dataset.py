"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member,logging-fstring-interpolation,eval-used
import os
import json
import pickle
import importlib
import enum
import logging
from dataclasses import dataclass, field
from typing import Iterator, Callable
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

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
    data_pkg: str | None = None
    preloaded: bool = False
    t2v_ratio: int = 1
    transforms: list[Callable] = field(default_factory=list)
    fetch_packages: list[list[str]] = field(default_factory=list)
    fetch_transforms: list[Callable] = field(default_factory=list)

    # def __post_init__(self):
    #     self.transforms = [eval(_t) for _t in self.transforms]  # type: ignore
    #     self.fetch_transforms = [eval(_t) for _t in self.fetch_transforms]  # type: ignore


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
    def __init__(
            self,
            meta_data: dict[str, np.ndarray] | None,
            meta_cfg: LFSMetaDesc,
            shuffle: bool,
            piece_len: int,
            offset_ratio: int,
            seq_offset: int,
            key_attrs: dict[str, dict[str, str]] = None) -> None:
        self._meta_data = meta_data
        self._meta_cfg = meta_cfg
        self._shuffle = shuffle
        self._piece_len = piece_len
        self._offset_ratio = offset_ratio
        self._seq_offset = seq_offset
        self._key_attrs = key_attrs if key_attrs is not None else dict()

        self._cur_idx = 0
        self._index_map = np.arange(meta_cfg.seq_count) + meta_cfg.seq_begin
        if self._shuffle:
            np.random.shuffle(self._index_map)

    def empty(self) -> bool:
        """
        Check if the meta instance is empty.
        """
        return self._meta_data is None

    @staticmethod
    def num_pieces(seqs_num_frames: np.ndarray, piece_len, offset_ratio) -> LFSMetaDesc:
        """
        Calculate the number of pieces for each sequence.
        """
        return np.maximum(0, (seqs_num_frames + int(offset_ratio * piece_len)) // piece_len - 1)

    def loading_seq_data(self):
        """
        Load the sequence data.
        """
        assert self._meta_data is not None
        sel_idx = self._index_map[self._cur_idx]
        seq_idx = np.searchsorted(\
            self._meta_cfg.pieces_offset, sel_idx, side='right') - 1
        seq_bias = sel_idx - self._meta_cfg.pieces_offset[seq_idx]

        seq_data = {_k: _v[seq_idx] for _k, _v in self._meta_data.items()}

        self._cur_idx += 1
        if self._cur_idx >= self._meta_cfg.seq_count:
            self._meta_data = None
        return seq_data, (seq_bias, seq_idx)

    def parse_seq_data(self, seq_data: dict[str, np.ndarray], seq_bias: int, seq_idx: int):
        """
        Parse the sequence data.
        """
        if self._piece_len > 0:
            if self._shuffle:
                const_bias = np.random.randint(0, self._meta_cfg.random_range[seq_idx])
            else:
                const_bias = self._seq_offset
            seq_start = seq_bias * self._piece_len + const_bias
            seq_end = seq_start + self._piece_len
            next_data = {_k: _v[seq_start:seq_end] for _k, _v in seq_data.items()}
            next_data['name'] = self._meta_data['name'] + f'_s{self._seq_offset}_p{seq_bias}'
        next_data['seq_name'] = self._meta_data['name']
        return next_data

    def next(self) -> dict[str, np.ndarray]:
        """
        Get the next meta instance.
        """
        next_seq_data, (sel_bias, seq_idx) = self.loading_seq_data()
        next_data = self.parse_seq_data(next_seq_data, sel_bias, seq_idx)
        return next_data


@dataset_register.register
class LFSSeqIterableDataset(IterableDataset):
    """
    Large-file-system dataset compatible with sequential protocols.
    """
    META_INSTANCE_TYPE = MetaInstance

    def __init__(self,
                 data_root,
                 desc_cfg,
                 used_keys: dict[str, str | dict],
                 seq_mode: bool = True,
                 seq_len: int = 0,
                 seq_split: int = 0,
                 overlap_ratio: float = 0.5,
                 seq_offset: int = 0,
                 max_num_samples: int = 0,
                 shuffle: bool = False,
                 log_level: str = 'INFO',
                 rank_method: str = 'Origin') -> None:
        super().__init__()

        self._desc_cfg_path = os.path.join(data_root, desc_cfg)
        if not os.path.exists(self._desc_cfg_path):
            Logger().warning(f'Description file {self._desc_cfg_path} does not exist')
            self._desc_cfg =  SequentialDataDescV0()
        else:
            with open(self._desc_cfg_path, 'r', encoding='utf-8') as desc_cfg_stream:
                self._desc_cfg = SequentialDataDescV0(**json.load(desc_cfg_stream))

        self._data_root = data_root
        self._used_keys: dict[str, KeyDataDesc] = {
            _k: KeyDataDesc(**_v) if isinstance(_v, dict) else KeyDataDesc(_v) \
                for _k, _v in used_keys.items()
        }

        self._key_attrs = self._gather_key_attrs(data_root, self._used_keys)

        self._desc_cfg: SequentialDataDescV0 = self._desc_cfg
        self._seq_len = seq_len
        self._seq_split = seq_split
        self._overlap_ratio = overlap_ratio
        self._seq_offset = seq_offset
        self._max_num_samples = max_num_samples
        self._shuffle = shuffle
        self._log_level = log_level.upper()
        self._rank_method = RankMethod[rank_method.upper()]

        self._preloaded_data = dict()
        self._data_cfg = self._desc_cfg.props

        self._logged = False

        # Support legacy settings, will be deprecated in the future
        if self._shuffle is not None and self._shuffle:
            self._rank_method = RankMethod.RANDOM
        if not seq_mode:
            assert seq_split <= 1, 'seq_split should be less than 1 in legacy seq mode'
            self._seq_split = 1

        if self._seq_offset > 0:
            assert not self._shuffle, 'seq_offset should be less than 0 in shuffle mode'

        if self._rank_method == RankMethod.RANDOM:
            self._shuffle = True

        self._meta_file_descs, self._num_rank_samples = None, None

        self._prefetch_pool = None
        self._prefetch_metas = list()
        self._next_meta_idx = 0
        self._cur_meta = None
        self._max_cached_metas = 3
        self._prefetch_event = None
        self._filled_event = None
        self._queue_lock = None

        # create an empty class
        self._install_packages()

        # preloading data
        self._preloading_data()

    def _gather_key_attrs(self, data_root: str, used_keys: dict[str, KeyDataDesc]):
        if not os.path.exists(os.path.join(data_root)):
            return dict()

        reverse_map = {_v.raw_key: _k for _k, _v in used_keys.items()}
        all_descs_files = [os.path.join(data_root, _f) for _f in os.listdir(data_root) \
            if str(_f).endswith('.json')]
        all_descs_cfg: list[SequentialDataDescV0] = list()
        for desc_file in all_descs_files:
            with open(desc_file, 'r', encoding='utf-8') as desc_stream:
                all_descs_cfg.append(SequentialDataDescV0(**json.load(desc_stream)))
        key_attrs = {}
        for desc_cfg in all_descs_cfg:
            local_key_attrs = getattr(desc_cfg.meta_files[0], 'key_attrs', {})
            key_attrs.update(local_key_attrs)
        key_attrs = {\
            reverse_map[_k] if _k in reverse_map else _k: _v for _k, _v in key_attrs.items()}
        return key_attrs

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

    def _preloading_data(self):
        for used_key, used_value in self._used_keys.items():
            if not used_value.preloaded:
                continue
            assert used_value.data_pkg is not None, \
                f'Package for {used_key} is not specified'
            with open(os.path.join(self._data_root, used_value.data_pkg), 'rb') as data_stream:
                data_dict: dict = pickle.load(data_stream)
            for d_key, d_value in data_dict.items():
                self._preloaded_data.setdefault(d_key, dict())[used_key] = d_value[used_key]

    def _calculate_meta_num_pieces(self, meta_file_cfg: MetaSeqFileDescV0) \
        -> tuple[LFSMetaDesc, int]:
        """
        Calculate the number of pieces for each sequence of given meta file.

        Returns:
            meta_desc: meta description, containing the number of pieces and
                random offset for each sequence.
            total_pieces: total number of pieces of given meta file.
        """
        if self._seq_split == 0:
            # sequence split equals to 0 means no sequence split
            num_pieces_all_seqs = np.ones_like(meta_file_cfg.local_nonseq_offset)
            rand_range_all_seqs = np.zeros_like(meta_file_cfg.local_nonseq_offset)
        else:
            num_frames_all_seqs = np.diff(np.concatenate(\
                [meta_file_cfg.local_nonseq_offset, [meta_file_cfg.num_nonseq_samples]]))
            num_pieces_all_seqs = self.META_INSTANCE_TYPE.num_pieces(\
                num_frames_all_seqs, self._seq_split, self._overlap_ratio)
            rand_range_all_seqs = np.maximum(0, num_frames_all_seqs - \
                num_pieces_all_seqs * self._seq_split)
        offset_pieces_all_seqs = np.concatenate([[0], np.cumsum(num_pieces_all_seqs)])

        lfs_mata_desc = LFSMetaDesc(\
            pieces_offset=offset_pieces_all_seqs, random_range=rand_range_all_seqs)
        lfs_num_pieces = offset_pieces_all_seqs[-1]
        return lfs_mata_desc, lfs_num_pieces

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
            lfs_mata_desc, num_pieces_meta = self._calculate_meta_num_pieces(meta_file_cfg)
            desc_all_metas.append(lfs_mata_desc)
            num_pieces_all_metas.append(num_pieces_meta)

        # Step 2: calculate the number of samples for each rank
        num_devices = DistDataUtils.get_rank_all()
        rank_device = DistDataUtils.get_rank_id()
        work_info = get_worker_info()
        if work_info is None:
            num_workers = 1
            rank_worker = 0
        else:
            num_workers = work_info.num_workers
            rank_worker = work_info.id
        total_splits = num_devices * num_workers
        rank_split = rank_device * num_workers + rank_worker

        num_pieces_all_metas = np.asarray(num_pieces_all_metas)
        num_all_samples = int(np.sum(num_pieces_all_metas))
        num_rank_samples = num_all_samples // total_splits

        # Step 3: assign meta files to each rank
        rank_sample_start = rank_split * num_rank_samples
        rank_sample_end = (rank_split + 1) * num_rank_samples
        offset_pieces_all_metas = np.concatenate([[0], np.cumsum(num_pieces_all_metas)])
        meta_start_idx = np.searchsorted(\
            offset_pieces_all_metas, rank_sample_start, side='right') - 1
        meta_end_idx = np.searchsorted(\
            offset_pieces_all_metas, rank_sample_end, side='left')
        desc_rank_metas: list[LFSMetaDesc] = list()
        for idx in range(meta_start_idx, meta_end_idx):
            raw_meta_cfg = desc_all_metas[idx]
            desc_rank_meta = LFSMetaDesc(pieces_offset=raw_meta_cfg.pieces_offset, \
                random_range=raw_meta_cfg.random_range, file_path=meta_files_cfg[idx].meta_file)
            desc_rank_meta.file_path = \
                os.path.join(self._data_root, meta_files_cfg[idx].meta_file)
            desc_rank_meta.seq_begin = \
                np.maximum(rank_sample_start - offset_pieces_all_metas[idx], 0)
            seq_end = \
                np.minimum(rank_sample_end - offset_pieces_all_metas[idx], \
                    offset_pieces_all_metas[idx + 1] - offset_pieces_all_metas[idx])
            desc_rank_meta.seq_count = seq_end - desc_rank_meta.seq_begin
            desc_rank_metas.append(desc_rank_meta)
        rank_meta_files = [os.path.basename(_m.file_path) for _m in desc_rank_metas]
        Logger().debug(f'Rank D{rank_device}W{rank_worker} - num_samples: {num_rank_samples}')
        Logger().debug(f'Rank D{rank_device}W{rank_worker} - metas: {rank_meta_files}')
        rank_meta_offsets = [_m.seq_begin for _m in desc_rank_metas]
        rank_meta_counts = [_m.seq_count for _m in desc_rank_metas]
        out_str = f'Rank D{rank_device}W{rank_worker} '
        out_str += f'start/count: {rank_meta_offsets}/{rank_meta_counts}'
        Logger().debug(out_str)
        return desc_rank_metas, num_rank_samples

    def _load_meta(self, meta_path: str):
        if meta_path.endswith('.pkl'):
            with open(meta_path, 'rb') as meta_stream:
                meta: dict[str, dict] = pickle.load(meta_stream)
        else:
            raise NotImplementedError(f'Unsupported meta file: {meta_path}')

        # Fetch data from separate data packages
        attached_metas = dict()
        key_metas_dict: dict[str, dict] = dict()  # The used key name and its belonging meta
        for _, u_value in self._used_keys.items():
            if u_value.data_pkg is not None and not u_value.preloaded:
                meta_name = os.path.splitext(os.path.basename(meta_path))[0]
                meta_index = meta_name.split('_')[-1]
                attached_file = u_value.data_pkg.replace('*', meta_index)
                if attached_file not in attached_metas:
                    with open(os.path.join(os.path.dirname(\
                        meta_path), attached_file), 'rb') as data_stream:
                        attached_metas[attached_file] = pickle.load(data_stream)
                key_metas_dict[u_value.raw_key] = \
                    {_k: _v[u_value.raw_key] for _k, _v in attached_metas[attached_file].items()}

        data_dict = {}
        data_names = []
        for m_name, m_value in meta.items():
            for u_dst, u_src in self._used_keys.items():
                if u_src.raw_key not in m_value.keys():
                    if u_src.raw_key not in key_metas_dict:
                        continue
                    t_value = key_metas_dict[u_src.raw_key][m_name]
                else:
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
        fetch_meta = self.META_INSTANCE_TYPE(\
            future.result(),
            meta_cfg,
            self._shuffle,
            self._seq_split,
            self._overlap_ratio,
            self._seq_offset,
            key_attrs=self._key_attrs)
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
                if self._used_keys[key].preloaded:
                    proc_data[key] = self._preloaded_data[raw_data['seq_name']][key]
                else:
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

    def _lazy_init(self):
        """
        Lazy initialization for multi-processing.
        """
        if not self._logged:
            logging.basicConfig(level=getattr(logging, self._log_level))
            self._meta_file_descs, self._num_rank_samples = \
                self._distributed_samples_assignment(self._desc_cfg.meta_files)
            self._logged = True

        self._prefetch_pool = None
        self._prefetch_metas = list()
        self._next_meta_idx = 0

        self._cur_meta = self.META_INSTANCE_TYPE(\
            None, LFSMetaDesc(), self._shuffle, 0, 0, 0)

        self._max_cached_metas = 3
        self._prefetch_event = Event()
        self._prefetch_event.clear()
        self._filled_event = Event()
        self._filled_event.clear()
        self._queue_lock = Lock()

        for _, value in self._used_keys.items():
            value.transforms = [eval(_t) for _t in value.transforms]
            value.fetch_transforms = [eval(_t) for _t in value.fetch_transforms]

    def __len__(self):
        if self._meta_file_descs is None or self._num_rank_samples is None:
            self._lazy_init()
        return self._num_rank_samples

    def __iter__(self):
        if self._meta_file_descs is None or self._num_rank_samples is None:
            self._lazy_init()

        if self._max_num_samples > 0:
            self._num_rank_samples = min(self._max_num_samples, self._num_rank_samples)

        for _ in range(self._num_rank_samples):
            yield self._get_item()

    def __getitem__(self, index):
        Logger().warning('Not implemented')
        raise NotImplementedError


@dataset_register.register
class MultiLFSIterableDatasets(IterableDataset):
    """
    Multi large-file-system dataset.
    """
    def __init__(self, datasets: list[LFSSeqIterableDataset]) -> None:
        super().__init__()
        self._datasets = datasets

        self._num_samples = list()
        self._left_samples = list()

    def __getitem__(self, index):
        Logger().warning('Not implemented')
        raise NotImplementedError

    def _get_item(self, data_iters: list[Iterator]):
        if np.sum(self._left_samples) == 0:
            raise StopIteration
        prob = self._left_samples / np.sum(self._left_samples)
        choice = np.random.multinomial(1, prob).argmax()
        assert self._left_samples[choice] > 0, \
            f'No more samples for dataset {choice}'
        self._left_samples[choice] -= 1
        return next(data_iters[choice])

    def __iter__(self):
        self._num_samples = np.asarray([len(_d) for _d in self._datasets])
        self._left_samples = self._num_samples.copy()
        data_iters = [iter(_d) for _d in self._datasets]
        while True:
            try:
                yield self._get_item(data_iters)
            except StopIteration:
                break
