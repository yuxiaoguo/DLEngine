"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member,logging-fstring-interpolation
import os
import re
import json
import zipfile
import pickle
from io import BytesIO

import torch
import numpy as np
from torch.utils import data
from torch import distributed as dist

from dl_engine.data.protocols import SequentialDataDescV0, MetaSeqFileDescV0
from dl_engine.core.register import dataset_register
from dl_engine.core.logger import Logger


class RankSamplesInfo:
    """
    The information of the samples in the rank.
    """
    def __init__(self) -> None:
        self.rank_id = 0
        self.rank_all = 1

        self.num_all_samples = 0
        self.num_rank_samples = 0
        self.rank_start = 0
        self.rank_end = 0
        self.rank_offset = 0
        self.rank_meta_files = list()

    def _init_rank_info(self):
        if dist.is_available() and dist.is_initialized():
            self.rank_id = dist.get_rank()
            self.rank_all = dist.get_world_size()
        else:
            self.rank_id = 0
            self.rank_all = 1

    def _init_rank_samples_desc(self, num_all_samples, offsets, meta_files):
        # Calculate the number of samples and the offset
        self.num_all_samples = num_all_samples
        self.rank_start = self.rank_id * self.num_all_samples // self.rank_all
        self.rank_end = (self.rank_id + 1) * self.num_all_samples // self.rank_all
        self.num_rank_samples = self.rank_end - self.rank_start
        Logger().info(f'Rank {self.rank_id}: {self.num_rank_samples}' +\
            f' between {self.rank_start}-{self.rank_end}')
        all_offset = offsets
        fidx_start = 0
        fidx_end = 0
        file_start = 0
        for o_idx, offset_item in enumerate(all_offset):
            if offset_item <= self.rank_start:
                fidx_start = o_idx
                file_start = offset_item
            if offset_item < self.rank_end:
                fidx_end = o_idx
        self.rank_offset = self.rank_start - file_start
        Logger().info(f'Rank {self.rank_id}: file starts {file_start}' +\
            f' and offset {self.rank_offset}')
        self.rank_meta_files = meta_files[fidx_start:fidx_end+1]
        return self

    def init(self, num_all_samples, offsets, meta_files):
        """
        Initialize the rank information.
        """
        self._init_rank_info()
        self._init_rank_samples_desc(num_all_samples, offsets, meta_files)
        return self


@dataset_register.register
class PartialLFSDataset(data.Dataset):
    """
    The dataset to deal with samples stored as large zip files. This dataset shows
        the case of sequential organized data and load samples saved as pickle
        file in ZIP. If you want to use this dataset, please reimplment the function
        `_func_file2zip_ids` and `_func_id2file`. `_func_file2zip_ids` is used to
        build the global sample via a unique key to ZIP file correspondance.
        `_func_id2file` is used to load the sample from predefined unique key. If
        you data is not organized as sequential, please set `seq_mode` to `True`.
        Since the original usage is organized as sequential, we use `seq_mode` to
        control whether to unstack the first dimension.

    The target folder stored ZIPs should be organized as:
        - data_root
            - $desc_cfg$  # The description file, should be a json.
            - *_0.zip
            - *_1.zip
            - ...
    
    For the description file, it should be a json file with the following format:
        {
            "sequential_offset": [
                [*_0, offset],         # The offset of the first sample in *_0.zip.
                [*_1, offset],
                ...
            ]
            "sequential_total": int    # The total number of the sequential data.
        }

    An example to jointly use this dataset and `FileStorageSampler` in Lightning.Fabric:
    ``` python
    from lightning import Fabric
    fabric = Fabric()
    fabric.launch()
    dataset = PartialLFSDataset(...)
    sampler = FileStorageSampler(dataset, ...)
    dataloader = DataLoader(dataset, sampler=sampler, ...)
    fabric.register_dataloader(dataloader, ...)
    ```
    """
    def __init__(self, data_root, desc_cfg, used_keys: dict[str, str],
        seq_mode: bool, cache=False, seq_len=0, shuffle=False, transform=None):
        """
        Args:
            data_root (string): Path to the data root.
            used_keys (dict): The keys used in the dataset.
            seq_mode (bool): Whether to use sequence mode.
            shuffle (bool): Whether to shuffle the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self._data_root = data_root
        self._desc_cfg = desc_cfg
        self._cache = cache

        self._used_keys = used_keys
        self._seq_mode = seq_mode
        self._shuffle = shuffle
        self._transform = transform
        self._seq_len = seq_len

        self._desc_file = os.path.join(data_root, desc_cfg)
        with open(self._desc_file, 'r', encoding='utf-8') as desc_file_stream:
            self._cfg_desc = SequentialDataDescV0(**json.load(desc_file_stream))

        # load the data description and merge split items
        self._data_desc = self._cfg_desc.props

        # Generate the rank information
        total_samples = self._cfg_desc.total_samples if seq_mode \
            else self._cfg_desc.total_nonseq_samples
        offsets = [_f.global_offset if seq_mode else _f.global_nonseq_offset\
            for _f in self._cfg_desc.meta_files]
        meta_file_names = [_f.meta_file for _f in self._cfg_desc.meta_files]
        self._rank_info = RankSamplesInfo().init(total_samples, offsets, meta_file_names)

        # Build zip mapping
        self._meta_mapping, self._meta_data = None, None

        # Split the used keys into two parts: saved in data and generated runtime.
        self._cached_kv_pairs, self._gen_kv_pairs = self._build_implicit_kv_map()

    def _func_file2zip_ids(self, meta_file: zipfile.ZipFile | dict, zip_cfg: MetaSeqFileDescV0):
        """
        The function is to give each sample (or file) a global unique name and
            thus to locate the sample to the zip file and its index in later use. 
        """
        if self._seq_mode:
            return meta_file.namelist()

        frame_ids = list()
        offsets = zip_cfg.local_nonseq_offset + [zip_cfg.num_nonseq_samples]

        saved_keys = meta_file.namelist() if isinstance(meta_file, zipfile.ZipFile) \
            else meta_file.keys()
        for _f, _o, _e in zip(saved_keys, offsets[:-1], offsets[1:]):
            frame_ids.extend([(_f, _i - _o) for _i in range(_o, _e)])
        return frame_ids

    def _func_id2file(self, meta_data: zipfile.ZipFile | dict, uni_id: str, src_key='',
        dst_key=''):
        """
        Query the content of the zip file with the given id (generated by
            _func_file2zip_ids). Key is used to query the specific content in
            the unique content, such as multiple images or keys stored in a
            single content. Key will be one of the keys in the `used_keys`.
        """
        def _get_meta(_meta_data: zipfile.ZipFile | dict, _uni_id: str) -> dict:
            if isinstance(_meta_data, zipfile.ZipFile):
                return pickle.loads(_meta_data.read(_uni_id))
            elif isinstance(_meta_data, dict):
                return _meta_data[_uni_id]
            else:
                raise NotImplementedError(f'The type of {type(_meta_data)} is not supported.')

        out_dict = dict()

        if isinstance(uni_id, str):
            meta_data = _get_meta(meta_data, uni_id)
        elif isinstance(uni_id, tuple):
            meta_data = _get_meta(meta_data, uni_id[0])
            meta_data = {_k: _v[uni_id[1]] for _k, _v in meta_data.items()}
        else:
            raise NotImplementedError(f'The type of uni_id {type(uni_id)} is not supported.')

        normed_data = self._auto_norm_vec(src_key, meta_data[src_key])
        np_data: np.ndarray = self._auto_random_clip(normed_data)

        out_dict[dst_key] = torch.as_tensor(self._auto_padding(np_data))
        if src_key in self._gen_kv_pairs:
            gen_key, gen_method = self._gen_kv_pairs[src_key]
            if gen_method == 'padding_mask':
                gen_data = self._auto_padding(np.ones(np_data.shape[:1], np.int32))
            else:
                raise NotImplementedError(f'The gen method {gen_method} is not supported.')
            out_dict[gen_key] = torch.as_tensor(gen_data)
        return out_dict

    def _build_implicit_kv_map(self):
        cached_kv_pairs = dict()
        gen_kv_pairs = dict()
        for u_key, u_value in self._used_keys.items():
            gen_flag = re.findall(r'\@(.*)\@(.*)', u_value)
            assert len(gen_flag) <= 1, 'The key should not contain more than one gen flag.'
            if not gen_flag:
                cached_kv_pairs[u_key] = u_value
                continue
            gen_method, gen_key = gen_flag[0]
            gen_kv_pairs[gen_key] = (u_key, gen_method)
        return cached_kv_pairs, gen_kv_pairs

    def _build_meta_mapping(self):
        meta_data_list = list()
        meta_file_names: list[str] = self._rank_info.rank_meta_files
        for meta_file_name in meta_file_names:
            assert os.path.exists(os.path.join(self._data_root, meta_file_name)),\
                f'Meta file {meta_file_name} does not exist.'
            Logger().info(f'Loading meta file {meta_file_name}')
            if meta_file_name.endswith('.zip'):
                with open(os.path.join(self._data_root, meta_file_name), 'rb') as zip_file_stream:
                    zip_io_meta = BytesIO(zip_file_stream.read())
                    meta_data = zipfile.ZipFile(zip_io_meta, 'r')
            elif meta_file_name.endswith('.pkl'):
                with open(os.path.join(self._data_root, meta_file_name), 'rb') as pkl_file_stream:
                    meta_data = pickle.load(pkl_file_stream)
            meta_data_list.append(meta_data)
        all_meta_map = list()
        for zip_idx, zip_file in enumerate(meta_data_list):
            meta_file, = [_f for _f in self._cfg_desc.meta_files \
                if _f.meta_file == meta_file_names[zip_idx]]
            all_meta_map.extend([(_f, zip_idx) for _f in \
                self._func_file2zip_ids(zip_file, meta_file)])
        partial_meta_map = all_meta_map[self._rank_info.rank_offset:\
            self._rank_info.rank_offset + self._rank_info.num_rank_samples]
        return partial_meta_map, meta_data_list

    def __len__(self):
        return self._rank_info.num_all_samples

    def _auto_random_clip(self, target_data: np.ndarray):
        if not self._seq_mode or self._seq_len == 0:
            return target_data
        if target_data.shape[0] <= self._seq_len:
            return target_data
        start_idx = np.random.randint(0, target_data.shape[0] - self._seq_len)
        end_idx = start_idx + self._seq_len
        return target_data[start_idx:end_idx]

    def _auto_padding(self, target_data: np.ndarray):
        if not self._seq_mode or self._seq_len == 0:
            return target_data
        rest_num_dims = target_data.ndim - 1
        padding_dim = (0, self._seq_len - target_data.shape[0])
        rest_dims = [(0, 0)] * rest_num_dims
        target_data = np.pad(target_data, (padding_dim, *rest_dims), 'constant')
        return target_data

    def _auto_norm_vec(self, target_key: str, target_value: np.ndarray):
        if f'{target_key}_max' not in self._data_desc or f'{target_key}_min' \
            not in self._data_desc:
            return target_value
        box_max = np.asarray(self._data_desc[f'{target_key}_max'], dtype=np.float32)
        box_min = np.asarray(self._data_desc[f'{target_key}_min'], dtype=np.float32)
        box_cnt = (box_max + box_min) / 2
        box_size = np.max(box_max - box_cnt)
        key_dims = box_max.shape[0]
        key_raw_shape = target_value.shape
        normed_value = \
            (target_value.reshape([-1, key_dims]) - box_cnt) / box_size
        normed_value = normed_value.reshape(key_raw_shape)
        return normed_value

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        if self._meta_mapping is None or self._meta_data is None:
            self._meta_mapping, self._meta_data = self._build_meta_mapping()

        sample_out = dict()
        r_idx = idx
        idx = idx - self._rank_info.rank_start
        if idx < 0 or idx >= self._rank_info.num_rank_samples:
            log_str = f'Rank {self._rank_info.rank_id}:'
            log_str += f' {r_idx}/{idx} is out of {self._rank_info.num_rank_samples}.'
            Logger().warning(log_str)
            idx = (0 if idx < 0 else idx) % self._rank_info.num_rank_samples
        sample_out = dict()
        uni_id, zip_file_idx = self._meta_mapping[idx]
        for dst_key, src_key in self._cached_kv_pairs.items():
            sample_out = {**sample_out, **self._func_id2file(\
                self._meta_data[zip_file_idx], uni_id, src_key, dst_key)}
        return sample_out
