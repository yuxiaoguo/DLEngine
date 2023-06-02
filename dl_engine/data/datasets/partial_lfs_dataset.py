"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member,logging-fstring-interpolation
import os
import re
import json
import pickle
import logging

import torch
import numpy as np
from torch.utils import data
from torch import distributed as dist

from dl_engine.core.register import dataset_register
from dl_engine.core.logger import Logger

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


@dataset_register.register
class PartialLFSDataset(data.Dataset):
    """
    The class is used to register the RAVDESS dataset.
    """
    def __init__(self, data_root, used_keys: dict[str, str], seq_mode: bool, seq_len=0,
        shuffle=False, transform=None):
        """
        Args:
            data_root (string): Path to the data root.
            used_keys (dict): The keys used in the dataset.
            seq_mode (bool): Whether to use sequence mode.
            shuffle (bool): Whether to shuffle the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self._seq_mode = seq_mode
        self._shuffle = shuffle
        self._transform = transform
        self._data_root = data_root
        self._seq_len = seq_len

        self._desc_file = os.path.join(data_root, 'desc.pkl')
        self._desc_key = 'sequential' if seq_mode else 'frame'

        if dist.is_available() and dist.is_initialized():
            self._rank_id = dist.get_rank()
            self._rank_all = dist.get_world_size()
        else:
            self._rank_id = 0
            self._rank_all = 1

        with open(self._desc_file, 'rb') as desc_file_stream:
            self._desc_dict = pickle.load(desc_file_stream)
        self._num_samples = self._desc_dict[f'{self._desc_key}_total']
        self._rank_start = self._rank_id * self._num_samples // self._rank_all
        self._rank_end = (self._rank_id + 1) * self._num_samples // self._rank_all
        self._rank_samples = self._rank_end - self._rank_start
        Logger().info(f'Rank {self._rank_id}: {self._rank_samples}' +\
            f' between {self._rank_start}-{self._rank_end}')
        all_offset = self._desc_dict[f'{self._desc_key}_offset']
        fidx_start = 0
        fidx_end = 0
        file_start = 0
        for o_idx, offset_item in enumerate(all_offset):
            if offset_item[1] <= self._rank_start:
                fidx_start = o_idx
                file_start = offset_item[1]
            if offset_item[1] < self._rank_end:
                fidx_end = o_idx
        self._rank_offset = self._rank_start - file_start
        Logger().info(f'Rank {self._rank_id}: pkl starts {file_start}' +\
            f' and offset {self._rank_offset}')
        self._data_pkl_files = [_f[0] for _f in all_offset[fidx_start:fidx_end+1]]

        # Load the data description and data
        assert transform is None, 'The transform is not supported yet.'
        desc_json_files = list()
        for file_name in os.listdir(data_root):
            file_ext = os.path.splitext(file_name)[-1]
            if file_ext == '.json':
                desc_json_files.append(file_name)
        assert desc_json_files and self._data_pkl_files, 'The data is not ready.'

        # load the data description and merge split items
        data_desc: dict[str, list] = dict()
        for desc_json_file in desc_json_files:
            desc_json_path = os.path.join(data_root, desc_json_file)
            with open(desc_json_path, 'r', encoding='utf-8') as json_file:
                data_desc_single: dict = json.load(json_file)
            for key, value in data_desc_single.items():
                data_desc.setdefault(key, list()).append(value)
        self._data_desc = dict()
        for key, value in data_desc.items():
            value = np.asarray(value)
            if key.endswith('min'):
                LOG.info(f'Rank {self._rank_id}: Key {key} is applied with reduce_min operator.')
                value = np.min(value, axis=0)
            else:
                LOG.info(f'Rank {self._rank_id}: Key {key} is applied with reduce_max operator.')
                value = np.max(value, axis=0)
            self._data_desc[key] = value

        # Split the used keys into two parts: saved in data and generated runtime.
        self._cached_kv_pairs = dict()
        self._gen_kv_pairs = dict()
        for u_key, u_value in used_keys.items():
            gen_flag = re.findall(r'\@(.*)\@(.*)', u_value)
            assert len(gen_flag) <= 1, 'The key should not contain more than one gen flag.'
            if not gen_flag:
                self._cached_kv_pairs[u_key] = u_value
                continue
            gen_method, gen_key = gen_flag[0]
            self._gen_kv_pairs[gen_key] = (u_key, gen_method)

        # Cache used data
        self._cached_data: dict[str, list] = dict()
        self._initialized = False

    def _cache_all_data(self):
        if self._initialized:
            return
        self._initialized = True
        for data_pkl_file in self._data_pkl_files:
            data_pkl_path = os.path.join(self._data_root, data_pkl_file)
            LOG.info(f'Rank {self._rank_id}: Loading data from {data_pkl_path}')
            with open(data_pkl_path, 'rb') as pkl_file:
                raw_data: dict[str, dict] = pickle.load(pkl_file)
            for _, date_data in raw_data.items():
                for i_key, i_value in date_data.items():
                    if i_key not in self._cached_kv_pairs.values():
                        continue
                    i_value = self._auto_norm_vec(i_key, i_value)
                    if self._seq_mode:
                        self._cached_data.setdefault(i_key, []).append(i_value)
                    else:
                        self._cached_data.setdefault(i_key, []).extend(i_value)

    def __len__(self):
        return self._num_samples

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
        key_length = target_value.shape[0]
        normed_value = \
            (target_value.reshape([key_length, -1, key_dims]) - box_cnt) / box_size
        normed_value = normed_value.reshape([key_length, -1])
        return normed_value

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        self._cache_all_data()
        sample_out = dict()
        r_idx = idx
        idx = idx - self._rank_start + self._rank_offset
        if idx < self._rank_offset or idx >= self._rank_offset + self._rank_samples:
            Logger().warning(f'Rank {self._rank_id}: {idx}/{r_idx} is out of {self._rank_samples}.')
            idx = idx % self._rank_samples
        for dst_key, src_key in self._cached_kv_pairs.items():
            np_data: np.ndarray = self._auto_random_clip(self._cached_data[src_key][idx])
            sample_out[dst_key] = torch.as_tensor(self._auto_padding(np_data))
            if src_key in self._gen_kv_pairs:
                gen_key, gen_method = self._gen_kv_pairs[src_key]
                if gen_method == 'padding_mask':
                    gen_data = self._auto_padding(np.ones(np_data.shape[:1], np.int32))
                else:
                    raise NotImplementedError(f'The gen method {gen_method} is not supported.')
                sample_out[gen_key] = torch.as_tensor(gen_data)
        return sample_out
