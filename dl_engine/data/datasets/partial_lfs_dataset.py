"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member,logging-fstring-interpolation
import os
import re
import json
import zipfile
import logging
import pickle

import torch
import numpy as np
from torch.utils import data
from torch import distributed as dist

from dl_engine.core.register import dataset_register
from dl_engine.core.logger import Logger

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class RankSamplesInfo:
    """
    The information of the samples in the rank.
    """
    def __init__(self) -> None:
        self.num_all_samples = 0
        self.num_rank_samples = 0
        self.rank_start = 0
        self.rank_end = 0
        self.rank_offset = 0
        self.rank_zip_files = list()


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

        self._seq_mode = seq_mode
        self._shuffle = shuffle
        self._transform = transform
        self._seq_len = seq_len

        self._desc_file = os.path.join(data_root, desc_cfg)
        self._desc_key = 'sequential' if seq_mode else 'frame'

        if dist.is_available() and dist.is_initialized():
            self._rank_id = dist.get_rank()
            self._rank_all = dist.get_world_size()
        else:
            self._rank_id = 0
            self._rank_all = 1

        with open(self._desc_file, 'r', encoding='utf-8') as desc_file_stream:
            self._desc_dict = json.load(desc_file_stream)

        # Generate the rank information
        self.rank_info = self._gen_rank_info()

        # Build zip mapping
        self.zip_mapping, self.zip_files = self._build_zip_mapping()

        # load the data description and merge split items
        self._data_desc = self._desc_dict.copy()

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

    def _gen_rank_info(self) -> RankSamplesInfo:
        """
        Generate the rank information.
        """
        rank_info = RankSamplesInfo()

        # Calculate the number of samples and the offset
        rank_info.num_all_samples = self._desc_dict[f'{self._desc_key}_total']
        rank_info.rank_start = self._rank_id * rank_info.num_all_samples // self._rank_all
        rank_info.rank_end = (self._rank_id + 1) * rank_info.num_all_samples // self._rank_all
        rank_info.num_rank_samples = rank_info.rank_end - rank_info.rank_start
        Logger().info(f'Rank {self._rank_id}: {rank_info.num_rank_samples}' +\
            f' between {rank_info.rank_start}-{rank_info.rank_end}')
        all_offset = self._desc_dict[f'{self._desc_key}_offset']
        fidx_start = 0
        fidx_end = 0
        file_start = 0
        for o_idx, offset_item in enumerate(all_offset):
            if offset_item[1] <= rank_info.rank_start:
                fidx_start = o_idx
                file_start = offset_item[1]
            if offset_item[1] < rank_info.rank_end:
                fidx_end = o_idx
        rank_info.rank_offset = rank_info.rank_start - file_start
        Logger().info(f'Rank {self._rank_id}: file starts {file_start}' +\
            f' and offset {rank_info.rank_offset}')
        rank_info.rank_zip_files = [f'{_f[0]}.zip' for _f in all_offset[fidx_start:fidx_end+1]]

        return rank_info

    @staticmethod
    def _func_file2zip_ids(zip_file: zipfile.ZipFile):
        """
        The function is to give each sample (or file) a global unique name and
            thus to locate the sample to the zip file and its index in later use. 
        """
        return zip_file.namelist()

    def _func_id2file(self, zip_file: zipfile.ZipFile, uni_id: str, src_key='',
        dst_key=''):
        """
        Query the content of the zip file with the given id (generated by
            _func_file2zip_ids). Key is used to query the specific content in
            the unique content, such as multiple images or keys stored in a
            single content. Key will be one of the keys in the `used_keys`.
        """
        out_dict = dict()

        meta_data = pickle.loads(zip_file.read(uni_id))
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

    def _build_zip_mapping(self):
        zip_files = list()
        zip_files_path = self.rank_info.rank_zip_files
        for zip_file in zip_files_path:
            assert os.path.exists(os.path.join(self._data_root, zip_file)),\
                f'Zip file {zip_file} does not exist.'
            zip_files.append(zipfile.ZipFile(os.path.join(self._data_root, zip_file), 'r'))
        all_file_maps = list()
        for zip_idx, zip_file in enumerate(zip_files):
            all_file_maps.extend([(_f, zip_idx) for _f in self._func_file2zip_ids(zip_file)])
        partial_file_maps = all_file_maps[self.rank_info.rank_offset:\
            self.rank_info.rank_offset + self.rank_info.num_rank_samples]
        return partial_file_maps, zip_files

    def __len__(self):
        return self.rank_info.num_all_samples

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
        sample_out = dict()
        r_idx = idx
        idx = idx - self.rank_info.rank_start
        if idx < 0 or idx >= self.rank_info.num_rank_samples:
            Logger().warning(\
                f'Rank {self._rank_id}: {idx}/{r_idx} is out of {self.rank_info.num_rank_samples}.')
            idx = (0 if idx < 0 else idx) % self.rank_info.num_rank_samples
        sample_out = dict()
        uni_id, zip_file_idx = self.zip_mapping[idx]
        for dst_key, src_key in self._cached_kv_pairs.items():
            sample_out = {**sample_out, **self._func_id2file(\
                self.zip_files[zip_file_idx], uni_id, src_key, dst_key)}
        return sample_out
