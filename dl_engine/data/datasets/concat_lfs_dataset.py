"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member,logging-fstring-interpolation
import torch
import numpy as np
from torch.utils import data
from torch import distributed as dist

from dl_engine.core.register import dataset_register
from dl_engine.core.logger import Logger

from .partial_lfs_dataset import PartialLFSDataset


@dataset_register.register
class ConcatLFSDataset(data.Dataset):
    """
    Concatenate multiple LFS datasets into a single one.
    """
    def __init__(self, datasets: list[PartialLFSDataset]) -> None:
        super().__init__()
        self.datasets = datasets

        self._merge_and_update_descs()

        self._rank_all = dist.get_world_size() if dist.is_available() \
            and dist.is_initialized() else 1
        self._rank_id = dist.get_rank() if dist.is_available() \
            and dist.is_initialized() else 0
        self._rank_samples = len(self) // self._rank_all
        self._rank_dataset_samples = [len(d) // self._rank_all for d in self.datasets]

    def _merge_and_update_descs(self):
        descs = [d.data_desc for d in self.datasets]
        assert len(descs) > 0, 'No dataset is provided.'
        keys: list[str] = list(descs[0].keys())
        merged_desc = dict()
        for key in keys:
            values = [d[key] for d in descs]
            method_str = key.split('_')[-1]
            if method_str not in ['max', 'min']:
                method_str = 'max'
            method = getattr(np, method_str)
            Logger().info_zero_rank(f'{key}: Processing with {method} applied on {len(values)}.')
            merged_desc[key] = method(np.stack(values, axis=0), axis=0).tolist()
        for dataset in self.datasets:
            dataset.data_desc = merged_desc

    def __len__(self) -> int:
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, index: int) -> dict:
        rank_index = index % self._rank_samples
        for dataset, rank_data_sample in zip(self.datasets, self._rank_dataset_samples):
            if rank_index < rank_data_sample:
                return dataset[dataset.rank_info.rank_start + rank_index]
            rank_index -= rank_data_sample
        raise IndexError(f'Index {index} is out of range.')
