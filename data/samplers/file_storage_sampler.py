"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=no-member
import math

import torch
from torch import distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from dl_engine.core.register import sampler_register
from dl_engine.core.logger import Logger


@sampler_register.register
class FileStorageSampler(DistributedSampler):
    """
    The class is used to register the file storage sampler.
    """
    def __init__(self, dataset: Dataset, num_replicas: int | None = None,
        rank: int | None = None, shuffle: bool = True, seed: int = 0,
        drop_last: bool = False) -> None:
        if not dist.is_available() or not dist.is_initialized():
            rank = 0
            num_replicas = 1
        Logger().info(f'Rank {rank}: FileStorageSampler initialized.')
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self):
        indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank * self.num_samples:(self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed + self.epoch)
            shuffle_indices = torch.randperm(len(indices), generator=gen).tolist()
            indices = [indices[_i] for _i in shuffle_indices]

        return iter(indices)
