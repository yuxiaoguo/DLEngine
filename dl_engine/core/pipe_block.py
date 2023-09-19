"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member,import-error
from abc import abstractmethod
from typing import List, Tuple, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule

from .network import BaseNetwork
from ..callbacks.base import BaseCallback
from ..callbacks.logger.tensorboard_logger import SingletonWriter


class PipeBlock:
    """
    A training/inference block in pipeline.
    """
    def __init__(self, dataloader: DataLoader, execution_flow: List[BaseNetwork],
        optimizer: Optional[Optimizer] = None, iter_callbacks: Optional[List] = None,
        saving_dir: Optional[str] = None, fabric: Optional[Fabric] = None):
        self._dataloader = dataloader
        self._execution_flow = execution_flow
        self._optimizer = optimizer
        self._saving_dir = saving_dir
        self._fabric: Optional[Fabric] = fabric

        self._iter_callbacks: Optional[List[Tuple[BaseCallback, int]]] = iter_callbacks

    @staticmethod
    def run_target(targets: list[BaseNetwork], data_in: dict[str, torch.Tensor])\
        -> dict:
        """
        Run target with input/output parsing.
        """
        io_dict: dict[str, torch.Tensor | dict] = dict(data_in)
        for target in targets:
            inv_out_dict = {v: k for k, v in target.out_descs.items()}
            in_dict = dict()
            for d_key, d_value in io_dict.items():
                if d_key in target.in_descs:
                    in_dict[target.in_descs[d_key]] = d_value
            target_out: dict = target(in_dict)
            out_dict = dict()
            for d_key, d_value in target_out.items():
                if d_key in inv_out_dict:
                    out_dict[inv_out_dict[d_key]] = d_value
                else:
                    out_dict[d_key] = d_value
            for d_key, d_value in out_dict.items():
                if d_key == 'losses' or d_key == 'metrics':
                    io_dict.setdefault(d_key, {})
                    io_dict[d_key] = dict(io_dict[d_key], **d_value)  # type: ignore
                    continue
                assert d_key not in io_dict, f'Key {d_key} already exists in io_dict.'
                io_dict[d_key] = d_value
        return io_dict

    @abstractmethod
    def run_iter(self, data_in):
        """
        Run one iteration.
        """
        raise NotImplementedError

    @abstractmethod
    def run_epoch(self, epoch_idx):
        """
        Run one batch.
        """
        raise NotImplementedError

    def _run_callbacks_before_epoch(self):
        if self._iter_callbacks is None:
            return
        for callback, _ in self._iter_callbacks:
            callback.reset()

    def _run_callbacks(self, iter_idx, iter_data):
        if self._iter_callbacks is None:
            return
        for callback, callback_freq in self._iter_callbacks:
            if iter_idx % callback_freq != 0:
                continue
            self.run_target([callback], iter_data)

    def _run_callbacks_after_epoch(self):
        if self._iter_callbacks is None:
            return
        for callback, _ in self._iter_callbacks:
            callback.close()


class TrainPipeBlock(PipeBlock):
    """
    A training block in pipeline.
    """
    def __init__(self, dataloader: DataLoader, execution_flow: list[BaseNetwork],
        optimizer: Optimizer, iter_callbacks: Optional[List] = None, acc_stride: int = 1,
        saving_dir: Optional[str] = None, fabric=None):
        super().__init__(dataloader, execution_flow, optimizer, iter_callbacks,\
            saving_dir, fabric)
        self.log_writer = SingletonWriter()
        self._acc_stride = acc_stride
        self._acc_iter = 0

    def run_iter(self, data_in: dict[str, torch.Tensor]):
        assert self._optimizer is not None
        self._acc_iter += 1
        self._acc_iter %= self._acc_stride

        if self._fabric is None:
            data_out = self.run_target(self._execution_flow, data_in)
            losses = list(data_out['losses'].values())
            loss_sum = torch.sum(torch.stack(losses))
            loss_sum.backward()
            self._optimizer.step()
            return data_out

        fabric_modules = [_m for _m in self._execution_flow if isinstance(_m, _FabricModule)]
        assert len(fabric_modules) == 1, 'Only one fabric module is allowed.'
        with self._fabric.no_backward_sync(fabric_modules[0], enabled=self._acc_iter != 0):
            data_out = self.run_target(self._execution_flow, data_in)
            losses = list(data_out['losses'].values())
            with self._fabric.autocast():
                loss_sum = torch.sum(torch.stack(losses))
            self._fabric.backward(loss_sum)

        if self._acc_iter == 0:
            self._optimizer.step()
            self._optimizer.zero_grad()

        return data_out

    def run_epoch(self, epoch_idx):
        for _flow in self._execution_flow:
            _flow.train()
        self._run_callbacks_before_epoch()
        for iter_idx, iter_data in enumerate(self._dataloader):
            out_data = self.run_iter(iter_data)
            self._run_callbacks(iter_idx, {**iter_data, **out_data})
            self.log_writer.update_iter()
        self._run_callbacks_after_epoch()


class TestPipeBlock(PipeBlock):
    """
    A test block in pipeline
    """
    def __init__(self, dataloader: DataLoader, execution_flow: List[BaseNetwork],
        iter_callbacks: Optional[list] = None, saving_dir: Optional[str] = None,
        fabric: Optional[Fabric]=None):
        super().__init__(dataloader, execution_flow, None, iter_callbacks, saving_dir, fabric)

        # self.log_writer = SingletonWriter()

    def run_iter(self, data_in):
        data_out = self.run_target(self._execution_flow, data_in)
        return data_out

    def run_epoch(self, epoch_idx):
        for _flow in self._execution_flow:
            _flow.eval()
        with torch.no_grad():
            self._run_callbacks_before_epoch()
            for iter_idx, iter_data in enumerate(self._dataloader):
                out_data = self.run_iter(iter_data)
                self. _run_callbacks(iter_idx, {**iter_data, **out_data})
                # self.log_writer.update_iter()
            self._run_callbacks_after_epoch()


ValidatePipeBlock = TestPipeBlock
