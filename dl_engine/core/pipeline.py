"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=unused-import,logging-fstring-interpolation
import os
import shutil
import inspect
from typing import Callable, Optional, Dict, List

import torch
import wandb
import lightning as L
from lightning.fabric.strategies import DDPStrategy
from torch import distributed as dist

from . import pipe_block
from .config import PipelineConfig
from .network import TrainState, BaseNetwork
from ..callbacks.logger.tensorboard_logger import SingletonWriter

from .logger import Logger

from .register import (
    RegisterModule,
    network_register,
    dataset_register,
    sampler_register,
    optimizer_register,
    dataloader_register,
    functional_register
)


class Pipeline:
    """
    Pipeline class for running training/inference pipeline.
    """
    def __init__(self, config_path: str, log_dir, ckpt_dir, prof_dir, devices='auto', num_nodes=1,
        wandb_key=''):
        self._config = PipelineConfig().from_yaml(config_path, log_dir, ckpt_dir, prof_dir)
        strategy = DDPStrategy(find_unused_parameters=True)
        self._fabric = L.Fabric(strategy=strategy,
            precision=self._config.precision, devices=devices, num_nodes=num_nodes)  # type: ignore
        self._fabric.launch()

        self._log_dir = log_dir
        self._ckpt_dir = ckpt_dir
        self._prof_dir = prof_dir

        self._start_epoch = 0

        self._datasets = dict()
        self._samplers = dict()
        self._networks: dict[str, torch.nn.Module] = dict()
        self._optimizers = dict()
        self._dataloaders = dict()
        self._func_components = dict()
        self._blocks: List[pipe_block.PipeBlock] = list()

        self._build()
        self._load_ckpt()

        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        if self._rank == 0:
            if wandb_key != '' and self._rank == 0:
                wandb.login(key=wandb_key)
                config_seps = config_path.replace('\\', '/').split('/')
                task, ms, job_id, _, _ = config_seps[-5:]
                project = f'{task}-{ms}'
                wandb.init(project=project, name=job_id, sync_tensorboard=True, dir=log_dir)
                os.makedirs(log_dir, exist_ok=True)
            # SingletonWriter().initialize(log_dir=f'{log_dir}/tensorboard')
        SingletonWriter().initialize(log_dir=log_dir)

    def args_matching(self, func: Callable, kwargs_dict: Dict):
        """
        Matching the args of function with args_dict.
        """
        matched_dict = dict()
        func_kwargs = inspect.getfullargspec(func).args
        func_kwargs.remove('self')

        if 'config' not in func_kwargs and 'config' in kwargs_dict:
            del kwargs_dict['config']

        Logger().info_zero_rank(f'Start to parse kwargs of {func.__name__}')
        for _k, _v in kwargs_dict.items():
            if _k in func_kwargs:
                Logger().info_zero_rank(f"-- Matched arg: {_k}={_v}")
                matched_dict[_k] = self._config.parse_item(_v, self._query_target())
            else:
                raise ValueError(f'Unmatched arg: {_k}={_v}')

        unmatched_args = [_k for _k in func_kwargs if _k not in kwargs_dict.keys()]
        Logger().warning_zero_rank(f'-- Unmatched args: {unmatched_args}')
        return matched_dict

    def _query_target(self):
        """
        Query target.
        """
        targets = {
            **self._dataloaders,
            **self._datasets,
            **self._samplers,
            **self._networks,
            **self._optimizers,
            **self._func_components
        }
        return targets

    def _build(self):
        """
        Build pipeline.
        """
        def _build_item(_register: RegisterModule, _config: dict, _target,
            func: Optional[Callable] = None, with_name=False):
            for _name, _desc in _config.items():
                _type = _register.get(_desc['type'])
                if _type is None:
                    raise ValueError(f'Cannot find {_desc["type"]} in {_register}')
                _kwargs = self.args_matching(_type, _desc['kwargs'])
                if with_name:
                    _kwargs['name'] = _name
                _instance = _type(**_kwargs)
                if func is not None:
                    _instance = func(_instance)
                _target[_name] = _instance

        # Build dataset/dataloader components
        _build_item(dataset_register, self._config.datasets, self._datasets)
        _build_item(sampler_register, self._config.samplers, self._samplers)
        _build_item(dataloader_register, self._config.dataloaders, self._dataloaders,\
            self._fabric.setup_dataloaders)

        # Build network components
        def _ondemand_setup_module(_module: BaseNetwork):
            if _module.trainable == TrainState.TRAINABLE:
                _module_warped = self._fabric.setup_module(_module)
            else:
                _module_warped = _module
            return _module_warped
        _build_item(network_register, self._config.networks, self._networks, \
            _ondemand_setup_module, True)

        _build_item(functional_register, self._config.functional_components, \
            self._func_components)

        # Build optimizers
        for _o_name, _o_desc in self._config.optimizers.items():
            o_type = optimizer_register.get(_o_desc['type'])
            o_kwargs = self.args_matching(o_type, _o_desc['kwargs'])
            o_kwargs['params'] = list(*[_o_k.parameters() for _o_k in o_kwargs['params']])
            self._optimizers[_o_name] = o_type(**o_kwargs)
        for key, value in self._optimizers.items():
            self._optimizers[key] = self._fabric.setup_optimizers(value)

        # Build pipelines and optimizers
        blocks = self._config.pipelines.get('blocks', [])
        for b_desc in blocks:
            block_type = getattr(pipe_block, f'{b_desc["phase"]}PipeBlock')
            b_kwargs = self.args_matching(block_type, b_desc['kwargs'])
            b_kwargs = {
                'saving_dir': self._ckpt_dir, **b_kwargs,
                'fabric': self._fabric
            }
            self._blocks.append(block_type(**b_kwargs))

    def _load_ckpt(self):
        """
        Load checkpoint.
        """
        if self._config.use_epoch != '':
            ckpt_dir = os.path.join(self._ckpt_dir, f'epoch_{self._config.use_epoch}.pt')
        else:
            ckpt_dir = self._ckpt_dir
        Logger().info(f'Loading checkpoint from {ckpt_dir}')
        if os.path.isdir(ckpt_dir):
            existed_files = os.listdir(ckpt_dir)
            if 'epoch_state' in existed_files and 'latest.pt' in existed_files:
                with open(os.path.join(ckpt_dir, 'epoch_state'), encoding='utf-8') as e_fp:
                    self._start_epoch = int(e_fp.read())
                ckpt_path = os.path.join(ckpt_dir, 'latest.pt')
            else:
                Logger().info('No checkpoint found, start from scratch.')
                return
        elif os.path.exists(ckpt_dir) and ckpt_dir.endswith('.pt'):
            self._start_epoch = 0
            ckpt_path = ckpt_dir
        else:
            Logger().info('No checkpoint found, start from scratch.')
            return
        ckpt: Dict = torch.load(ckpt_path, map_location='cpu')
        for _k, _v in ckpt.items():
            if _k in self._networks:
                Logger().info(f'Loading network: {_k}')
                missing, unexcepted = self._networks[_k].load_state_dict(_v, False)
                Logger().info(f'-- Missing keys: {missing}')
                Logger().info(f'-- Unexpected keys: {unexcepted}')
            elif len([_n for _n in self._networks.keys() if _n.startswith(_k)]) > 0:
                matching_pairs = \
                    [(_n, _k) for _n, _ in self._networks.items() if _n.startswith(_k)]
                assert len(matching_pairs) == 1, f'Multiple matching pairs: {matching_pairs}'
                Logger().info(f'Loading network: {matching_pairs[0][0]} from {_k}')
                missing, unexcepted = \
                    self._networks[matching_pairs[0][0]].load_state_dict(_v, False)
            elif _k in self._optimizers:
                Logger().info(f'Loading optimizer: {_k}')
                self._optimizers[_k].load_state_dict(_v)
            else:
                Logger().info(f'Unknown key: {_k}')

    def _save_ckpt(self, epoch_idx: int):
        Logger().info(f'Saving checkpoint to {self._ckpt_dir}')
        os.makedirs(self._ckpt_dir, exist_ok=True)
        epoch_path = os.path.join(self._ckpt_dir, f'epoch_{epoch_idx:03d}.pt')
        saved_dict = dict()
        for _k, _v in self._networks.items():
            saved_dict[_k] = _v.state_dict()
        for _k, _v in self._optimizers.items():
            saved_dict[_k] = _v.state_dict()
        torch.save(saved_dict, epoch_path)

        shutil.copy(epoch_path, os.path.join(self._ckpt_dir, 'latest.pt'))
        with open(os.path.join(self._ckpt_dir, 'epoch_state'), 'w', encoding='utf-8') as e_fp:
            e_fp.write(str(epoch_idx))
        if epoch_idx == self._config.pipelines['num_times']:
            shutil.copy(epoch_path, os.path.join(self._ckpt_dir, 'final.pt'))

    def run(self):
        """
        Run pipeline.
        """
        for epoch in range(self._start_epoch, self._config.pipelines['num_times']):
            Logger().info_zero_rank(f'Epoch: {epoch + 1}')
            if epoch == 0 and self._config.save_ckpt and self._rank == 0:
                self._save_ckpt(0)
            for block in self._blocks:
                Logger().info_zero_rank(f'-- Block: {block.__class__.__name__}')
                block.run_epoch(epoch_idx=epoch)
            if self._config.save_ckpt and self._rank == 0:
                self._save_ckpt(epoch + 1)
