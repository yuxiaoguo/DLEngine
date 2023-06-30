"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
import re
import os
from typing import Optional

import yaml

from ..utils import Singleton


class BaseConfig(metaclass=Singleton):
    """
    Base config class
    """
    def __init__(self) -> None:
        self.envs = dict()

    def load_from_dict(self, config_dict: dict):
        """
        Load config from dict
        """
        local_cfg = config_dict.copy()

        self.envs = os.environ.copy()

        envs = local_cfg.pop('envs', dict())
        assert isinstance(envs, (dict, list)), f'envs should be dict or list, but got {type(envs)}'
        if isinstance(envs, list):
            local_envs = {_k: self.parse_args(_v) for _k, _v in envs}
        else:
            local_envs = {_k: self.parse_args(_v) for _k, _v in envs.items()}
        self.envs.update(local_envs)

        for key, value in local_cfg.items():
            if hasattr(self, key):
                setattr(self, key, self.parse_args(value))
        return self

    def _parse_env_cfg_args(self, arg_str: str, match_str: str) -> str:
        # if type_src == 'ENV':
        #     env_str = self.envs.get(type_name, None)
        #     if env_str is None:
        #         raise ValueError(f'Unknown env var: {type_name}')
        #     arg_value = arg_value.replace(f'${tv_pair}$', env_str)
        if match_str.startswith('ENV'):
            raise NotImplementedError('ENV not implemented yet')
        elif match_str.startswith('CFG'):
            cfg_path = match_str.split(':')[1]
            with open(cfg_path, 'r', encoding='utf-8') as cfg_stream:
                cfg_dict = yaml.load(cfg_stream, Loader=yaml.FullLoader)
        else:
            raise ValueError(f'Unknown type source: {match_str}')
        return ''

    def parse_args(self, arg_str: str | list | tuple | dict, extra_kwargs: Optional[dict] = None):
        """
        Parse args from string.
        """
        if extra_kwargs is None:
            extra_kwargs = dict()
        if isinstance(arg_str, str):
            type_var_pairs: list[str] = re.findall(r'\$([A-Za-z0-9\:\_\/\.]*)\$', arg_str)
            if not type_var_pairs:
                return arg_str
            arg_value = arg_str
            for tv_pair in type_var_pairs:
                tv_part = tv_pair.split(':')
                assert len(tv_part) <= 2, f'Invalid type var pair: {tv_pair}'
                type_name = tv_part[-1]
                type_src = 'MODULE' if len(tv_part) == 1 else tv_part[0]
                assert type_src in ['ENV', 'MODULE', 'CFG'], f'Invalid type source: {type_src}'
                if type_src == 'MODULE':
                    arg_value = extra_kwargs[type_name]
                    continue
                arg_value = self._parse_env_cfg_args(arg_value, tv_pair)
            return arg_value
        elif isinstance(arg_str, list):
            return [self.parse_args(_i, extra_kwargs) for _i in arg_str]
        elif isinstance(arg_str, tuple):
            return tuple([self.parse_args(_i, extra_kwargs) for _i in arg_str])
        elif isinstance(arg_str, dict):
            dict_str = dict()
            for _ik, _iv in arg_str.items():
                print(_ik, _iv)
                dict_str[_ik] = self.parse_args(_iv, extra_kwargs)
            return dict_str
            # return {_ik: self.parse_args(_iv) for _ik, _iv in arg_str.items()}
        elif isinstance(arg_str, (int, float)):
            return arg_str
        raise NotImplementedError(f'Unknown type of arg_str: {type(arg_str)}')



class PipelineConfig(BaseConfig):
    """
    Config used for pipeline training
    """
    def __init__(self) -> None:
        super().__init__()

        self._raw_config = dict()

        self.config_path = str()
        self.log_dir = str()
        self.ckpt_dir = str()
        self.prof_dir = str()

        self.save_ckpt = 0
        self.precision = str()

        self.datasets = dict()
        self.samplers = dict()
        self.dataloaders = dict()
        self.networks = dict()
        self.optimizers = dict()
        self.functional_components = dict()

        self.pipelines = dict()

    def from_yaml(self, config_path, log_dir, ckpt_dir, prof_dir):
        """
        Load config from yaml file
        """
        self.config_path = config_path
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.prof_dir = prof_dir

        with open(config_path, 'r', encoding='utf-8') as yaml_file:
            self._raw_config: dict = yaml.safe_load(yaml_file)

        self.load_from_dict(self._raw_config)
        return self

    def set_epoch(self, epoch: Optional[int | str] = None):
        """
        Set epoch
        """
        # TODO: make me unify with ckpt loader
        if epoch is None:
            if os.path.isdir(self.ckpt_dir):
                if os.path.exists(os.path.join(self.ckpt_dir, 'final.pt')):
                    epoch = 'final'
                else:
                    epoch = 0
            else:
                epoch = os.path.basename(self.ckpt_dir).split('.')[0]
        self.envs['EPOCH'] = epoch
