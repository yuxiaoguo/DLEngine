"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
import re
import os
import yaml
from typing import Optional

from ..utils import Singleton


class PipelineConfig(metaclass=Singleton):
    """
    Config used for pipeline training
    """
    def __init__(self) -> None:
        self._raw_config = dict()

        self.config_path = str()
        self.log_dir = str()
        self.ckpt_dir = str()
        self.prof_dir = str()

        self.save_ckpt = 0
        self.precision = str()

        self.envs = dict()

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

        self.save_ckpt = self._raw_config.get('save_ckpt', 0)
        self.precision = self._raw_config.get('precision', '16-mixed')

        self.set_epoch()
        self.envs = {**os.environ, **self.envs}
        for c_key, c_value in self._raw_config.get('envs', dict()).items():
            self.envs[c_key] = self.parse_args(c_value)

        self.datasets = self._raw_config.get('datasets', dict())
        self.samplers = self._raw_config.get('samplers', dict())
        self.dataloaders = self._raw_config.get('dataloaders', dict())
        self.networks = self._raw_config.get('networks', dict())
        self.optimizers = self._raw_config.get('optimizers', dict())
        self.functional_components = self._raw_config.get('functional_components', dict())

        self.pipelines = self._raw_config.get('pipelines', dict())
        return self

    def set_epoch(self, epoch: Optional[int | str] = None):
        """
        Set epoch
        """
        if epoch is None:
            if os.path.isdir(self.ckpt_dir):
                assert os.path.exists(os.path.join(self.ckpt_dir, 'final.pt'))
                epoch = 'final'
            else:
                epoch = os.path.basename(self.ckpt_dir).split('.')[0]
        self.envs['EPOCH'] = epoch

    def parse_args(self, arg_str: str | list | tuple | dict, extra_kwargs: Optional[dict] = None):
        """
        Parse args from string.
        """
        if extra_kwargs is None:
            extra_kwargs = dict()
        if isinstance(arg_str, str):
            type_var_pairs = re.findall(r'\$([A-Za-z0-9\:\_]*)\$', arg_str)
            if not type_var_pairs:
                return arg_str
            arg_value = arg_str
            for tv_pair in type_var_pairs:
                tv_part = tv_pair.split(':')
                if len(tv_part) == 2:
                    type_src, type_name = tv_part
                elif len(tv_part) == 1:
                    type_src = 'MODULE'
                    type_name, = tv_part
                else:
                    raise NotImplementedError(f'Unknown type vars: {tv_pair}')
                if type_src == 'ENV':
                    env_str = self.envs.get(type_name, None)
                    if env_str is None:
                        raise ValueError(f'Unknown env var: {type_name}')
                    arg_value = arg_value.replace(f'${tv_pair}$', env_str)
                elif type_src == 'MODULE':
                    arg_value = extra_kwargs[type_name]
                else:
                    raise NotImplementedError(f'Unknown type source: {type_src}')
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
