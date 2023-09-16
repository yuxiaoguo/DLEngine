"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
import re
import os
import enum
from typing import Optional

import yaml

from ..utils import Singleton
from .logger import Logger


class EnvAnnoType(enum.Enum):
    """
    Annotation type for env

    - ENV: the environment variable name
    - MODULE: the module name which will be imported
    - CFG: the config file which will be expanded as a dict
    """
    ENV = 1
    MODULE = 2
    CFG = 3


class EnvAnno:
    """
    Annotation for env
    """
    def __init__(self, name, anno_type: EnvAnnoType) -> None:
        self.name = name
        self.anno_type = anno_type

    @property
    def anno_str(self):
        """
        Return annotation string
        """
        return f'{self.anno_type.name}:{self.name}'


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

        self.envs.update(os.environ.copy())

        local_envs = local_cfg.pop('envs', dict())
        assert isinstance(local_envs, (dict, list)), \
            f'envs should be dict or list, but got {type(local_envs)}'
        if isinstance(local_envs, list):
            local_envs = {_k: _v for _k, _v in local_envs}
        for k_env, v_env in local_envs.items():
            self.envs[k_env] = self._parse_args(v_env)

        for key, value in local_cfg.items():
            if hasattr(self, key):
                setattr(self, key, self._parse_args(value))
        return self

    def load_from_yaml(self, config_file: str):
        """
        Load config from yaml file
        """
        with open(config_file, 'r', encoding='utf-8') as yaml_file:
            config_dict: dict = yaml.safe_load(yaml_file)
        return self.load_from_dict(config_dict)

    def _parse_env_cfg_args(self, arg_str: str, match_str: EnvAnno):
        if match_str.anno_type == EnvAnnoType.ENV:
            replace_str = self.envs[match_str.name]
            if isinstance(replace_str, str):
                arg_str = arg_str.replace(f'${match_str.anno_str}$', str(replace_str))
                return self._parse_args(arg_str)
            elif isinstance(replace_str, (dict, list)):
                return replace_str
            elif isinstance(replace_str, (int, float)):
                return replace_str
            else:
                raise NotImplementedError(f'Unknown type of env: {type(replace_str)}')
        elif match_str.anno_type == EnvAnnoType.CFG:
            with open(match_str.name, 'r', encoding='utf-8') as cfg_stream:
                cfg_dict = yaml.load(cfg_stream, Loader=yaml.FullLoader)
            return self._parse_args(cfg_dict)
        else:
            raise ValueError(f'Unknown type source: {match_str}')

    def _acquire_anno_type(self, arg_str: str) -> list[EnvAnno]:
        type_var_pairs: list[str] = re.findall(r'\$([A-Za-z0-9\:\_\/\.]*)\$', arg_str)
        if not type_var_pairs:
            return list()
        env_annos = list()
        for tv_pair in type_var_pairs:
            tv_part = tv_pair.split(':')
            assert len(tv_part) <= 2, f'Invalid type var pair: {tv_pair}'
            anno_name = tv_part[-1]
            anno_type = 'MODULE' if len(tv_part) == 1 else tv_part[0]
            assert anno_type in ['ENV', 'MODULE', 'CFG'], f'Invalid type source: {anno_type}'
            env_annos.append(EnvAnno(anno_name, EnvAnnoType[anno_type]))
        return env_annos

    def _parse_args(self, arg_str: str | list | tuple | dict, extra_kwargs: Optional[dict] = None):
        """
        Parse args from string.
        """
        if extra_kwargs is None:
            extra_kwargs = dict()
        if isinstance(arg_str, str):
            arg_value = arg_str
            env_annos = self._acquire_anno_type(arg_value)
            for env_anno in env_annos:
                if env_anno.anno_type == EnvAnnoType.MODULE:
                    Logger().warning_zero_rank(\
                        f'Using parse_module to importing module - {env_anno.name}')
                    continue
                assert isinstance(arg_value, str)
                arg_value = self._parse_env_cfg_args(arg_value, env_anno)
            return arg_value
        elif isinstance(arg_str, list):
            return [self._parse_args(_i, extra_kwargs) for _i in arg_str]
        elif isinstance(arg_str, tuple):
            return tuple([self._parse_args(_i, extra_kwargs) for _i in arg_str])
        elif isinstance(arg_str, dict):
            dict_str = dict()
            for _ik, _iv in arg_str.items():
                print(_ik, _iv)
                dict_str[_ik] = self._parse_args(_iv, extra_kwargs)
            return dict_str
        elif isinstance(arg_str, (int, float)):
            return arg_str
        raise NotImplementedError(f'Unknown type of arg_str: {type(arg_str)}')

    def parse_item(self, arg_str, extra_kwargs: dict | None = None):
        """
        Parse args from string as a module.
        """
        if extra_kwargs is None:
            extra_kwargs = dict()

        if isinstance(arg_str, str):
            env_annos = self._acquire_anno_type(arg_str)
            if not env_annos:
                return arg_str
            assert len(env_annos) == 1, 'Only one anno is allowed for a single module'
            env_anno = env_annos[0]
            assert env_anno.anno_type == EnvAnnoType.MODULE, \
                f'Only MODULE is allowed for parse_module, but got {env_anno.anno_type}'
            return extra_kwargs[env_anno.name]
        elif isinstance(arg_str, list):
            return [self.parse_item(_i, extra_kwargs) for _i in arg_str]
        elif isinstance(arg_str, tuple):
            return tuple([self.parse_item(_i, extra_kwargs) for _i in arg_str])
        elif isinstance(arg_str, dict):
            dict_str = dict()
            for _ik, _iv in arg_str.items():
                dict_str[_ik] = self.parse_item(_iv, extra_kwargs)
            return dict_str
        else:
            return arg_str


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

        self.set_epoch()

        self.load_from_dict(self._raw_config)
        return self

    def set_epoch(self, epoch: Optional[int | str] = None):
        """
        Set epoch
        """
        if epoch is None:
            if os.path.isdir(self.ckpt_dir):
                if os.path.exists(os.path.join(self.ckpt_dir, 'final.pt')):
                    epoch = 'final'
                else:
                    epoch = 0
            else:
                epoch = os.path.basename(self.ckpt_dir).split('.')[0]
        self.envs['EPOCH'] = epoch
