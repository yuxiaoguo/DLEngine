"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from typing import Dict

import yaml


class PipelineConfig:
    """
    Config used for pipeline training
    """
    def __init__(self) -> None:
        self._raw_config = dict()

        self.save_ckpt = 0
        self.precision = str()

        self.datasets = dict()
        self.samplers = dict()
        self.dataloaders = dict()
        self.networks = dict()
        self.optimizers = dict()
        self.functional_components = dict()

        self.pipelines = dict()

    def from_yaml(self, config_path):
        """
        Load config from yaml file
        """
        with open(config_path, 'r', encoding='utf-8') as yaml_file:
            self._raw_config: Dict = yaml.safe_load(yaml_file)

        self.save_ckpt = self._raw_config.get('save_ckpt', 0)
        self.precision = self._raw_config.get('precision', '16-mixed')

        self.datasets = self._raw_config.get('datasets', dict())
        self.samplers = self._raw_config.get('samplers', dict())
        self.dataloaders = self._raw_config.get('dataloaders', dict())
        self.networks = self._raw_config.get('networks', dict())
        self.optimizers = self._raw_config.get('optimizers', dict())
        self.functional_components = self._raw_config.get('functional_components', dict())

        self.pipelines = self._raw_config.get('pipelines', dict())
        return self
