"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from torch import optim
from torch.utils import data

from dl_engine.core.logger import Logger


class RegisterModule:
    """
    Register a module globally and could be later used by pipeline.
    """
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._registered_modules = {}
        self._register_existed()

    def _register_existed(self):
        """
        Register a module that has been registered.
        """

    def register(self, cls):
        """
        Register a module.
        """
        Logger().info(f'Registering {cls.__name__}.')
        self._registered_modules[cls.__name__] = cls
        return cls

    def get(self, name):
        """
        Get a registered module.
        """
        return self._registered_modules[name]


class OptimizerRegister(RegisterModule):
    """
    Register optimizer.
    """
    def _register_existed(self):
        methods = [_m for _m in dir(optim) if not _m.startswith('_')]
        for method in methods:
            self.register(getattr(optim, method))


class DataloaderRegister(RegisterModule):
    """
    Register optimizer.
    """
    def _register_existed(self):
        methods = [_m for _m in dir(data) if not _m.startswith('_')]
        for method in methods:
            self.register(getattr(data, method))


NetworkRegister = type('NetworkRegister', (RegisterModule,), {})
RefNetworkRegister = type('RefNetworkRegister', (RegisterModule,), {})
DatasetRegister = type('DatasetRegister', (RegisterModule,), {})
SamplerRegister = type('SamplerRegister', (RegisterModule,), {})
FunctionalComponmentRegister = type('VisualizerRegister', (RegisterModule,), {})

network_register = NetworkRegister()
ref_network_register = RefNetworkRegister()
dataset_register = DatasetRegister()
sampler_register = SamplerRegister()
functional_register = FunctionalComponmentRegister()
optimizer_register = OptimizerRegister()
dataloader_register = DataloaderRegister()
