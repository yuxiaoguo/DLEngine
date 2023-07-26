"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
from torch import optim
from torch.utils import data

from dl_engine.core.logger import Logger
from dl_engine.utils import Singleton


class RegisterModule(metaclass=Singleton):
    """
    Register a module globally and could be later used by pipeline.
    """
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

global_register = RegisterModule()

network_register = NetworkRegister()
ref_network_register = RefNetworkRegister()

dataset_register = DatasetRegister()
dataset_register.register(data.ChainDataset)

sampler_register = SamplerRegister()
functional_register = FunctionalComponmentRegister()
optimizer_register = OptimizerRegister()
dataloader_register = DataloaderRegister()
