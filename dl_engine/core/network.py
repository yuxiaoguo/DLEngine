"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=no-member
import enum
from abc import abstractmethod

import torch
from torch.nn import Module
from torch import distributed as dist

from .logger import Logger
from .register import network_register


class BaseIO:
    """
    Base IO class defines the keys of input and output tensors. It provides
        a way to access the tensors by key of class member rather than string
        in the `BaseNetwork`. The class member could be defined in the subclass.
    
    Base IO class could help track all tensors declared in the subclass and
        be easily adapted to the output dictory. All important tensors in
        `BaseNetwork` should be defined in the subclass of `BaseIO` and assign/
        access the tensor via defined member.
    """
    def __init__(self):
        self.losses: dict[str, torch.Tensor] = dict()
        self.metrics: dict[str, torch.Tensor] = dict()

    def launch_from_str_dict(self, tensors: dict[str, torch.Tensor]):
        """
        Launch the IO tensors
        """
        for src_key, src_value in tensors.items():
            assert hasattr(self, src_key), f'Key {src_key} not found in {self.__class__.__name__}'
            setattr(self, src_key, src_value)
        return self

    def convert_to_str_dict(self, out_descs: dict[str, str]) -> dict[str, torch.Tensor]:
        """
        Convert the IO tensors to string dict
        """
        if out_descs is None:
            out_descs = dict()
        str_dict = dict()
        for dst_key in out_descs.values():
            assert hasattr(self, dst_key), f'Key {dst_key} not found in {self.__class__.__name__}'
            str_dict[dst_key] = getattr(self, dst_key)
        str_dict['losses'] = self.losses
        str_dict['metrics'] = self.metrics
        return str_dict
    

class TrainState(enum.Enum):
    """
    Train state enum class.
    """
    TRAINABLE = 0
    FIX_ALL = 1
    FIX_VARIABLE = 2


class BaseNetwork(Module):
    """
    Base network class.
    """
    def __init__(self, in_descs=None, out_descs=None, weights_path=None,
        io_type=BaseIO, name=None, trainable=None) -> None:
        super().__init__()
        self._io_type = io_type
        self._weights_path = weights_path
        self._name = name
        self._trainable = TrainState[trainable] if trainable is not None \
            else TrainState.TRAINABLE

        self._initialized = False
        self._rank = 0

        self._in_descs = in_descs if in_descs is not None else dict()
        self._out_descs = out_descs if out_descs is not None else dict()

    @property
    def in_descs(self):
        """
        Get the keys of input tensors
        """
        return self._in_descs

    @property
    def out_descs(self):
        """
        Get the keys of output tensors
        """
        return self._out_descs

    @property
    def io_type(self):
        """
        Get the type of IO class
        """
        return self._io_type

    def initialize(self, **extra_kwargs):
        """
        Initialize function.
        """
        del extra_kwargs
        if not self._initialized:
            self._initialized = True

        if self._weights_path is not None:
            if self._name is None:
                raise ValueError('Name must be specified when loading weights')
            Logger().info(f'{__class__} Loading weights from {self._weights_path}')
            weights_dict = torch.load(self._weights_path)
            if self._name in weights_dict:
                target_name = self._name
                Logger().info(f'{__class__} Loading weights from {self._weights_path}')
            elif 'model_state_dict' in weights_dict:
                target_name = 'model_state_dict'
                Logger().info(f'{__class__} Loading weights from legacy dict key')
            else:
                raise ValueError(f'Weights for {self._name} not found in {self._weights_path}')
            self.load_state_dict(weights_dict[target_name])

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
        self.to(torch.device(f'cuda:{self._rank}'))

        if self._trainable in [TrainState.FIX_ALL, TrainState.FIX_VARIABLE]:
            self.requires_grad_(False).eval()

    @property
    def trainable(self):
        """
        Get the trainable state
        """
        return self._trainable

    @abstractmethod
    def _run(self, io_proto: BaseIO, **extra_kwargs):
        """
        Run function.
        """
        raise NotImplementedError

    def forward(self, inputs: dict[str, torch.Tensor], **extra_kwargs):  # pylint: disable=redefined-builtin
        """
        Forward function.
        """
        if not self._initialized:
            self.initialize()

        if isinstance(inputs, dict):
            io_instance: BaseIO = self._io_type().launch_from_str_dict(inputs)
        elif isinstance(inputs, BaseIO):
            io_instance = inputs
        else:
            raise NotImplementedError(f'Input type {type(inputs)} not supported')
        self._run(io_instance, **extra_kwargs)
        if isinstance(inputs, dict):
            return io_instance.convert_to_str_dict(self._out_descs)
        else:
            return io_instance


@network_register.register
class RefNetwork(BaseNetwork):
    """
    Reference network class. Different from BaseNetwork, RefNetwork refers to
        the network defined in other modules. It re-defines IO during the forward
        of referred module.
    """
    def __init__(self, ref: BaseNetwork, in_descs=None, out_descs=None,
        name=None) -> None:
        super().__init__(in_descs, out_descs, None, ref.io_type, name)
        self._ref = ref
        self._name = name

    def _run(self, io_proto: BaseIO, **extra_kwargs) -> BaseIO:
        return self._ref(io_proto, **extra_kwargs)
