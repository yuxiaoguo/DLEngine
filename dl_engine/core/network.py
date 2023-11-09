"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=no-member
import enum
import os
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
        self.statistics: dict[str, torch.Tensor] = dict()

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
        str_dict['statistics'] = self.statistics
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
    IO_INTERFACE = BaseIO

    def __init__(self, in_descs=None, out_descs=None, weights_path=None,
        io_type=BaseIO, name=None, trainable=None) -> None:
        super().__init__()
        if self.__class__.IO_INTERFACE != BaseIO:
            assert io_type == BaseIO or self.__class__.IO_INTERFACE == io_type,\
                'IO type must be BaseIO when IO_INTERFACE is not BaseIO'
            io_type = self.__class__.IO_INTERFACE
        self._io_type = io_type
        self._weights_path = weights_path
        self._name = name
        if isinstance(trainable, str):
            trainable = TrainState[trainable.upper()]
        elif isinstance(trainable, TrainState):
            pass
        elif trainable is None:
            trainable = TrainState.TRAINABLE
        self._trainable: TrainState = trainable

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

        if self._weights_path is not None and os.path.exists(self._weights_path):
            if self._name is None:
                raise ValueError('Name must be specified when loading weights')
            Logger().info(f'{self.__class__} Loading weights from {self._weights_path}')
            weights_dict = torch.load(self._weights_path, map_location='cpu')
            near_queries = [_t for _t in weights_dict.keys() \
                if _t.find(self._name) != -1 or self._name.find(_t) != -1]
            if self._name in weights_dict:
                target_name = self._name
                Logger().info(f'{self.__class__} Loading weights from {self._weights_path}')
            elif 'model_state_dict' in weights_dict:
                target_name = 'model_state_dict'
                Logger().info(f'{self.__class__} Loading weights from legacy dict key')
            elif len(near_queries) == 1:
                target_name = near_queries[0]
                Logger().info(f'{self.__class__} Loading weights from near query {target_name}')
            else:
                raise ValueError(f'Weights for {self._name} not found in {self._weights_path}')
            self.load_state_dict(weights_dict[target_name])
        else:
            Logger().info(f'{self.__class__} No weights loaded')

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

    def parse_input(self, inputs: dict[str, torch.Tensor]):
        """
        Parse the input tensors
        """
        parsed_dict = dict()
        for in_name, in_tensor in inputs.items():
            if in_name not in self._in_descs.keys():
                continue
            parsed_dict[self._in_descs[in_name]] = in_tensor
        return parsed_dict

    def parse_output(self, outputs: dict[str, torch.Tensor]):
        """
        Parse the output tensors
        """
        parsed_dict = dict()
        for out_name, out_tensor in outputs.items():
            if out_name not in self._out_descs.values():
                continue
            raw_name, = [_k for _k, _v in self._out_descs.items() if _v == out_name]
            parsed_dict[raw_name] = out_tensor
        return parsed_dict

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


@network_register.register
class SequentialNetwork(BaseNetwork):
    """
    Sequential execution of networks with auto I/O parsing
    """
    def __init__(self,
                 streams: list[BaseNetwork],
                 with_mediates: bool = False,
                 in_descs=None,
                 out_descs=None,
                 weights_path=None,
                 io_type=BaseIO,
                 name=None,
                 trainable=None) -> None:
        super().__init__(in_descs, out_descs, weights_path, io_type, name, trainable)
        self._streams = streams
        self._with_mediates = with_mediates

        self._io_type = type('SequentialIO', (BaseIO,), dict(
            **{_k: None for _k in self._in_descs.values()},
            **{_k: None for _k in self._out_descs.values()},
        ))

    def _run(self, io_proto: BaseIO, **extra_kwargs) -> BaseIO:
        io_dict = io_proto.convert_to_str_dict(self._in_descs)
        for stream in self._streams:
            out_dict = \
                stream.parse_output(stream(stream.parse_input(io_dict), **extra_kwargs))
            io_dict.update(out_dict)
        for out_name in self._out_descs.values():
            if out_name in io_dict:
                setattr(io_proto, out_name, io_dict[out_name])
