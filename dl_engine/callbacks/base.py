"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
from dl_engine.core.network import BaseIO, BaseNetwork


class BaseCallback(BaseNetwork):
    """
    Base class for callbacks.
    """
    def _run(self, io_proto: BaseIO, **extra_kwargs) -> BaseIO:
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset the callback, e.g. at the beginning of each epoch for an iteration callback.
        """

    def close(self) -> None:
        """
        Close the callback, e.g. at the end of each epoch for an iteration callback.
        """
