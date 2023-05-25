"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
import os
import zipfile
from typing import Optional

from dl_engine.core.register import functional_register


@functional_register.register
class ZipRecorder:
    """
    ZipRecorder is a recorder that records the data into zip file.
    """
    def __init__(self, out_dir: str, record_alias: str) -> None:
        self._zip_file: Optional[zipfile.ZipFile] = None
        self._out_dir = out_dir
        self._record_alias = record_alias
        self._dump_keys = []

        self._iter = 0

    def start(self, epoch):
        """
        Start recording the data. Zip file will be created if not exists.
        """
        assert self._zip_file is None, "ZipRecorder is already recording."
        zip_file_name = f"{self._record_alias}_{epoch:05d}.zip"
        zip_path = os.path.join(self._out_dir, zip_file_name)
        self._zip_file = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED)
        self._iter = 0

    def tick(self, io_proto):
        """
        Tick the recorder. The data will be dumped to zip file.
        """
        self._iter += 1

    def stop(self):
        """
        Stop recording the data. Zip file will be closed.
        """
        if self._zip_file is not None:
            self._zip_file.close()
        self._zip_file = None
