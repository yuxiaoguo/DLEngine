"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
from dataclasses import dataclass, field


@dataclass
class VersionDesc:
    """
    Version controlled description.
    """
    proto_verison: str = '0.0.0'
    proto_name: str = ''

    def __post_init__(self):
        if self.proto_name == '':
            self.proto_name = self.__class__.__name__
        assert self.proto_name == self.__class__.__name__, \
            f'proto_name: {self.proto_name}, class_name: {self.__class__.__name__}'


@dataclass
class MetaFileDescV0(VersionDesc):
    """Meta File Protocol Version 0

    Attributes:
        meta_name: str, the name of meta file without extension
    """
    meta_file: str = ''
    num_samples: int = 0
    global_offset: int = 0


@dataclass
class MetaDatasetDescV0(VersionDesc):
    """Meta Dataset Protocol Version 0

    Attributes:
        meta_name: str, the name of meta file without extension
    """
    package_type: str = 'zip'
    meta_files: list[MetaFileDescV0] = field(default_factory=list)
    total_samples: int = 0
    props: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.meta_files = [\
            globals()[_meta_file['proto_name']](**_meta_file) \
                for _meta_file in self.meta_files]  # type: ignore


@dataclass
class MetaSeqFileDescV0(MetaFileDescV0):
    """Meta Sequential File Protocol Version 0

    Attributes:
        meta_name: str, the name of meta file without extension
    """
    num_nonseq_samples: int = 0
    global_nonseq_offset: int = 0
    local_nonseq_offset: list[int] = field(default_factory=list)
    key_attrs: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class SequentialDataDescV0(MetaDatasetDescV0):
    """Sequential Data Protocol Version 0

    Attributes:
        package_type: str, 'zip', 'pkl' or 'tar'
    """
    total_nonseq_samples: int = 0
