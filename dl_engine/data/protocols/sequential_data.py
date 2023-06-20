"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
from dataclasses import dataclass, field


@dataclass
class MetaFileDescV0:
    """Meta File Protocol Version 0

    Attributes:
        meta_name: str, the name of meta file without extension
    """
    meta_file: str = ''
    num_samples: int = 0
    global_offset: int = 0


@dataclass
class MetaSeqFileDescV0(MetaFileDescV0):
    """Meta Sequential File Protocol Version 0

    Attributes:
        meta_name: str, the name of meta file without extension
    """
    num_nonseq_samples: int = 0
    global_nonseq_offset: int = 0
    local_nonseq_offset: list[int] = field(default_factory=list)


@dataclass
class SequentialDataDescV0:
    """Sequential Data Protocol Version 0

    Attributes:
        package_type: str, 'zip', 'pkl' or 'tar'
    """
    package_type: str = 'zip'
    meta_files: list[MetaSeqFileDescV0] = field(default_factory=list)
    total_samples: int = 0
    total_nonseq_samples: int = 0
    props: dict = field(default_factory=dict)

    def __post_init__(self):
        self.meta_files = [\
            MetaSeqFileDescV0(**_meta_file) for _meta_file in self.meta_files]  # type: ignore
