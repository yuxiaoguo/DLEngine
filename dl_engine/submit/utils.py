"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=logging-fstring-interpolation
import os


class ExpPathManager:
    """
    Manage the paths of the experiment.
    """
    def __init__(self, exp_name, exp_ctg, milestone, create_folder=False, exp_root=None) -> None:
        self._exp_name = exp_name
        self._exp_ctg = exp_ctg
        self._milestone = milestone
        self._create_folder = create_folder

        if exp_root is None:
            assert os.environ['MERCURY_EXP_ROOT'] is not None, \
                'Please set the environment variable MERCURY_EXP_ROOT.'
            self._exp_root = os.environ['MERCURY_EXP_ROOT']
        else:
            self._exp_root = exp_root

        self._exp_dir = '/'.join((self._exp_root, self._exp_ctg, self._milestone, self._exp_name))
        self._exp_cfg_dir = '/'.join((self._exp_dir, 'configs'))
        self._exp_log_dir = '/'.join((self._exp_dir, 'logs'))
        self._exp_ckpt_dir = '/'.join((self._exp_dir, 'ckpts'))
        self._exp_prof_dir = '/'.join((self._exp_dir, 'profiles'))

        if self._create_folder:
            os.makedirs(self._exp_dir, exist_ok=True)
            os.makedirs(self._exp_cfg_dir, exist_ok=True)
            os.makedirs(self._exp_log_dir, exist_ok=True)
            os.makedirs(self._exp_ckpt_dir, exist_ok=True)
            os.makedirs(self._exp_prof_dir, exist_ok=True)

    exp_root = property(lambda self: self._exp_root)
    exp_dir = property(lambda self: self._exp_dir)
    exp_cfg_dir = property(lambda self: self._exp_cfg_dir)
    exp_log_dir = property(lambda self: self._exp_log_dir)
    exp_ckpt_dir = property(lambda self: self._exp_ckpt_dir)
    exp_prof_dir = property(lambda self: self._exp_prof_dir)
