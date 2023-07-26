"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
import os

from dl_engine.utils import Singleton


class GlobalEnvs(metaclass=Singleton):
    """
    GlobalEnvs is a singleton class that stores global environment variables.
    """
    def __init__(self) -> None:
        assert os.environ.get('EXP_ROOT') is not None, \
            'Please set the environment variable EXP_ROOT to the experiment directory.'
        assert os.environ.get('DATA_ROOT') is not None, \
            'Please set the environment variable DATA_ROOT to the data directory.'
        self._exp_root: str = os.environ.get('EXP_ROOT', '')
        self._data_root: str = os.environ.get('DATA_ROOT', '')

    @property
    def exp_root(self) -> str:
        """
        Returns the experiment directory.
        """
        return self._exp_root

    @property
    def data_root(self) -> str:
        """
        Returns the data directory.
        """
        return self._data_root


class TaskManager:
    """
    TaskManager is a class that managing a task. A task is defined by the final goal of
        the project. For example, the task of the project "NeRF" is to generate the 3D
        structure of a scene from a set of images. A task contains several milestones,
        which are defined by the intermediate goals of the project. For example, the
        milestones of the project "NeRF" are to "Milestone1 -> Train NeRF with hundreds
        of images", "Milestone2 -> Train NeRF with tens of images", etc.
    """


class MilestoneManager:
    """
    MilestoneManager is a class that managing a milestone. A milestone is defined by the
        intermediate goal of the project. For example, the milestones of the project
        "NeRF" are to "Milestone1 -> Train NeRF with hundreds of images", "Milestone2 ->
        Train NeRF with tens of images", etc. A milestone contains several experiments,
        each of which is designed to figure out a specific problem.
    """


class Experiment:
    """
    Experiment is a class that managing an experiment. Any experiment is belong to a task,
        defined by the final goal of 

    Attributes:
        profiles_path (str): The path to the profiles directory. The profiles directory
            is created automatically when the experiment is created and used for saving
            the contents generated during the training and testing process. Its absolute
            path is ``{EXP_ROOT}/{task}/{milestone}/{exp_id}/profiles``. And its subfolders'
            pattern is ``{profiles_path}/{epoch_idx}/{dataset_name}``.
        reports_path (str): The path to the analysis directory. The analysis directory
            is created automatically when the experiment is created and used for saving
            the analysis results generated by the evaluation pipeline -- 
            [DLProfiler](https://github.com/yuxiaoguo/DLProfiler).
    """
    def __init__(self, exp_id, task=None, milestone=None, create=False) -> None:
        self._exp_id = exp_id
        if task is None or milestone is None:
            self._exp_path, self._task, self._milestone = self._fetch_by_exp_id(exp_id)
            assert self._task == task or task is None, \
                f'Expect task {task}, but got {self._task}'
            assert self._milestone == milestone or milestone is None, \
                f'Expect milestone {milestone}, but got {self._milestone}'
        else:
            self._task = task
            self._milestone = milestone
            self._exp_path = os.path.join(GlobalEnvs().exp_root, task, milestone, exp_id)

        # Subfolders
        folders = [self._exp_path]
        self._profiles_path = os.path.join(self._exp_path, 'profiles')
        self._logs_path = os.path.join(self._exp_path, 'logs')
        self._ckpts_path = os.path.join(self._exp_path, 'ckpts')
        self._config_path = os.path.join(self._exp_path, 'configs')
        self._reports_path = os.path.join(self._exp_path, 'reports')
        folders.append(self._profiles_path)
        folders.append(self._logs_path)
        folders.append(self._ckpts_path)
        folders.append(self._config_path)
        folders.append(self._reports_path)

        # Check if the experiment exists
        if not create:
            assert os.path.exists(self._exp_path), \
                f'Expect experiment {exp_id} exists, but got not found'
        else:
            for folder in folders:
                os.makedirs(folder, exist_ok=True)

    @property
    def profiles_path(self) -> str:
        """
        Returns the profiles directory.
        """
        return self._profiles_path

    @property
    def reports_path(self) -> str:
        """
        Returns the analysis directory.
        """
        return self._reports_path

    def _fetch_by_exp_id(self, exp_id) -> tuple[str, str, str]:
        """
        Fetch the task and milestone of an experiment by its ID.

        Args:
            exp_id (str): The ID of the experiment.

        Returns:
            tuple[str, str, str]: The path, task and milestone of the experiment.
        """
        envs = GlobalEnvs()
        exp_paths: list[str] = list()
        for root, folders, _ in os.walk(envs.exp_root):
            if exp_id in folders:
                exp_paths.append(os.path.join(root, exp_id))
        exp_infos: list[tuple[str, str, str]] = list()
        for exp_path in exp_paths:
            exp_infos.append(tuple([exp_path, *exp_path.split(os.path.sep)[-3:-1]]))
        assert len(exp_infos) == 1, \
            f'Expect one experiment with ID {exp_id}, but got {len(exp_infos)}'
        return exp_infos[0]
