"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=logging-fstring-interpolation
import os
import logging
from typing import List

import numpy as np

from .utils import ExpPathManager


def submit_local_test(exp_name: str, exp_ctg: str, milestone: str):
    """
    Test the trained model.
    """
    logging.info(f'Testing the trained model {exp_name}...')
    path_ctx = ExpPathManager(exp_name, exp_ctg, milestone, create_folder=True)

    config_path = os.path.join(path_ctx.exp_cfg_dir, 'test.yaml')
    logging.info(f'Testing wih config file: {config_path}')

    last_ckpt = os.path.join(path_ctx.exp_ckpt_dir, 'final.pt')
    if os.path.exists(last_ckpt):
        ckpt_path = last_ckpt
        logging.info(f'Select last checkpoint: {ckpt_path}')
    else:
        ckpt_files: List[str] = os.listdir(path_ctx.exp_ckpt_dir)
        ckpt_epochs = [int(os.path.splitext(_f)[0].split('_')[1]) for _f in ckpt_files]
        latest_ckpt = ckpt_files[np.argmax(ckpt_epochs)]
        ckpt_path = os.path.join(path_ctx.exp_ckpt_dir, latest_ckpt)
        logging.info(f'Select latest checkpoint: {ckpt_path}')

    run_cmd = 'python -m dl_engine.run'
    run_cmd += f' --config_path {config_path}'
    run_cmd += f' --log_dir {os.path.abspath(path_ctx.exp_log_dir)}'
    run_cmd += f' --ckpt_dir {ckpt_path}'
    run_cmd += f' --prof_dir {os.path.abspath(path_ctx.exp_prof_dir)}'
    logging.info(f'Running command: {run_cmd}')
    os.system(run_cmd)
