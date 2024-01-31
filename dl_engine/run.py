"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=unused-import
import os
import logging
import argparse
import tempfile

import wandb
from torch import distributed as dist

from dl_engine.core.pipeline import Pipeline
from dl_engine import data


def scanning_user_registered_modules(cwd=None, registery_cands=None, pkg_name=None):
    """
    Scanning user registered modules.
    """
    if cwd is None:
        cwd = os.getcwd()

    if registery_cands is None:
        registery_cands = [
            'network_register',
            'ref_network_register',
            'dataset_register',
            'sampler_register',
            'functional_register',
            'optimizer_register',
            'dataloader_register'
        ]

    # Get all python files and folders in the current directory recursively.
    py_files: list[str] = list()
    for root, _, files in os.walk(cwd):
        for file in files:
            if not file.endswith('.py'):
                continue
            py_path = os.path.join(root, file)
            registery = False
            with open(py_path, 'r', encoding='utf-8') as p_fp:
                for line in p_fp.readlines():
                    for registery_cand in registery_cands:
                        if registery_cand in line and line.startswith('@'):
                            registery = True
                            break
                    if registery:
                        break
            if registery:
                py_files.append(py_path)
    py_modules_str = [py_file.replace(cwd, '').replace('/', '.').replace('\\', '.')[1:-3]\
        for py_file in py_files]
    _ = [__import__(f'{pkg_name}.{py_module_str}' if pkg_name else py_module_str) \
        for py_module_str in py_modules_str]

if __name__ == '__main__':
    scanning_user_registered_modules()

    parser = argparse.ArgumentParser()
    parser.add_argument(\
        '--config_path', type=str, help='Path to the train.yaml.')
    parser.add_argument(\
        '--log_dir', type=str, default='', help='Path to the log directory.')
    parser.add_argument(\
        '--ckpt_dir', type=str, default='', help='Path to the checkpoint directory.')
    parser.add_argument(\
        '--prof_dir', type=str, default='', help='Path to the profiler directory.')
    parser.add_argument(\
        '--log_level', type=str, default='INFO', help='Log level.')
    parser.add_argument(\
        '--num_nodes', type=int, default=1, help='Number of nodes.')
    parser.add_argument(\
        '--wandb_key', type=str, default='', help='Wandb key.')
    args = parser.parse_args()

    if args.wandb_key != '' and ((not dist.is_initialized()) or dist.get_rank() == 0):
        wandb.login(key=args.wandb_key)
        config_path = args.config_path
        config_seps = config_path.replace('\\', '/').split('/')
        task, ms, job_id, _, _ = config_seps[-5:]
        project = f'{task}-{ms}'
        wandb.init(project=project, name=job_id, sync_tensorboard=True, dir=args.log_dir)
        os.makedirs(args.log_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix='dl_engine_')
    if args.log_dir == '':
        args.log_dir = os.path.join(tmp_dir, 'dl_engine_log')
    if args.ckpt_dir == '':
        args.ckpt_dir = os.path.join(tmp_dir, 'dl_engine_ckpt')
    if args.prof_dir == '':
        args.prof_dir = os.path.join(tmp_dir, 'dl_engine_prof')

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, handlers=[logging.StreamHandler()])
    pipe_core = Pipeline(\
        args.config_path, args.log_dir, args.ckpt_dir, args.prof_dir, num_nodes=args.num_nodes)
    pipe_core.run()
