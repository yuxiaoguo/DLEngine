"""
Copyright (c) 2023 Yu-Xiao Guo All rights reserved.
"""
# pylint: disable=unused-import
import os
import logging
import argparse

from dl_engine.core.pipeline import Pipeline
from dl_engine import data


def scanning_user_registered_modules(cwd=None, registery_cands=None):
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
    _ = [__import__(py_module_str) for py_module_str in py_modules_str]

if __name__ == '__main__':
    scanning_user_registered_modules()

    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the train.yaml.')
    parser.add_argument('--log_dir', type=str, help='Path to the log directory.')
    parser.add_argument('--ckpt_dir', type=str, help='Path to the checkpoint directory.')
    parser.add_argument('--prof_dir', type=str, help='Path to the profiler directory.')
    args = parser.parse_args()
    pipe_core = Pipeline(args.config_path, args.log_dir, args.ckpt_dir, args.prof_dir)
    pipe_core.run()
