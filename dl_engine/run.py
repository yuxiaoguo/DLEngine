"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=unused-import
import logging
import argparse

from dl_engine.core.pipeline import Pipeline
from dl_engine import data

from tokenizer import vqvae
from generator import lfm
from functional import callbacks


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the train.yaml.')
    parser.add_argument('--log_dir', type=str, help='Path to the log directory.')
    parser.add_argument('--ckpt_dir', type=str, help='Path to the checkpoint directory.')
    parser.add_argument('--prof_dir', type=str, help='Path to the profiler directory.')
    args = parser.parse_args()
    pipe_core = Pipeline(args.config_path, args.log_dir, args.ckpt_dir, args.prof_dir)
    pipe_core.run()
