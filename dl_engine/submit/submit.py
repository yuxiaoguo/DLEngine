"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=logging-fstring-interpolation
import logging
import argparse

from .test_scripts import submit_local_test
from .train_scripts import submit_local_train


def register_train_parser(sub_parser: argparse._SubParsersAction, train_cmd='local_train'):
    """
    Add the submit_local command to the parser.
    """
    task_parser: argparse.ArgumentParser = sub_parser.add_parser(train_cmd)
    task_parser.description = 'Submit an experiment to the local machine.'

    task_parser.add_argument('--config_path', type=str, help='Path to the config file.')
    task_parser.add_argument('--exp_name', type=str, help='Name of the experiment.')
    task_parser.add_argument('--exp_ctg', type=str, help='Category of the experiment.')
    task_parser.add_argument('--milestone', type=str, help='Milestone of the experiment.')

def register_aml_train_parser(sub_parser: argparse._SubParsersAction, train_cmd='aml_train'):
    """
    Add the submit_local command to the parser.
    """
    task_parser: argparse.ArgumentParser = sub_parser.add_parser(train_cmd)
    task_parser.description = 'Submit an experiment to the aml cluster.'

    task_parser.add_argument('--target', type=str, help='Target Azure to run the experiment on.')
    task_parser.add_argument('--config_path', type=str, help='Path to the config file.')
    task_parser.add_argument('--exp_name', type=str, help='Name of the experiment.')
    task_parser.add_argument('--exp_ctg', type=str, help='Category of the experiment.')
    task_parser.add_argument('--milestone', type=str, help='Milestone of the experiment.')

def register_test_local_parser(sub_parser: argparse._SubParsersAction, test_cmd='local_test'):
    """
    Add the test command to the parser.
    """
    task_parser: argparse.ArgumentParser = sub_parser.add_parser(test_cmd)
    task_parser.description = 'Test the experiment.'

    task_parser.add_argument('--exp_name', type=str, help='Name of the experiment.')
    task_parser.add_argument('--exp_ctg', type=str, help='Category of the experiment.')
    task_parser.add_argument('--milestone', type=str, help='Milestone of the experiment.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    cmd_parser = parser.add_subparsers(help='Command to run.', dest='option')
    register_train_parser(cmd_parser)
    register_aml_train_parser(cmd_parser, train_cmd='aml_train')
    register_test_local_parser(cmd_parser)
    register_test_local_parser(cmd_parser, test_cmd='amlt_fetch')
    args = parser.parse_args()
    if args.option == 'local_train':
        submit_local_train(args.config_path, args.exp_name, args.exp_ctg, args.milestone)
    elif args.option == 'local_test':
        submit_local_test(args.exp_name, args.exp_ctg, args.milestone)
    else:
        raise NotImplementedError(f'Option {args.option} is not implemented.')
