"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=logging-fstring-interpolation
import os
import logging
import git

from .utils import ExpPathManager


def register_environment(config_path: str, exp_name: str, exp_ctg: str, milestone: str):
    """
    Register the environment to AMLT/local workspace.
    """
    path_ctx = ExpPathManager(exp_name, exp_ctg, milestone, create_folder=True)

    gen_flag_str = '# Template Generated: (DO NOT DELETE IT UNLESS THE SCRIPT IS READY!)\n'

    src_config_files = os.listdir(config_path)
    for file in src_config_files:
        if os.path.exists(os.path.join(path_ctx.exp_cfg_dir, file)):
            continue
        if not file.endswith('.yaml'):
            continue
        with open(os.path.join(config_path, file), 'r', encoding='utf-8') as f_src:
            src_lines = f_src.readlines()
        with open(os.path.join(path_ctx.exp_cfg_dir, file), 'w', encoding='utf-8') as f_dst:
            dst_lines = [gen_flag_str]
            dst_lines.extend(src_lines)
            f_dst.writelines(dst_lines)

    dst_config_files: list[str] = os.listdir(path_ctx.exp_cfg_dir)
    template_files = list()
    for file in dst_config_files:
        if not file.endswith('.yaml'):
            continue
        with open(os.path.join(path_ctx.exp_cfg_dir, file), 'r', encoding='utf-8') as f_dst:
            if f_dst.readline().find(gen_flag_str) != -1:
                template_files.append(file)
    if template_files:
        logging.warning(f'The following files are not ready: {template_files}')
        return

    logging.info('All config files are ready. Start submitting the experiment.')
    repo = git.Repo(search_parent_directories=True)  # type: ignore
    if repo.active_branch.name != 'main':
        logging.warning('The current branch is not main. Please check it.')
    with open(os.path.join(path_ctx.exp_dir, 'git.md5'), 'w', encoding='utf-8') as f_git:
        f_git.write(repo.head.object.hexsha)
    return path_ctx


def setup_run_cmd(path_ctx: ExpPathManager):
    """
    Setup the run command.
    """
    run_cmd = 'python run.py'
    run_cmd += f' --config_path {path_ctx.exp_cfg_dir}/train.yaml'
    run_cmd += f' --log_dir {path_ctx.exp_log_dir}'
    run_cmd += f' --ckpt_dir {path_ctx.exp_ckpt_dir}'
    run_cmd += f' --prof_dir {path_ctx.exp_prof_dir}'
    return run_cmd



def submit_local_train(config_path: str, exp_name: str, exp_ctg: str, milestone: str):
    """
    Submit an experiment to the local machine.
    """
    path_ctx = register_environment(config_path, exp_name, exp_ctg, milestone)
    if path_ctx is None:
        return

    logging.info('Start local training...')
    run_cmd = setup_run_cmd(path_ctx)
    logging.info(f'Running command: {run_cmd}')
    os.system(run_cmd)
