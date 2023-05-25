"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
"""
# pylint: disable=logging-fstring-interpolation
import os
import logging
import git
import yaml

from amlt.cli import project as amlt_project
from amlt.api.registry import ProjectRegistry

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



def switch_amlt_project(exp_ctg, milestone, exp_root, allow_create=True):
    """
    Switch the AMLT project.
    """
    storage_account_name = 'igsharestorage'
    storage_container_name = 'yuxgu'
    t0_registry_name = 'projects/mercury'
    registry_name = f'{t0_registry_name}/{exp_ctg}'
    project_name = f'{milestone}'

    project_registry = \
        ProjectRegistry(storage_account_name, storage_container_name, registry_name)
    existed_projects = [_p.name for _p in project_registry.projects]  # type: ignore
    if project_name not in existed_projects and allow_create:
        logging.info('Creating a new project...')
        amlt_project.create_project(
            project_name=project_name,
            storage_account_name=storage_account_name,
            storage_container_name=storage_container_name,
            registry_name=registry_name,
            blob_storage_account_name=storage_account_name,
            output_storage_path_format=None)
    logging.info(f'Checkout project {project_name} at entry {registry_name}')
    amlt_project.checkout_project(
        project_name=project_name,
        storage_account_name=storage_account_name,
        storage_container_name=storage_container_name,

        registry_name=registry_name,
        project_path='.',
        default_output_dir=exp_root,
        blob_storage_account_name=storage_account_name,
        output_storage_path_format=None)

    return t0_registry_name


def setup_exp_yaml(exp_name, run_cmd: str, out_path: str, sku="G4", num_rounds=1, target='sing'):
    """
    Generate the experiment yaml
    """
    resource_dir = os.path.dirname(os.path.abspath(__file__))
    tmpl_file = os.path.join(resource_dir, 'resources', 'amlt_template.yaml')

    with open(tmpl_file, 'r', encoding='utf-8') as meta_fp:
        xbx_ctx = yaml.safe_load(meta_fp)
    exp_list = list()

    priority_dict = dict(
        sing=dict(sla_tier='premium'),
        amlk8s=dict(priority='high', preemptible='False'),
    )
    for n_r in range(num_rounds):
        exp_dict = dict(
            name=f'{exp_name}-R{n_r}',
            sku=sku,
            execution_mode='Basic',
            command=[run_cmd],
            **priority_dict[target]
        )
        exp_list.append(exp_dict)
    xbx_ctx['jobs'] = exp_list

    # Select target cluster
    target_dict = dict(
        sing=dict(service='sing', name='msroctovc'),
        amlk8s=dict(service='amlk8s', name='itplabrr1cl1'),
    )
    xbx_ctx['target'] = target_dict[target]

    with open(out_path, 'w', encoding='utf-8') as meta_fp:
        yaml.dump(xbx_ctx, meta_fp)


def submit_amlt_train(config_path: str, exp_name: str, exp_ctg: str, milestone: str):
    """
    Submit an experiment to the AMLT.
    """
    path_ctx = register_environment(config_path, exp_name, exp_ctg, milestone)
    if path_ctx is None:
        return

    # Whether to create a new project
    logging.info('Auto-decise whether to create a new project...')
    registry_path = switch_amlt_project(exp_ctg, milestone, path_ctx.exp_root)

    # Generate the amlt description yaml
    amlt_desc_path = os.path.join(path_ctx.exp_cfg_dir, 'amlt_desc.yaml')
    amlt_target_path_ctx = \
        ExpPathManager(exp_name, exp_ctg, milestone, False, f'/mnt/input/{registry_path}')
    run_cmd = setup_run_cmd(amlt_target_path_ctx)
    setup_exp_yaml(exp_name, run_cmd, amlt_desc_path)

    # Upload the experiment config files
    amlt_path_ctx = ExpPathManager(exp_name, exp_ctg, milestone, False, registry_path)
    os.system(f'amlt storage upload {path_ctx.exp_cfg_dir} {amlt_path_ctx.exp_cfg_dir}')

    logging.info('Start AMLT job submitting...')
    submit_cmd = f'amlt run {amlt_desc_path} {exp_name} -d {exp_name}'
    os.system(submit_cmd)


def submit_amlt_fetch(exp_name: str, exp_ctg: str, milestone: str):
    """
    Fetch training results
    """
    logging.info('Start AMLT job fetching...')
    path_ctx = ExpPathManager(exp_name, exp_ctg, milestone, False, None)

    logging.info('Checkout existed project...')
    registry_path = switch_amlt_project(exp_ctg, milestone, path_ctx.exp_root, False)
    amlt_path_ctx = ExpPathManager(exp_name, exp_ctg, milestone, False, registry_path)

    os.system(f'amlt storage download {amlt_path_ctx.exp_log_dir} {path_ctx.exp_log_dir}')
    os.system(f'amlt storage download {amlt_path_ctx.exp_ckpt_dir} {path_ctx.exp_ckpt_dir}' +\
        ' -I latest.pt -I final.pt -Ickpt_state')
