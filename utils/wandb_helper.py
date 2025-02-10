import datetime
import logging
import os
import re

import wandb


def wandb_init(
    run_name,
    project_name,
    dataset_name,
    config=None,
    log_dir_root="logs/tensorboard/",
    log_dir=None,
    sync_tensorboard=True,
):
    # Logging dir
    if log_dir is None:
        project_folder = re.sub(r"[^\w\-_]", "_", project_name)
        run_folder = re.sub(r"[^\w\-_]", "_", run_name)
        now = datetime.datetime.now()
        time_folder = now.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_dir_root, project_folder, run_folder, time_folder)

    wandb.tensorboard.patch(root_logdir=log_dir)

    wandb_run = wandb.init(
        name=run_name,
        sync_tensorboard=sync_tensorboard,
        project=project_name,
        tags=[dataset_name],
        config=config,
    )

    return wandb_run, log_dir


def wandb_close(wandb_run=None, exit_code=0):
    try:
        wandb_run.finish(exit_code=exit_code)
    except:
        logging.debug(f"WandB run {wandb_run} could not be finished")
    try:
        wandb.tensorboard.unpatch()
    except:
        logging.debug(f"WandB tensorboard could not be unpatched")
