import datetime
import logging
import os
import re

import wandb

from config.config import WANDB_ACTIVE


def wandb_init(
    run_name,
    project_name,
    dataset_name,
    config=None,
    log_dir_root="logs/tensorboard/",
    log_dir=None,
    sync_tensorboard=True,
    wandb_activate_local=True,
):
    # Logging dir
    if log_dir is None:
        project_folder = re.sub(r"[^\w\-_]", "_", project_name)
        run_folder = re.sub(r"[^\w\-_]", "_", run_name)
        now = datetime.datetime.now()
        time_folder = now.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_dir_root, project_folder, run_folder, time_folder)

    wandb.tensorboard.patch(root_logdir=log_dir)

    wandb_run = None
    if wandb_activate_local and WANDB_ACTIVE:
        wandb_run = wandb.init(
            name=run_name,
            sync_tensorboard=sync_tensorboard,
            project=project_name,
            tags=[dataset_name],
            config=config,
        )
        logging.info(
            f"Launched Weights and Biases run {wandb_run} with config {config}"
        )
    else:
        logging.warning(
            "WandB run was not initialized. Set 'wandb_activate' and WANDB_ACTIVE to True."
        )

    return wandb_run, log_dir


def wandb_close(exit_code=0):

    try:
        logging.info(f"Closing Weights and Biases run with exit_code {exit_code}")
        wandb.finish(exit_code=exit_code)
    except:
        logging.warning(f"WandB run could not be finished")
    try:
        wandb.tensorboard.unpatch()
    except:
        logging.warning(f"WandB tensorboard could not be unpatched")
