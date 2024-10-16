from teacher import Teacher
from utils.dataset_loader import *

import wandb
import logging


def main():
    # Get dataset with Voxel51
    run = wandb.init(
        allow_val_change=True,
        sync_tensorboard=True,
        group="Teacher",
        job_type="train",
    )
    config = wandb.config

    run.tags += (config["v51_dataset_name"], config["model_name"], "docker")
    dataset_name = config["v51_dataset_name"]
    dataset_info = load_dataset_info(dataset_name)
    if dataset_info:
        loader_function = dataset_info.get("loader_fct")
        dataset = globals()[loader_function](dataset_info)
    else:
        logging.error(
            str(dataset_name)
            + " is not a valid dataset name. Check _datasets_ in datasets.yaml."
        )

    # Train with Huggingface Trainer
    teacher = Teacher(dataset, dataset_info, config)
    teacher.train()


if __name__ == "__main__":
    main()
