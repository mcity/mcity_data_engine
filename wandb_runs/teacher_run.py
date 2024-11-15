import logging

import wandb

from utils.dataset_loader import (
    load_dataset_info,
    load_fisheye_8k,
    load_mars_multiagent,
    load_mars_multitraversal,
    load_mcity_fisheye_3_months,
    load_mcity_fisheye_2000,
)
from utils.logging import configure_logging
from workflows.teacher import Teacher


def main():
    configure_logging()
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
    teacher = Teacher(
        dataset=dataset,
        config=config,
    )
    teacher.train()


if __name__ == "__main__":
    main()
