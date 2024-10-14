import argparse
import time
import signal
import json

from utils.logging import configure_logging
from utils.wandb_helper import launch_to_queue_terminal, get_wandb_conf
import logging

from config.config import (
    SELECTED_WORKFLOW,
    SELECTED_DATASET,
    V51_EMBEDDING_MODELS,
    ANOMALIB_IMAGE_MODELS,
)

from tqdm import tqdm

from utils.data_loader import *
from brain import Brain
from ano_dec import Anodec
from teacher import Teacher

import wandb

import sys


def panel_embeddings(v51_brain, color_field="unique"):
    """
    Creates a layout of panels for visualizing sample embeddings.

    Args:
        v51_brain: An object containing the embeddings visualization data.
        color_field (str, optional): The field used to color the embeddings. Defaults to "unique".

    Returns:
        fo.Space: A layout space containing the samples panel and embeddings panel.
    """
    samples_panel = fo.Panel(type="Samples", pinned=True)

    embeddings_panel = fo.Panel(
        type="Embeddings",
        state=dict(
            brainResult=next(iter(v51_brain.embeddings_vis)), colorByField=color_field
        ),
    )

    spaces = fo.Space(
        children=[
            fo.Space(
                children=[
                    fo.Space(children=[samples_panel]),
                ],
                orientation="horizontal",
            ),
            fo.Space(children=[embeddings_panel]),
        ],
        orientation="horizontal",
    )

    return spaces


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    # Perform any cleanup or final actions here
    try:
        wandb.finish()
    except:
        pass
    sys.exit(0)


def main(args):
    time_start = time.time()
    configure_logging()

    if args.run_mode == "local":
        signal.signal(signal.SIGINT, signal_handler)  # Signal handler for CTRL+C

        # Move results from remote Docker runs to local folder

        # Load the selected dataset
        dataset_info = load_dataset_info(SELECTED_DATASET)

        if dataset_info:
            loader_function = dataset_info.get("loader_fct")
            dataset = globals()[loader_function](dataset_info)
        else:
            logging.error(
                str(SELECTED_DATASET)
                + " is not a valid dataset name. Check _datasets_ in datasets.yaml."
            )
        spaces = None

    logging.info("Running workflow " + SELECTED_WORKFLOW.upper())
    if SELECTED_WORKFLOW == "brain_selection":
        v51_brain = Brain(dataset, dataset_info)
        # Compute model embeddings and index for similarity
        v51_brain.compute_embeddings(V51_EMBEDDING_MODELS)
        v51_brain.compute_similarity()

        # Find representative and unique samples as center points for further selections
        v51_brain.compute_representativeness()
        v51_brain.compute_unique_images_greedy()
        v51_brain.compute_unique_images_deterministic()

        # Select samples similar to the center points to enlarge the dataset
        v51_brain.compute_similar_images()
        spaces = panel_embeddings(v51_brain)

    elif SELECTED_WORKFLOW == "learn_normality":
        config_file_path = "wandb_runs/anomalib_config.json"
        with open(config_file_path, "r") as file:
            config = json.load(file)
        config["overrides"]["run_config"]["v51_dataset_name"] = SELECTED_DATASET
        wandb_project = "Data Engine Anomalib"

        for MODEL_NAME in (pbar := tqdm(ANOMALIB_IMAGE_MODELS, desc="Anomalib")):
            pbar.set_description("Training/Loading Anomalib model " + MODEL_NAME)
            config["overrides"]["run_config"]["model_name"] = MODEL_NAME

            if args.run_mode == "local":
                run = wandb.init(
                    allow_val_change=True,
                    sync_tensorboard=True,
                    project=wandb_project,
                    group="Anomalib",
                    job_type="train",
                    config=config,
                )
                config = wandb.config["overrides"]["run_config"]
                run.tags += (config["v51_dataset_name"], config["model_name"])
                ano_dec = Anodec(
                    dataset=dataset,
                    dataset_info=dataset_info,
                    config=config,
                    wandb_run=run,
                    wandb_project=wandb_project,
                )
                ano_dec.train_and_export_model()
                ano_dec.run_inference()
                ano_dec.eval_v51()
                del ano_dec

            elif args.run_mode == "wandb":
                # Update config file
                with open(config_file_path, "w") as file:
                    json.dump(config, file, indent=4)

                # Add job to queue
                launch_to_queue_terminal(
                    name="Anomalib_" + SELECTED_DATASET + "_" + MODEL_NAME,
                    project=wandb_project,
                    config_file=config_file_path,
                )

    elif SELECTED_WORKFLOW == "train_teacher":
        if args.run_mode == "local":
            wandb_project = "Data Engine Teacher"
            config_file_path = "wandb_runs/teacher_config.json"
            with open(config_file_path, "r") as file:
                config = json.load(file)

            run = wandb.init(
                allow_val_change=True,
                sync_tensorboard=True,
                group="Anomalib",
                job_type="train",
                config=config,
                project=wandb_project,
            )

            config = wandb.config["overrides"]["run_config"]
            teacher = Teacher(
                dataset,
                dataset_info,
                config,
                wandb_project,
            )

            finetune = True
            teacher.train(finetune)

    else:
        logging.error(
            str(SELECTED_WORKFLOW)
            + " is not a valid workflow. Check _WORKFLOWS_ in config.py."
        )

    # Create layout for the web interface

    # Launch V51 session
    if args.run_mode == "local":
        dataset.save()
        logging.info(dataset)
        fo.pprint(dataset.stats(include_media=True))
        session = fo.launch_app(dataset, spaces=spaces)
        session.wait(-1)

    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script locally or in W&B queue.")
    parser.add_argument(
        "--run-mode",
        choices=["local", "wandb"],
        default="local",
        help="Run mode: 'local' or 'wandb queue'",
    )
    args = parser.parse_args()
    main(args)
