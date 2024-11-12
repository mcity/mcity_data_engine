import argparse
import json
import logging
import signal
import sys
import time

import wandb
from tqdm import tqdm

from ano_dec import Anodec
from brain import Brain
from config.config import (
    ANOMALIB_IMAGE_MODELS,
    SELECTED_DATASET,
    SELECTED_WORKFLOW,
    V51_ADDRESS,
    V51_PORT,
    WORKFLOWS,
)
from ensemble_exploration import EnsembleExploration
from teacher import Teacher
from utils.data_loader import FiftyOneTorchDatasetCOCO
from utils.dataset_loader import *
from utils.logging import configure_logging
from utils.wandb_helper import launch_to_queue_terminal


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

    if args.tags:
        wandb_tags = args.tags.split(",")
    if args.queue == None:
        signal.signal(signal.SIGINT, signal_handler)  # Signal handler for CTRL+C

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
        embedding_models = WORKFLOWS["brain_selection"]["embedding_models"]
        config_file_path = "wandb_runs/brain_config.json"
        with open(config_file_path, "r") as file:
            config = json.load(file)
        config["overrides"]["run_config"]["v51_dataset_name"] = SELECTED_DATASET
        wandb_project = "Data Engine Brain"

        for MODEL_NAME in (pbar := tqdm(embedding_models, desc="Brain")):
            run = None
            try:
                pbar.set_description("Generating/Loading embeddings with " + MODEL_NAME)
                config["overrides"]["run_config"]["model_name"] = MODEL_NAME

                if args.queue == None:
                    run = wandb.init(
                        name=MODEL_NAME,
                        allow_val_change=True,
                        sync_tensorboard=True,
                        group="Brain",
                        job_type="eval",
                        project=wandb_project,
                        config=config,
                    )
                    wandb_config = wandb.config["overrides"]["run_config"]
                    run.tags += (
                        wandb_config["v51_dataset_name"],
                        wandb_config["v51_dataset_name"],
                        "local",
                    )
                    # Compute model embeddings and index for similarity
                    v51_brain = Brain(dataset, dataset_info, MODEL_NAME)
                    v51_brain.compute_embeddings()
                    v51_brain.compute_similarity()

                    # Find representative and unique samples as center points for further selections
                    v51_brain.compute_representativeness()
                    v51_brain.compute_unique_images_greedy()
                    v51_brain.compute_unique_images_deterministic()

                    # Select samples similar to the center points to enlarge the dataset
                    v51_brain.compute_similar_images()
                    run.finish(exit_code=0)
            except Exception as e:
                logging.error(f"An error occurred with model {MODEL_NAME}: {e}")
                logging.error("Stack trace:", exc_info=True)
                if run:
                    run.finish(exit_code=1)
                continue

    elif SELECTED_WORKFLOW == "learn_normality":
        config_file_path = "wandb_runs/anomalib_config.json"
        with open(config_file_path, "r") as file:
            config = json.load(file)
        config["overrides"]["run_config"]["v51_dataset_name"] = SELECTED_DATASET
        wandb_project = "Data Engine Anomalib"

        for MODEL_NAME in (pbar := tqdm(ANOMALIB_IMAGE_MODELS, desc="Anomalib")):
            pbar.set_description("Training/Loading Anomalib model " + MODEL_NAME)
            config["overrides"]["run_config"]["model_name"] = MODEL_NAME

            if args.queue == None:
                run = wandb.init(
                    allow_val_change=True,
                    sync_tensorboard=True,
                    project=wandb_project,
                    group="Anomalib",
                    job_type="train",
                    config=config,
                )
                config = wandb.config["overrides"]["run_config"]
                run.tags += (config["v51_dataset_name"], config["model_name"], "local")
                ano_dec = Anodec(
                    dataset=dataset,
                    dataset_info=dataset_info,
                    config=config,
                )
                ano_dec.train_and_export_model()
                ano_dec.run_inference()
                ano_dec.eval_v51()
                del ano_dec

            elif args.queue != None:
                # Update config file
                with open(config_file_path, "w") as file:
                    json.dump(config, file, indent=4)

                # Add job to queue
                wandb_entry_point = config["overrides"]["entry_point"]
                launch_to_queue_terminal(
                    name=MODEL_NAME,
                    project=wandb_project,
                    config_file=config_file_path,
                    entry_point=wandb_entry_point,
                    queue=args.queue,
                )

    elif SELECTED_WORKFLOW == "train_teacher":
        teacher_models = WORKFLOWS["train_teacher"]["hf_models_objectdetection"]
        wandb_project = "Data Engine Teacher"
        config_file_path = "wandb_runs/teacher_config.json"
        with open(config_file_path, "r") as file:
            config = json.load(file)

        # Train teacher models
        for MODEL_NAME in (pbar := tqdm(teacher_models, desc="Teacher Models")):
            run = None
            try:
                pbar.set_description("Training/Loading Teacher model " + MODEL_NAME)
                config["overrides"]["run_config"]["model_name"] = MODEL_NAME
                config["overrides"]["run_config"]["v51_dataset_name"] = SELECTED_DATASET
                if args.queue == None:

                    run = wandb.init(
                        name=MODEL_NAME,
                        allow_val_change=True,
                        sync_tensorboard=True,
                        group="Teacher",
                        job_type="train",
                        config=config,
                        project=wandb_project,
                    )
                    wandb_config = wandb.config["overrides"]["run_config"]

                    run.tags += (
                        wandb_config["v51_dataset_name"],
                        "local",
                    )
                    teacher = Teacher(
                        dataset=dataset,
                        config=wandb_config,
                    )

                    teacher.train()
                    run.finish(exit_code=0)
                elif args.queue != None:
                    # Update config file
                    with open(config_file_path, "w") as file:
                        json.dump(config, file, indent=4)

                    wandb_entry_point = config["overrides"]["entry_point"]
                    # Add job to queue
                    launch_to_queue_terminal(
                        name=MODEL_NAME,
                        project=wandb_project,
                        config_file=config_file_path,
                        entry_point=wandb_entry_point,
                        queue=args.queue,
                    )
            except Exception as e:
                logging.error(f"An error occurred with model {MODEL_NAME}: {e}")
                logging.error("Stack trace:", exc_info=True)
                if run:
                    run.finish(exit_code=1)
                continue

    elif SELECTED_WORKFLOW == "zero_shot_teacher":
        zero_shot_teacher_models = WORKFLOWS["train_teacher"][
            "hf_models_zeroshot_objectdetection"
        ]
        wandb_project = "Data Engine Teacher"
        config_file_path = "wandb_runs/teacher_zero_shot_config.json"
        with open(config_file_path, "r") as file:
            config = json.load(file)

        pytorch_dataset = FiftyOneTorchDatasetCOCO(dataset)
        for MODEL_NAME in (pbar := tqdm(zero_shot_teacher_models, desc="Zero Shot")):
            run = None
            try:
                pbar.set_description("Evaluating Zero Shot Teacher model " + MODEL_NAME)
                config["overrides"]["run_config"]["model_name"] = MODEL_NAME
                config["overrides"]["run_config"]["v51_dataset_name"] = SELECTED_DATASET
                if args.queue == None:

                    run = wandb.init(
                        name=MODEL_NAME,
                        allow_val_change=True,
                        sync_tensorboard=True,
                        group="Teacher",
                        job_type="eval",
                        config=config,
                        project=wandb_project,
                    )
                    wandb_config = wandb.config["overrides"]["run_config"]

                    run.tags += (wandb_config["v51_dataset_name"], "local", "zero-shot")

                    teacher = Teacher(dataset=dataset, config=wandb_config)
                    batch_size = wandb_config["batch_size"]
                    detection_threshold = wandb_config["detection_threshold"]
                    object_classes = wandb_config["object_classes"]
                    if len(object_classes) == 0:
                        object_classes = None
                    teacher.zero_shot_inference(
                        pytorch_dataset=pytorch_dataset,
                        batch_size=batch_size,
                        detection_threshold=detection_threshold,
                        object_classes=object_classes,
                    )
                    run.finish(exit_code=0)
            except Exception as e:
                logging.error(f"An error occurred with model {MODEL_NAME}: {e}")
                logging.error("Stack trace:", exc_info=True)
                if run:
                    run.finish(exit_code=1)
                continue
    elif SELECTED_WORKFLOW == "ensemble_exploration":
        run = None
        try:
            wandb_project = "Data Engine Ensemble Exploration"
            config_file_path = "wandb_runs/ensemble_exploration_config.json"
            with open(config_file_path, "r") as file:
                config = json.load(file)
            config["overrides"]["run_config"]["v51_dataset_name"] = SELECTED_DATASET
            if args.queue == None:
                run = wandb.init(
                    name="ensemble-exploration",
                    allow_val_change=True,
                    sync_tensorboard=True,
                    group="Exploration",
                    job_type="eval",
                    config=config,
                    project=wandb_project,
                )
                wandb_config = wandb.config["overrides"]["run_config"]
                run.tags += (
                    wandb_config["v51_dataset_name"],
                    "local",
                    "ensemble-exploration",
                )
                explorer = EnsembleExploration(dataset, wandb_config)
                explorer.ensemble_exploration()
                run.finish(exit_code=0)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.error("Stack trace:", exc_info=True)
            if run:
                run.finish(exit_code=1)

    else:
        logging.error(
            str(SELECTED_WORKFLOW)
            + " is not a valid workflow. Check _WORKFLOWS_ in config.py."
        )

    # Launch V51 session
    if args.queue == None:
        dataset.save()
        logging.info(dataset)
        fo.pprint(dataset.stats(include_media=True))
        session = fo.launch_app(
            dataset, spaces=spaces, address=V51_ADDRESS, port=V51_PORT, remote=True
        )

    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script locally or in W&B queue.")
    parser.add_argument(
        "--queue",
        choices=[None, "data-engine", "lighthouse"],
        default=None,
        help="If no WandB queue selected, code will run locally.'",
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="Comma-separated list of WandB tags",
    )
    args = parser.parse_args()
    main(args)
