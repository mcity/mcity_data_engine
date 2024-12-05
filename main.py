import argparse
import datetime
import json
import logging
import signal
import sys
import time
import warnings

IGNORE_FUTURE_WARNINGS=True
if IGNORE_FUTURE_WARNINGS:
    warnings.simplefilter("ignore", category=FutureWarning)
import fiftyone as fo
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm

from config.config import (SELECTED_DATASET, SELECTED_WORKFLOW, V51_ADDRESS,
                           V51_PORT, WORKFLOWS)
from utils.data_loader import FiftyOneTorchDatasetCOCO
# Called with globals()
from utils.dataset_loader import (load_dataset_info, load_fisheye_8k,
                                  load_mars_multiagent,
                                  load_mars_multitraversal,
                                  load_mcity_fisheye_3_months,
                                  load_mcity_fisheye_2000)
from utils.logging import configure_logging
from utils.mp_distribution import ZeroShotDistributer
from utils.wandb_helper import launch_to_queue_terminal
from workflows.ano_dec import Anodec
from workflows.aws_download import AwsDownloader
from workflows.brain import Brain
from workflows.ensemble_exploration import EnsembleExploration
from workflows.teacher import Teacher, ZeroShotTeacher


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    # Perform any cleanup or final actions here
    try:
        wandb.finish()
    except:
        pass
    sys.exit(0)

def workflow_brain(dataset, dataset_info, MODEL_NAME):
    v51_brain = Brain(dataset, dataset_info, MODEL_NAME)
    v51_brain.compute_embeddings()
    v51_brain.compute_similarity()

    # Find representative and unique samples as center points for further selections
    v51_brain.compute_representativeness()
    v51_brain.compute_unique_images_greedy()
    v51_brain.compute_unique_images_deterministic()

    # Select samples similar to the center points to enlarge the dataset
    v51_brain.compute_similar_images()

    v51_keys = {}
    v51_keys["embedding"] = v51_brain.embedding_key
    v51_keys["similarity"] = v51_brain.similiarity_key
    v51_keys["uniqueness"] = v51_brain.uniqueness_key

    return v51_keys

def workflow_zero_shot_teacher(dataset, dataset_info):
    # Zero-shot object detector teacher models from Huggingface
    # Optimized for parallel multi-GPU inference, also supports single GPU
    config = WORKFLOWS["zero_shot_teacher"]
    dataset_torch = FiftyOneTorchDatasetCOCO(dataset)
    teacher = ZeroShotTeacher(dataset_torch=dataset_torch, dataset_info=dataset_info, config=config)
    
    # Check if model detections are already stored in V51 dataset or on disk
    # TODO Think about config. Teacher already has it in init before it is processed, but does not use the hf_models. Could be more elegant.
    models_splits_dict = teacher.exclude_stored_predictions(dataset_v51=dataset, config=config)
    if len(models_splits_dict) > 0:
        config["hf_models_zeroshot_objectdetection"] = models_splits_dict
        distributor = ZeroShotDistributer(config=config, n_samples=len(dataset_torch), dataset_info=dataset_info, teacher=teacher)
        distributor.distribute_and_run()
    else:
        logging.info("All teacher models already have predictions stored in the dataset.")
    

def main(args):
    configure_logging()
    time_start = time.time()
    selected_dataset = SELECTED_DATASET

    if args.tags:
        wandb_tags = args.tags.split(",")
    if args.queue == None:
        signal.signal(signal.SIGINT, signal_handler)  # Signal handler for CTRL+C

        # Load the selected dataset
        dataset_info = load_dataset_info(selected_dataset)

        if dataset_info:
            loader_function = dataset_info.get("loader_fct")
            dataset = globals()[loader_function](dataset_info)
        else:
            logging.error(
                str(selected_dataset)
                + " is not a valid dataset name. Check _datasets_ in datasets.yaml."
            )
        spaces = None

    logging.info(
        f"Running workflows {SELECTED_WORKFLOW} for dataset {selected_dataset}"
    )

    workflows_started = 0
    if "aws_download" in SELECTED_WORKFLOW:
        workflows_started += 1
        run = None
        try:
            test_run = WORKFLOWS["aws_download"]["test_run"]
            source = WORKFLOWS["aws_download"]["source"]
            sample_rate = WORKFLOWS["aws_download"]["sample_rate_hz"]
            start_date_str = WORKFLOWS["aws_download"]["start_date"]
            end_date_str = WORKFLOWS["aws_download"]["end_date"]
            start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
            dataset_name = f"data_engine_rolling_{start_date_str}_to_{end_date_str}"

            run = wandb.init(
                name=dataset_name,
                sync_tensorboard=True,
                group="S3",
                job_type="download",
                project="Data Engine Download",
            )

            aws_downloader = AwsDownloader(
                name=dataset_name,
                start_date=start_date,
                end_date=end_date,
                source=source,
                sample_rate_hz=sample_rate,
                test_run=test_run,
            )
            dataset = aws_downloader.load_data()

            # Select downloaded dataset for further workflows if configured
            selected_dataset_overwrite = WORKFLOWS["aws_download"][
                "selected_dataset_overwrite"
            ]
            if selected_dataset_overwrite:
                selected_dataset = dataset_name
                logging.info(f"Selected dataset overwritten to {selected_dataset}")

            run.finish(exit_code=0)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            logging.error("Stack trace:", exc_info=True)
            if run:
                run.finish(exit_code=1)

    if "brain_selection" in SELECTED_WORKFLOW:
        workflows_started += 1
        embedding_models = WORKFLOWS["brain_selection"]["embedding_models"]
        config_file_path = "wandb_runs/brain_config.json"
        with open(config_file_path, "r") as file:
            config = json.load(file)
        config["overrides"]["run_config"]["v51_dataset_name"] = selected_dataset
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
                    workflow_brain(dataset, dataset_info, MODEL_NAME)
                    run.finish(exit_code=0)
            except Exception as e:
                logging.error(f"An error occurred with model {MODEL_NAME}: {e}")
                logging.error("Stack trace:", exc_info=True)
                if run:
                    run.finish(exit_code=1)
                continue

    if "learn_normality" in SELECTED_WORKFLOW:
        workflows_started += 1
        anomalib_image_models = WORKFLOWS["learn_normality"]["anomalib_image_models"]
        eval_metrics = WORKFLOWS["learn_normality"]["anomalib_eval_metrics"]

        config_file_path = "wandb_runs/anomalib_config.json"
        with open(config_file_path, "r") as file:
            config = json.load(file)
        config["overrides"]["run_config"]["v51_dataset_name"] = selected_dataset
        wandb_project = "Data Engine Anomalib"

        for MODEL_NAME in (pbar := tqdm(anomalib_image_models, desc="Anomalib")):
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
                    eval_metrics=eval_metrics,
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

    if "train_teacher" in SELECTED_WORKFLOW:
        workflows_started += 1
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
                config["overrides"]["run_config"]["v51_dataset_name"] = selected_dataset
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

    if "zero_shot_teacher" in SELECTED_WORKFLOW:
        workflows_started += 1
        workflow_zero_shot_teacher(dataset, dataset_info)

    if "ensemble_exploration" in SELECTED_WORKFLOW:
        workflows_started += 1
        run = None
        try:
            wandb_project = "Data Engine Ensemble Exploration"
            config_file_path = "wandb_runs/ensemble_exploration_config.json"
            with open(config_file_path, "r") as file:
                config = json.load(file)
            config["overrides"]["run_config"]["v51_dataset_name"] = selected_dataset
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

    if workflows_started == 0:
        logging.error(
            str(SELECTED_WORKFLOW)
            + " is not a valid workflow. Check WORKFLOWS in config.py."
        )

    # Launch V51 session
    if args.queue == None:
        dataset.save()
        logging.info(f"Launching Voxel51 session for dataset {dataset.name}:")
        logging.info(dataset)
        fo.pprint(dataset.stats(include_media=True))
        session = fo.launch_app(
            dataset, spaces=spaces, address=V51_ADDRESS, port=V51_PORT, remote=True
        )

    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")


if __name__ == "__main__":
    # Set multiprocessing mode for CUDA multiprocessing
    mp.set_start_method("spawn")

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
