import argparse
import datetime
import json
import logging
import signal
import sys
import time
import warnings
from typing import Dict, List

IGNORE_FUTURE_WARNINGS = True
if IGNORE_FUTURE_WARNINGS:
    warnings.simplefilter("ignore", category=FutureWarning)
import fiftyone as fo
import torch.multiprocessing as mp
import wandb
from tqdm import tqdm

from config.config import (
    GLOBAL_SEED,
    SELECTED_DATASET,
    SELECTED_WORKFLOW,
    V51_ADDRESS,
    V51_PORT,
    WORKFLOWS,
)
from utils.data_loader import FiftyOneTorchDatasetCOCO

# Called with globals()
from utils.dataset_loader import (
    load_annarbor_rolling,
    load_dataset_info,
    load_fisheye_8k,
    load_mars_multiagent,
    load_mars_multitraversal,
    load_mcity_fisheye_3_months,
    load_mcity_fisheye_2000,
)
from utils.logging import configure_logging
from utils.mp_distribution import ZeroShotDistributer
from utils.wandb_helper import launch_to_queue_terminal
from workflows.ano_dec import Anodec
from workflows.auto_labeling import (
    CustomCoDETRObjectDetection,
    HuggingFaceObjectDetection,
    ZeroShotObjectDetection,
)
from workflows.aws_download import AwsDownloader
from workflows.brain import Brain
from workflows.ensemble_exploration import EnsembleExploration


def signal_handler(sig, frame):
    logging.error("You pressed Ctrl+C!")
    # Perform any cleanup or final actions here
    try:
        wandb.finish()
    except:
        pass
    sys.exit(0)


def workflow_aws_download():
    wandb_run = None
    dataset = None
    try:
        bucket = WORKFLOWS["aws_download"]["bucket"]
        prefix = WORKFLOWS["aws_download"]["prefix"]
        download_path = WORKFLOWS["aws_download"]["download_path"]
        test_run = WORKFLOWS["aws_download"]["test_run"]

        # Logging
        now = datetime.datetime.now()
        datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = f"logs/tensorboard/aws_{datetime_str}"

        dataset_name = f"annarbor_rolling_{datetime_str}"

        wandb.tensorboard.patch(root_logdir=log_dir)
        wandb_run = wandb.init(
            name=dataset_name,
            sync_tensorboard=True,
            group="S3",
            job_type="download",
            project="Data Engine Download",
        )

        aws_downloader = AwsDownloader(
            bucket=bucket,
            prefix=prefix,
            download_path=download_path,
            test_run=test_run,
        )

        sub_folder, files, DOWNLOAD_NUMBER_SUCCESS, DOWNLOAD_SIZE_SUCCESS = (
            aws_downloader.download_files(log_dir=log_dir)
        )
        dataset = aws_downloader.decode_data(
            sub_folder=sub_folder,
            files=files,
            log_dir=log_dir,
            dataset_name=dataset_name,
        )

        wandb_run.finish(exit_code=0)
    except Exception as e:
        logging.error(f"AWS Download and Extraction failed: {e}")
        if wandb_run:
            wandb_run.finish(exit_code=1)

    return dataset, dataset_name


def workflow_anomaly_detection():
    pass


def workflow_brain_selection(dataset, dataset_info, MODEL_NAME):
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


def workflow_auto_labeling():
    pass


def workflow_zero_shot_object_detection(dataset, dataset_info):
    # Zero-shot object detector models from Huggingface
    # Optimized for parallel multi-GPU inference, also supports single GPU
    config = WORKFLOWS["auto_labeling_zero_shot"]
    dataset_torch = FiftyOneTorchDatasetCOCO(dataset)
    detector = ZeroShotObjectDetection(
        dataset_torch=dataset_torch, dataset_info=dataset_info, config=config
    )

    # Check if model detections are already stored in V51 dataset or on disk
    # TODO Think about config. Detector already has it in init before it is processed, but does not use the hf_models. Could be more elegant.
    models_splits_dict = detector.exclude_stored_predictions(
        dataset_v51=dataset, config=config
    )
    if len(models_splits_dict) > 0:
        config["hf_models_zeroshot_objectdetection"] = models_splits_dict
        distributor = ZeroShotDistributer(
            config=config,
            n_samples=len(dataset_torch),
            dataset_info=dataset_info,
            detector=detector,
        )
        distributor.distribute_and_run()
    else:
        logging.info(
            "All zero shot models already have predictions stored in the dataset."
        )


def workflow_ensemble_exploration():
    pass


class WorkflowExecutor:
    def __init__(
        self,
        workflows: List[str],
        selected_dataset: str,
        dataset: fo.Dataset,
        dataset_info: Dict,
        args,
    ):
        self.workflows = workflows
        self.selected_dataset = selected_dataset
        self.dataset = dataset
        self.dataset_info = dataset_info
        self.args = args

    def execute(self) -> bool:
        """Execute workflows in sequential order"""
        if len(self.workflows) == 0:
            logging.error("No workflows selected.")
            return False

        logging.info(f"Selected workflows: {self.workflows}")

        for workflow in self.workflows:
            logging.info(
                f"Running workflow {workflow} for dataset {self.selected_dataset}"
            )
            wandb_run = None
            try:
                if workflow == "aws_download":
                    dataset, dataset_name = workflow_aws_download()

                    # Select downloaded dataset for further workflows if configured
                    if WORKFLOWS["aws_download"]["selected_dataset_overwrite"] == True:

                        dataset_info = {
                            "name": dataset_name,
                            "v51_type": "FiftyOneDataset",
                            "splits": [],
                        }

                        self.dataset = dataset
                        self.dataset_info = dataset_info
                        self.selected_dataset = dataset_name
                        logging.warning(
                            f"Selected dataset overwritten to {dataset_name}"
                        )

                elif workflow == "brain_selection":
                    embedding_models = WORKFLOWS["brain_selection"]["embedding_models"]
                    config_file_path = "wandb_runs/brain_config.json"
                    with open(config_file_path, "r") as file:
                        config = json.load(file)
                    config["overrides"]["run_config"][
                        "v51_dataset_name"
                    ] = self.selected_dataset

                    for MODEL_NAME in (pbar := tqdm(embedding_models, desc="Brain")):
                        wandb_run = None
                        try:
                            pbar.set_description(
                                "Generating/Loading embeddings with " + MODEL_NAME
                            )
                            config["overrides"]["run_config"]["model_name"] = MODEL_NAME

                            if args.queue == None:
                                wandb_run = wandb.init(
                                    name=MODEL_NAME,
                                    allow_val_change=True,
                                    sync_tensorboard=True,
                                    group="Brain",
                                    job_type="eval",
                                    project="Data Engine Brain",
                                    config=config,
                                )
                                wandb_config = wandb.config["overrides"]["run_config"]
                                wandb_run.tags += (
                                    wandb_config["v51_dataset_name"],
                                    wandb_config["v51_dataset_name"],
                                    "local",
                                )
                                workflow_brain_selection(
                                    self.dataset, self.dataset_info, MODEL_NAME
                                )
                                wandb_run.finish(exit_code=0)
                        except Exception as e:
                            logging.error(
                                f"An error occurred with model {MODEL_NAME}: {e}"
                            )
                            if wandb_run:
                                wandb_run.finish(exit_code=1)
                            continue

                elif workflow == "anomaly_detection":
                    anomalib_image_models = WORKFLOWS["anomaly_detection"][
                        "anomalib_image_models"
                    ]
                    eval_metrics = WORKFLOWS["anomaly_detection"][
                        "anomalib_eval_metrics"
                    ]

                    config_file_path = "wandb_runs/anomalib_config.json"
                    with open(config_file_path, "r") as file:
                        config = json.load(file)
                    config["overrides"]["run_config"][
                        "v51_dataset_name"
                    ] = self.selected_dataset
                    wandb_project = "Data Engine Anomalib"

                    for MODEL_NAME in (
                        pbar := tqdm(anomalib_image_models, desc="Anomalib")
                    ):
                        pbar.set_description(
                            "Training/Loading Anomalib model " + MODEL_NAME
                        )
                        config["overrides"]["run_config"]["model_name"] = MODEL_NAME

                        # Local execution
                        if self.args.queue == None:
                            wandb_run = wandb.init(
                                allow_val_change=True,
                                sync_tensorboard=True,
                                project=wandb_project,
                                group="Anomalib",
                                job_type="train",
                                config=config,
                            )
                            config = wandb.config["overrides"]["run_config"]
                            wandb_run.tags += (
                                config["v51_dataset_name"],
                                config["model_name"],
                                "local",
                            )
                            ano_dec = Anodec(
                                dataset=self.dataset,
                                eval_metrics=eval_metrics,
                                dataset_info=self.dataset_info,
                                config=config,
                            )
                            ano_dec.train_and_export_model()
                            ano_dec.run_inference()
                            ano_dec.eval_v51()

                        # Queue execution
                        elif self.args.queue != None:
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
                                queue=self.args.queue,
                            )

                        wandb_run.finish(exit_code=0)

                elif workflow == "auto_labeling":
                    logging.info("Model training with Hugging Face")
                    mode = WORKFLOWS["auto_labeling"]["mode"]
                    selected_model_source = WORKFLOWS["auto_labeling"]["model_source"]

                    if selected_model_source == "hf_models_objectdetection":
                        hf_models = WORKFLOWS["auto_labeling"][
                            "hf_models_objectdetection"
                        ]
                        wandb_project = "Data Engine Auto Labeling"
                        config_file_path = "wandb_runs/auto_label_config.json"
                        with open(config_file_path, "r") as file:
                            config = json.load(file)

                        # Train models
                        for MODEL_NAME in (
                            pbar := tqdm(hf_models, desc="Auto Labeling Models")
                        ):
                            wandb_run = None
                            try:
                                pbar.set_description("Training model " + MODEL_NAME)
                                config["overrides"]["run_config"][
                                    "model_name"
                                ] = MODEL_NAME
                                config["overrides"]["run_config"][
                                    "v51_dataset_name"
                                ] = self.selected_dataset
                                if self.args.queue == None:

                                    wandb_run = wandb.init(
                                        name=MODEL_NAME,
                                        allow_val_change=True,
                                        sync_tensorboard=True,
                                        group="Auto Labeling HF",
                                        job_type="train",
                                        config=config,
                                        project=wandb_project,
                                    )
                                    wandb_config = wandb.config["overrides"][
                                        "run_config"
                                    ]

                                    wandb_run.tags += (
                                        wandb_config["v51_dataset_name"],
                                        "local",
                                    )
                                    detector = HuggingFaceObjectDetection(
                                        dataset=self.dataset,
                                        config=wandb_config,
                                    )

                                    if mode == "train":
                                        detector.train()
                                    elif mode == "inference":
                                        detector.inference()
                                    else:
                                        logging.error(f"Mode {mode} is not supported.")
                                    wandb_run.finish(exit_code=0)

                                elif self.args.queue != None:
                                    # Update config file
                                    with open(config_file_path, "w") as file:
                                        json.dump(config, file, indent=4)

                                    wandb_entry_point = config["overrides"][
                                        "entry_point"
                                    ]
                                    # Add job to queue
                                    launch_to_queue_terminal(
                                        name=MODEL_NAME,
                                        project=wandb_project,
                                        config_file=config_file_path,
                                        entry_point=wandb_entry_point,
                                        queue=self.args.queue,
                                    )
                            except Exception as e:
                                logging.error(
                                    f"An error occurred with model {MODEL_NAME}: {e}"
                                )
                                if wandb_run:
                                    wandb_run.finish(exit_code=1)
                                continue
                    elif selected_model_source == "custom_codetr":
                        # Export dataset into the format Co-DETR expects
                        export_dir = WORKFLOWS["auto_labeling"]["custom_codetr"][
                            "export_dataset_root"
                        ]
                        container_tool = WORKFLOWS["auto_labeling"]["custom_codetr"][
                            "container_tool"
                        ]
                        param_n_gpus = WORKFLOWS["auto_labeling"]["custom_codetr"][
                            "n_gpus"
                        ]
                        detector = CustomCoDETRObjectDetection(
                            self.dataset,
                            self.selected_dataset,
                            self.dataset_info["v51_splits"],
                            export_dir,
                        )
                        detector.convert_data()

                        codetr_configs = WORKFLOWS["auto_labeling"]["custom_codetr"][
                            "configs"
                        ]
                        for config in codetr_configs:
                            detector.update_config_file(
                                dataset_name=self.selected_dataset, config_file=config
                            )
                            detector.train(config, param_n_gpus, container_tool)

                    else:
                        logging.error(
                            f"Selected model source {selected_model_source} is not supported."
                        )

                elif workflow == "auto_labeling_zero_shot":
                    workflow_zero_shot_object_detection(self.dataset, self.dataset_info)

                elif workflow == "ensemble_exploration":
                    wandb_project = "Data Engine Ensemble Exploration"
                    config_file_path = "wandb_runs/ensemble_exploration_config.json"
                    with open(config_file_path, "r") as file:
                        config = json.load(file)
                    config["overrides"]["run_config"][
                        "v51_dataset_name"
                    ] = self.selected_dataset
                    if self.args.queue == None:
                        wandb_run = wandb.init(
                            name="ensemble-exploration",
                            allow_val_change=True,
                            sync_tensorboard=True,
                            group="Exploration",
                            job_type="eval",
                            config=config,
                            project=wandb_project,
                        )
                        wandb_config = wandb.config["overrides"]["run_config"]
                        wandb_run.tags += (
                            wandb_config["v51_dataset_name"],
                            "local",
                            "ensemble-exploration",
                        )
                        explorer = EnsembleExploration(self.dataset, wandb_config)
                        explorer.ensemble_exploration()
                        wandb_run.finish(exit_code=0)

                else:
                    logging.error(
                        f"Workflow {workflow} not found. Check available workflows in config.py."
                    )
                    return False

            except Exception as e:
                logging.error(f"Workflow {workflow}: An error occurred: {e}")
                if wandb_run:
                    wandb_run.finish(exit_code=1)

        return True


def load_dataset(selected_dataset: str) -> fo.Dataset:
    dataset_info = load_dataset_info(selected_dataset["name"])

    if dataset_info:
        loader_function = dataset_info.get("loader_fct")
        dataset = globals()[loader_function](dataset_info)
        n_samples_original = len(dataset)
        n_samples_requested = selected_dataset["n_samples"]

        if (
            n_samples_requested is not None
            and n_samples_requested <= n_samples_original
        ):
            dataset_reduced_view = dataset.take(
                n_samples_requested, seed=GLOBAL_SEED
            )  # Returns a view rather than a full dataset, might lead to issues for some operations that require a fiftyone dataset
            logging.info(
                f"Dataset size was reduced from {n_samples_original} to {n_samples_requested} samples."
            )
            return dataset_reduced_view, dataset_info
    else:
        logging.error(
            str(selected_dataset["name"])
            + " is not a valid dataset name. Check supported datasets in datasets.yaml."
        )

    return dataset, dataset_info


def main(args):
    time_start = time.time()
    configure_logging()

    if args.tags:
        wandb_tags = args.tags.split(",")
        # TODO Implement usage of tags
    if not args.queue:
        # Signal handler for CTRL + C
        signal.signal(signal.SIGINT, signal_handler)

    # Execute workflows
    dataset, dataset_info = load_dataset(SELECTED_DATASET)
    executor = WorkflowExecutor(
        SELECTED_WORKFLOW, SELECTED_DATASET["name"], dataset, dataset_info, args
    )
    executor.execute()

    # Launch V51 session
    if not args.queue:
        dataset.reload()
        dataset.save()
        logging.info(f"Launching Voxel51 session for dataset {dataset.name}:")
        logging.info(dataset)
        fo.pprint(dataset.stats(include_media=True))
        session = fo.launch_app(
            dataset, address=V51_ADDRESS, port=V51_PORT, remote=True
        )

    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")


if __name__ == "__main__":
    # Set multiprocessing mode for CUDA multiprocessing
    try:
        mp.set_start_method("spawn")
    except:
        pass
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
