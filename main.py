import datetime
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
from tqdm import tqdm

import wandb
from config.config import (
    SELECTED_DATASET,
    SELECTED_WORKFLOW,
    V51_ADDRESS,
    V51_PORT,
    V51_REMOTE,
    WORKFLOWS,
)
from utils.anomaly_detection_data_preparation import AnomalyDetectionDataPreparation
from utils.data_loader import FiftyOneTorchDatasetCOCO, TorchToHFDatasetCOCO
from utils.dataset_loader import load_dataset
from utils.logging import configure_logging
from utils.mp_distribution import ZeroShotDistributer
from utils.wandb_helper import wandb_close, wandb_init
from workflows.anomaly_detection import Anodec
from workflows.auto_labeling import (
    CustomCoDETRObjectDetection,
    HuggingFaceObjectDetection,
    ZeroShotObjectDetection,
)
from workflows.aws_download import AwsDownloader
from workflows.embedding_selection import EmbeddingSelection
from workflows.ensemble_exploration import EnsembleExploration

wandb_run = None  # Init globally to make sure it is available


def signal_handler(sig, frame):
    logging.error("You pressed Ctrl+C!")
    try:
        wandb.finish()
    except:
        pass
    sys.exit(0)


def workflow_aws_download(wandb_activate=True):
    try:
        dataset = None
        dataset_name = None
        wandb_exit_code = 0

        # Config
        bucket = WORKFLOWS["aws_download"]["bucket"]
        prefix = WORKFLOWS["aws_download"]["prefix"]
        download_path = WORKFLOWS["aws_download"]["download_path"]
        test_run = WORKFLOWS["aws_download"]["test_run"]

        # Logging
        now = datetime.datetime.now()
        datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = f"logs/tensorboard/aws_{datetime_str}"

        dataset_name = f"annarbor_rolling_{datetime_str}"

        # Weights and Biases
        wandb_run = wandb_init(
            run_name=dataset_name,
            project_name="AWS Download",
            dataset_name=dataset_name,
            log_dir=log_dir,
            wandb_activate=wandb_activate,
        )

        # Workflow
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

    except Exception as e:
        logging.error(f"AWS Download and Extraction failed: {e}")
        wandb_exit_code = 1

    finally:
        wandb_close(wandb_run, wandb_exit_code)

    return dataset, dataset_name


def workflow_anomaly_detection(
    dataset, dataset_info, eval_metrics, run_config, wandb_activate=True
):
    try:
        # Weights and Biases
        wandb_exit_code = 0
        wandb_run, log_dir = wandb_init(
            run_name=run_config["model_name"],
            project_name="Selection by Anomaly Detection",
            dataset_name=dataset_info["name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )

        # Workflow
        ano_dec = Anodec(
            dataset=dataset,
            eval_metrics=eval_metrics,
            dataset_info=dataset_info,
            config=run_config,
            tensorboard_output=log_dir,
        )
        if run_config["mode"] == "train":
            ano_dec.train_and_export_model()
            ano_dec.run_inference()
            ano_dec.eval_v51()
        elif run_config["mode"] == "inference":
            ano_dec.run_inference()
        else:
            logging.error(f"Mode {run_config["mode"]} not suported.")
    except Exception as e:
        logging.error(f"Error in Anomaly Detection: {e}")
        wandb_exit_code = 1
    finally:
        wandb_close(wandb_run=wandb_run, exit_code=wandb_exit_code)

    return True


def workflow_embedding_selection(
    dataset, dataset_info, MODEL_NAME, config, wandb_activate=True
):
    try:
        wandb_exit_code = 0
        wandb_run, log_dir = wandb_init(
            run_name=MODEL_NAME,
            project_name="Selection by Embedding",
            dataset_name=dataset_info["name"],
            config=config,
            wandb_activate=wandb_activate,
        )
        embedding_selector = EmbeddingSelection(
            dataset, dataset_info, MODEL_NAME, log_dir
        )
        embedding_selector.compute_embeddings(config["mode"])
        embedding_selector.compute_similarity()

        # Find representative and unique samples as center points for further selections
        thresholds = config["parameters"]
        embedding_selector.compute_representativeness(
            thresholds["compute_representativeness"]
        )
        embedding_selector.compute_unique_images_greedy(
            thresholds["compute_unique_images_greedy"]
        )
        embedding_selector.compute_unique_images_deterministic(
            thresholds["compute_unique_images_deterministic"]
        )

        # Select samples similar to the center points to enlarge the dataset
        embedding_selector.compute_similar_images(
            thresholds["compute_similar_images"], thresholds["neighbour_count"]
        )

    except Exception as e:
        logging.error(f"An error occurred with model {MODEL_NAME}: {e}")
        wandb_exit_code = 1
    finally:
        wandb_close(wandb_run, wandb_exit_code)

    return True


def workflow_auto_labeling(
    dataset, dataset_info, hf_dataset, run_config, wandb_activate=True
):
    try:
        wandb_exit_code = 0
        wandb_run = wandb_init(
            run_name=run_config["model_name"],
            project_name="Auto Labeling Hugging Face",
            dataset_name=dataset_info["name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )

        detector = HuggingFaceObjectDetection(
            dataset=dataset,
            config=run_config,
        )
        if run_config["mode"] == "train":
            logging.info(f"Training model {run_config["model_name"]}")
            detector.train(hf_dataset)
            detector.inference(hf_dataset)
        elif run_config["mode"] == "inference":
            logging.info(f"Running inference for model {run_config["model_name"]}")
            detector.inference(hf_dataset)
        else:
            logging.error(f"Mode {run_config["mode"]} is not supported.")
    except Exception as e:
        logging.error(f"An error occurred with model {run_config["model_name"]}: {e}")
        wandb_exit_code = 1

    finally:
        wandb_close(wandb_run, wandb_exit_code)

    return True


def workflow_auto_labeling_custom_codetr(
    dataset_info, run_config, dataset=None, detector=None, wandb_activate=True
):
    try:
        if detector is None:
            # Export dataset into the format Co-DETR expects
            try:
                if dataset is None:
                    logging.error(
                        f"Dataset is '{dataset}' but needs to be passed for dataset conversion."
                    )
                detector = CustomCoDETRObjectDetection(
                    dataset,
                    dataset_info["name"],
                    dataset_info["v51_splits"],
                    run_config["export_dataset_root"],
                )
                detector.convert_data()
            except Exception as e:
                logging.error(f"Error during CoDETR dataset export: {e}")

        wandb_exit_code = 0
        wandb_run = wandb_init(
            run_name="MODEL_NAME",
            project_name="Selection by Embedding",
            dataset_name=dataset_info["name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )

        detector.update_config_file(
            dataset_name=dataset_info["name"], config_file=run_config["codetr_config"]
        )
        detector.train(
            run_config["codetr_config"],
            run_config["n_gpus"],
            run_config["container_tool"],
        )
    except Exception as e:
        logging.error(f"Error during CoDETR training: {e}")
        wandb_exit_code = 1
    finally:
        wandb_close(wandb_run, wandb_exit_code)

    return True


def workflow_zero_shot_object_detection(dataset, dataset_info):
    # Set multiprocessing mode for CUDA multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
        logging.debug("Successfully set multiprocessing start method to 'spawn'")
    except RuntimeError as e:
        # This is expected if the start method was already set
        logging.debug(f"Multiprocessing start method was already set: {e}")
    except Exception as e:
        # Handle any other unexpected errors
        logging.error(f"Failed to set multiprocessing start method: {e}")

    # Zero-shot object detector models from Huggingface
    # Optimized for parallel multi-GPU inference, also supports single GPU
    config = WORKFLOWS["auto_labeling_zero_shot"]
    dataset_torch = FiftyOneTorchDatasetCOCO(dataset)
    detector = ZeroShotObjectDetection(
        dataset_torch=dataset_torch, dataset_info=dataset_info, config=config
    )

    # Check if model detections are already stored in V51 dataset or on disk
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

    return True


def workflow_ensemble_exploration(
    dataset, dataset_info, run_config, wandb_activate=True
):
    try:
        wandb_exit_code = 0

        wandb_run = wandb_init(
            run_name="Ensemble Exploration",
            project_name="Ensemble Exploration",
            dataset_name=dataset_info["name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )
        explorer = EnsembleExploration(dataset, run_config)
        explorer.ensemble_exploration()
    except Exception as e:
        logging.error(f"An error occured during Ensemble Exploration: {e}")
        wandb_exit_code = 1

    finally:
        wandb_close(wandb_run, wandb_exit_code)

    return True


class WorkflowExecutor:
    def __init__(
        self,
        workflows: List[str],
        selected_dataset: str,
        dataset: fo.Dataset,
        dataset_info: Dict,
    ):
        self.workflows = workflows
        self.selected_dataset = selected_dataset
        self.dataset = dataset
        self.dataset_info = dataset_info

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
            try:
                if workflow == "aws_download":
                    dataset, dataset_name = workflow_aws_download()

                    # Select downloaded dataset for further workflows if configured
                    if dataset is not None:
                        if (
                            WORKFLOWS["aws_download"]["selected_dataset_overwrite"]
                            == True
                        ):

                            dataset_info = {
                                "name": dataset_name,
                                "v51_type": "FiftyOneDataset",
                                "splits": [],
                            }

                            self.dataset = dataset
                            self.dataset_info = dataset_info
                            logging.warning(
                                f"Overwritting selected dataset {self.selected_dataset} with {dataset_name}"
                            )
                            self.selected_dataset = dataset_name

                elif workflow == "embedding_selection":
                    embedding_models = WORKFLOWS["embedding_selection"][
                        "embedding_models"
                    ]

                    for MODEL_NAME in (
                        pbar := tqdm(embedding_models, desc="Selection by Embeddings")
                    ):

                        # Status
                        pbar.set_description(
                            f"Selection by embeddings with model {MODEL_NAME}."
                        )

                        # Config
                        mode = WORKFLOWS["embedding_selection"]["mode"]
                        parameters = WORKFLOWS["embedding_selection"]["parameters"]
                        config = {"mode": mode, "parameters": parameters}

                        # Workflow
                        workflow_embedding_selection(
                            self.dataset,
                            self.dataset_info,
                            MODEL_NAME,
                            config,
                        )

                elif workflow == "anomaly_detection":

                    # Config
                    ano_dec_config = WORKFLOWS["anomaly_detection"]
                    anomalib_image_models = ano_dec_config["anomalib_image_models"]
                    eval_metrics = ano_dec_config["anomalib_eval_metrics"]

                    try:
                        data_preparer = AnomalyDetectionDataPreparation(
                            self.dataset, self.selected_dataset
                        )
                    except Exception as e:
                        logging.error(
                            f"Error during data preparation for Anomaly Detection: {e}"
                        )

                    for MODEL_NAME in (
                        pbar := tqdm(anomalib_image_models, desc="Anomalib")
                    ):
                        # Status
                        pbar.set_description(f"Anomalib model {MODEL_NAME}.")

                        # Config
                        run_config = {
                            "model_name": MODEL_NAME,
                            "image_size": anomalib_image_models[MODEL_NAME].get(
                                "image_size", None
                            ),
                            "batch_size": anomalib_image_models[MODEL_NAME][
                                "batch_size"
                            ],
                            "epochs": ano_dec_config["epochs"],
                            "early_stop_patience": ano_dec_config[
                                "early_stop_patience"
                            ],
                            "data_root": data_preparer.export_root,
                            "mode": ano_dec_config["mode"],
                        }

                        # Workflow
                        workflow_anomaly_detection(
                            data_preparer.dataset_ano_dec,
                            self.dataset_info,
                            eval_metrics,
                            run_config,
                        )

                elif workflow == "auto_labeling":

                    # Config
                    config_autolabel = WORKFLOWS["auto_labeling"]
                    selected_model_source = config_autolabel["model_source"]

                    if selected_model_source == "hf_models_objectdetection":
                        hf_models = config_autolabel["hf_models_objectdetection"]

                        # Dataset Conversion
                        try:
                            logging.info("Converting dataset into Hugging Face format.")
                            pytorch_dataset = FiftyOneTorchDatasetCOCO(self.dataset)
                            pt_to_hf_converter = TorchToHFDatasetCOCO(pytorch_dataset)
                            hf_dataset = pt_to_hf_converter.convert()
                        except Exception as e:
                            logging.error(f"Error during dataset conversion: {e}")

                        # Train models
                        for MODEL_NAME in (
                            pbar := tqdm(hf_models, desc="Auto Labeling Models")
                        ):
                            # Status Update
                            pbar.set_description(
                                f"Processing Hugging Face model {MODEL_NAME}"
                            )

                            # Config
                            config_model = config_autolabel[
                                "hf_models_objectdetection"
                            ][MODEL_NAME]

                            run_config = {
                                "mode": config_autolabel["mode"],
                                "model_name": MODEL_NAME,
                                "v51_dataset_name": self.selected_dataset,
                                "epochs": config_autolabel["epochs"],
                                "early_stop_threshold": config_autolabel[
                                    "early_stop_threshold"
                                ],
                                "early_stop_patience": config_autolabel[
                                    "early_stop_patience"
                                ],
                                "learning_rate": config_autolabel["learning_rate"],
                                "weight_decay": config_autolabel["weight_decay"],
                                "max_grad_norm": config_autolabel["max_grad_norm"],
                                "batch_size": config_model["batch_size"],
                                "image_size": config_model["image_size"],
                                "n_worker_dataloader": config_autolabel[
                                    "n_worker_dataloader"
                                ],
                            }

                            # Workflow
                            workflow_auto_labeling(
                                self.dataset,
                                self.dataset_info,
                                hf_dataset,
                                run_config,
                            )

                    elif selected_model_source == "custom_codetr":

                        # Config
                        config_codetr = WORKFLOWS["auto_labeling"]["custom_codetr"]
                        run_config = {
                            "export_dataset_root": config_codetr["export_dataset_root"],
                            "container_tool": config_codetr["container_tool"],
                            "n_gpus": config_codetr["n_gpus"],
                        }
                        codetr_models = config_codetr["configs"]

                        # Export dataset into the format Co-DETR expects
                        try:
                            detector = CustomCoDETRObjectDetection(
                                dataset,
                                dataset_info["name"],
                                dataset_info["v51_splits"],
                                config_codetr["export_dataset_root"],
                            )
                            detector.convert_data()
                        except Exception as e:
                            logging.error(f"Error during CoDETR dataset export: {e}")

                        for MODEL_NAME in (
                            pbar := tqdm(codetr_models, desc="CoDETR training")
                        ):
                            # Status Update
                            pbar.set_description(f"CoDETR model {MODEL_NAME}")

                            # Update config
                            run_config["codetr_config"] = MODEL_NAME

                            # Workflow
                            workflow_auto_labeling_custom_codetr(
                                self.dataset_info, run_config
                            )
                    else:
                        logging.error(
                            f"Selected model source {selected_model_source} is not supported."
                        )

                elif workflow == "auto_labeling_zero_shot":
                    workflow_zero_shot_object_detection(self.dataset, self.dataset_info)

                elif workflow == "ensemble_exploration":
                    # Config
                    run_config = WORKFLOWS["ensemble_exploration"]

                    # Workflow
                    workflow_ensemble_exploration(
                        self.dataset, self.dataset_info, run_config
                    )

                else:
                    logging.error(
                        f"Workflow {workflow} not found. Check available workflows in config.py."
                    )
                    return False

            except Exception as e:
                logging.error(f"Workflow {workflow}: An error occurred: {e}")
                wandb_close(wandb_run, exit_code=1)

        return True


def main():
    time_start = time.time()
    configure_logging()

    # Signal handler for CTRL + C
    signal.signal(signal.SIGINT, signal_handler)

    # Execute workflows
    dataset, dataset_info = load_dataset(SELECTED_DATASET)
    executor = WorkflowExecutor(
        SELECTED_WORKFLOW, SELECTED_DATASET["name"], dataset, dataset_info
    )
    executor.execute()

    # Launch V51 session
    dataset.reload()
    dataset.save()
    logging.info(f"Launching Voxel51 session for dataset {dataset.name}:")

    # Dataset stats
    logging.info(dataset)
    fo.pprint(dataset.stats(include_media=True))

    # V51 UI launch
    session = fo.launch_app(
        dataset, address=V51_ADDRESS, port=V51_PORT, remote=V51_REMOTE
    )

    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")


if __name__ == "__main__":
    main()
