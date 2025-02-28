import datetime
import logging
import signal
import sys
import time
import warnings
from typing import Dict, List

import torch

IGNORE_FUTURE_WARNINGS = True
if IGNORE_FUTURE_WARNINGS:
    warnings.simplefilter("ignore", category=FutureWarning)

import gc

import fiftyone as fo
import torch.multiprocessing as mp
from tqdm import tqdm

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
from utils.sidebar_groups import arrange_fields_in_groups
from utils.wandb_helper import wandb_close, wandb_init
from workflows.anomaly_detection import Anodec
from workflows.auto_labeling import (
    CustomCoDETRObjectDetection,
    HuggingFaceObjectDetection,
    UltralyticsObjectDetection,
    ZeroShotObjectDetection,
)
from workflows.aws_download import AwsDownloader
from workflows.embedding_selection import EmbeddingSelection
from workflows.ensemble_selection import EnsembleSelection
from workflows.teacher_mask import MaskTeacher

wandb_run = None  # Init globally to make sure it is available


def signal_handler(sig, frame):
    """Handle Ctrl+C signal by cleaning up resources and exiting."""
    logging.error("You pressed Ctrl+C!")
    try:
        wandb_close(exit_code=1)
        cleanup_memory()
    except:
        pass
    sys.exit(0)


def workflow_aws_download(parameters, wandb_activate=True):
    """Download and process data from AWS S3 bucket."""
    dataset = None
    dataset_name = None
    wandb_exit_code = 0
    files_to_be_downloaded = 0
    try:
        # Config
        bucket = parameters["bucket"]
        prefix = parameters["prefix"]
        download_path = parameters["download_path"]
        test_run = parameters["test_run"]

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

        (
            sub_folder,
            files,
            files_to_be_downloaded,
            DOWNLOAD_NUMBER_SUCCESS,
            DOWNLOAD_SIZE_SUCCESS,
        ) = aws_downloader.download_files(log_dir=log_dir)

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
        wandb_close(wandb_exit_code)

    return dataset, dataset_name, files_to_be_downloaded


def workflow_anomaly_detection(
    dataset_normal,
    dataset_ano_dec,
    dataset_info,
    eval_metrics,
    run_config,
    wandb_activate=True,
):
    """Run anomaly detection workflow using specified models and configurations."""
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

        SUPPORTED_MODES = ["train", "inference"]
        # Check if all selected modes are supported
        for mode in run_config["mode"]:
            if mode not in SUPPORTED_MODES:
                logging.error(f"Selected mode {mode} is not supported.")
        if SUPPORTED_MODES[0] in run_config["mode"]:
            ano_dec = Anodec(
                dataset=dataset_ano_dec,
                eval_metrics=eval_metrics,
                dataset_info=dataset_info,
                config=run_config,
                tensorboard_output=log_dir,
            )
            ano_dec.train_and_export_model()
            ano_dec.run_inference(mode=SUPPORTED_MODES[0])
            ano_dec.eval_v51()
        if SUPPORTED_MODES[1] in run_config["mode"]:
            ano_dec = Anodec(
                dataset=dataset_normal,
                eval_metrics=eval_metrics,
                dataset_info=dataset_info,
                config=run_config,
                tensorboard_output=log_dir,
            )
            ano_dec.run_inference(mode=SUPPORTED_MODES[1])

    except Exception as e:
        logging.error(
            f"Error in Anomaly Detection for model {run_config['model_name']}: {e}"
        )
        wandb_exit_code = 1
    finally:
        wandb_close(exit_code=wandb_exit_code)

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

        if embedding_selector.model_already_used == False:

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
        else:
            logging.warning(
                f"Skipping model {embedding_selector.model_name_key}. It was already used for sample selection."
            )

    except Exception as e:
        logging.error(f"An error occurred with model {MODEL_NAME}: {e}")
        wandb_exit_code = 1
    finally:
        wandb_close(wandb_exit_code)

    return True


def workflow_auto_labeling_ultralytics(dataset, run_config, wandb_activate=True):
    try:
        wandb_exit_code = 0
        wandb_run = wandb_init(
            run_name=run_config["model_name"],
            project_name="Auto Labeling Ultralytics",
            dataset_name=run_config["v51_dataset_name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )

        detector = UltralyticsObjectDetection(dataset=dataset, config=run_config)

        # Check if all selected modes are supported
        SUPPORTED_MODES = ["train", "inference"]
        for mode in run_config["mode"]:
            if mode not in SUPPORTED_MODES:
                logging.error(f"Selected mode {mode} is not supported.")

        if SUPPORTED_MODES[0] in run_config["mode"]:
            logging.info(f"Training model {run_config['model_name']}")
            detector.train()
        if SUPPORTED_MODES[1] in run_config["mode"]:
            logging.info(f"Running inference for model {run_config['model_name']}")
            detector.inference()

    except Exception as e:
        logging.error(f"An error occurred with model {run_config['model_name']}: {e}")
        wandb_exit_code = 1

    finally:
        wandb_close(wandb_exit_code)

    return True


def workflow_auto_labeling_hf(dataset, hf_dataset, run_config, wandb_activate=True):
    try:
        wandb_exit_code = 0
        wandb_run = wandb_init(
            run_name=run_config["model_name"],
            project_name="Auto Labeling Hugging Face",
            dataset_name=run_config["v51_dataset_name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )

        detector = HuggingFaceObjectDetection(
            dataset=dataset,
            config=run_config,
        )
        SUPPORTED_MODES = ["train", "inference"]

        # Check if all selected modes are supported
        for mode in run_config["mode"]:
            if mode not in SUPPORTED_MODES:
                logging.error(f"Selected mode {mode} is not supported.")
        if SUPPORTED_MODES[0] in run_config["mode"]:
            logging.info(f"Training model {run_config['model_name']}")
            detector.train(hf_dataset)
        if SUPPORTED_MODES[1] in run_config["mode"]:
            logging.info(f"Running inference for model {run_config['model_name']}")
            detector.inference(inference_settings=run_config["inference_settings"])

    except Exception as e:
        logging.error(f"An error occurred with model {run_config['model_name']}: {e}")
        wandb_exit_code = 1

    finally:
        wandb_close(wandb_exit_code)

    return True


def workflow_auto_labeling_custom_codetr(
    dataset, dataset_info, run_config, wandb_activate=True
):

    try:
        wandb_exit_code = 0
        wandb_run = wandb_init(
            run_name=run_config["config"],
            project_name="Co-DETR Auto Labeling",
            dataset_name=dataset_info["name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )

        mode = run_config["mode"]

        detector = CustomCoDETRObjectDetection(dataset, dataset_info, run_config)
        if "train" in mode:
            detector.convert_data()
            detector.update_config_file(
                dataset_name=dataset_info["name"],
                config_file=run_config["config"],
                max_epochs=run_config["epochs"],
            )
            detector.train(
                run_config["config"], run_config["n_gpus"], run_config["container_tool"]
            )
        if "inference" in mode:
            detector.run_inference(
                dataset,
                run_config["config"],
                run_config["n_gpus"],
                run_config["container_tool"],
                run_config["inference_settings"],
            )
    except Exception as e:
        logging.error(f"Error during CoDETR training: {e}")
        wandb_exit_code = 1
    finally:
        wandb_close(wandb_exit_code)

    return True


def workflow_zero_shot_object_detection(dataset, dataset_info, config):
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

    # To make new fields available to follow-up processes
    dataset.reload()
    dataset.save()

    return True


def workflow_mask_teacher(dataset, dataset_info):
    try:
        DEPTH_ESTIMATION_MODELS = WORKFLOWS["mask_teacher"]["depth_estimation"]
        SEMANTIC_SEGMENTATION_MODELS = WORKFLOWS["mask_teacher"][
            "semantic_segmentation"
        ]

        for model_name in DEPTH_ESTIMATION_MODELS:
            teacher = MaskTeacher(
                dataset=dataset,
                dataset_info=dataset_info,
                model_name=model_name,
                task_type="depth_estimation",
                model_config=WORKFLOWS["mask_teacher"]["depth_estimation"][model_name],
            )
            teacher.run_inference()

        for model_name in SEMANTIC_SEGMENTATION_MODELS:
            teacher = MaskTeacher(
                dataset=dataset,
                dataset_info=dataset_info,
                model_name=model_name,
                task_type="semantic_segmentation",
                model_config=WORKFLOWS["mask_teacher"]["semantic_segmentation"][
                    model_name
                ],
            )
            teacher.run_inference()

        return dataset

    except Exception as e:
        logging.error(f"Mask Teacher failed: {e}")
        raise


def workflow_ensemble_selection(dataset, dataset_info, run_config, wandb_activate=True):
    try:
        wandb_exit_code = 0

        wandb_run = wandb_init(
            run_name="Selection by Ensemble",
            project_name="Ensemble Selection",
            dataset_name=dataset_info["name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )
        ensemble_selecter = EnsembleSelection(dataset, run_config)
        ensemble_selecter.ensemble_selection()
    except Exception as e:
        logging.error(f"An error occured during Ensemble Selection: {e}")
        wandb_exit_code = 1

    finally:
        wandb_close(wandb_exit_code)

    return True


def cleanup_memory(do_extensive_cleanup=False):
    """Clean up memory after workflow execution. 'do_extensive_cleanup' recommended for multiple training sessions in a row."""
    logging.info("Starting memory cleanup")
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

    if do_extensive_cleanup:

        # Clear any leftover tensors
        n_deleted_torch_objects = 0
        for obj in tqdm(
            gc.get_objects(), desc="Deleting objects from Python Garbage Collector"
        ):
            try:
                if torch.is_tensor(obj):
                    del obj
                    n_deleted_torch_objects += 1
            except:
                pass

        logging.info(f"Deleted {n_deleted_torch_objects} torch objects")

        # Final garbage collection
        gc.collect()


class WorkflowExecutor:
    """Orchestrates the execution of multiple data processing workflows in sequence."""

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
        """Execute all configured workflows in sequence and handle errors."""
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
                    parameter_group = "mcity"
                    parameters = WORKFLOWS["aws_download"].get(parameter_group, None)
                    if parameter_group == "mcity":
                        dataset, dataset_name, _ = workflow_aws_download(parameters)
                    else:
                        logging.error(
                            f"The parameter group {parameter_group} is not supported. As AWS are highly specific, please provide a separate set of parameters and a workflow."
                        )

                    # Select downloaded dataset for further workflows if configured
                    if dataset is not None:
                        if parameters["selected_dataset_overwrite"] == True:

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

                    dataset_ano_dec = None
                    data_root = None
                    if "train" in ano_dec_config["mode"]:
                        try:
                            data_preparer = AnomalyDetectionDataPreparation(
                                self.dataset, self.selected_dataset
                            )
                            dataset_ano_dec = data_preparer.dataset_ano_dec
                            data_root = data_preparer.export_root
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
                            "batch_size": anomalib_image_models[MODEL_NAME].get(
                                "batch_size", 1
                            ),
                            "epochs": ano_dec_config["epochs"],
                            "early_stop_patience": ano_dec_config[
                                "early_stop_patience"
                            ],
                            "data_root": data_root,
                            "mode": ano_dec_config["mode"],
                        }

                        # Workflow
                        workflow_anomaly_detection(
                            self.dataset,
                            dataset_ano_dec,
                            self.dataset_info,
                            eval_metrics,
                            run_config,
                        )

                elif workflow == "auto_labeling":

                    # Config
                    SUPPORTED_MODEL_SOURCES = [
                        "hf_models_objectdetection",
                        "custom_codetr",
                        "ultralytics",
                    ]

                    # Common parameters between models
                    config_autolabel = WORKFLOWS["auto_labeling"]
                    mode = config_autolabel["mode"]
                    epochs = config_autolabel["epochs"]
                    selected_model_source = config_autolabel["model_source"]

                    # Check if all selected modes are supported
                    for model_source in selected_model_source:
                        if model_source not in SUPPORTED_MODEL_SOURCES:
                            logging.error(
                                f"Selected model source {model_source} is not supported."
                            )

                    if SUPPORTED_MODEL_SOURCES[0] in selected_model_source:
                        # Hugging Face Models
                        hf_models = config_autolabel["hf_models_objectdetection"]

                        # Dataset Conversion
                        try:
                            logging.info("Converting dataset into Hugging Face format.")
                            pytorch_dataset = FiftyOneTorchDatasetCOCO(self.dataset)
                            pt_to_hf_converter = TorchToHFDatasetCOCO(pytorch_dataset)
                            hf_dataset = pt_to_hf_converter.convert()
                        except Exception as e:
                            logging.error(f"Error during dataset conversion: {e}")

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
                                "mode": mode,
                                "model_name": MODEL_NAME,
                                "v51_dataset_name": self.selected_dataset,
                                "epochs": epochs,
                                "early_stop_threshold": config_autolabel[
                                    "early_stop_threshold"
                                ],
                                "early_stop_patience": config_autolabel[
                                    "early_stop_patience"
                                ],
                                "learning_rate": config_autolabel["learning_rate"],
                                "weight_decay": config_autolabel["weight_decay"],
                                "max_grad_norm": config_autolabel["max_grad_norm"],
                                "batch_size": config_model.get("batch_size", 1),
                                "image_size": config_model.get("image_size", None),
                                "n_worker_dataloader": config_autolabel[
                                    "n_worker_dataloader"
                                ],
                                "inference_settings": config_autolabel[
                                    "inference_settings"
                                ],
                            }

                            # Workflow
                            workflow_auto_labeling_hf(
                                self.dataset,
                                hf_dataset,
                                run_config,
                            )

                    if SUPPORTED_MODEL_SOURCES[1] in selected_model_source:
                        # Custom Co-DETR
                        config_codetr = config_autolabel["custom_codetr"]
                        run_config = {
                            "export_dataset_root": config_codetr["export_dataset_root"],
                            "container_tool": config_codetr["container_tool"],
                            "n_gpus": config_codetr["n_gpus"],
                            "mode": config_autolabel["mode"],
                            "epochs": config_autolabel["epochs"],
                            "inference_settings": config_autolabel[
                                "inference_settings"
                            ],
                            "config": None,
                        }
                        codetr_configs = config_codetr["configs"]

                        for config in tqdm(
                            codetr_configs, desc="Processing Co-DETR configurations"
                        ):
                            pbar.set_description(f"Co-DETR model {MODEL_NAME}")
                            run_config["config"] = config
                            workflow_auto_labeling_custom_codetr(
                                self.dataset, self.dataset_info, run_config
                            )
                    if SUPPORTED_MODEL_SOURCES[2] in selected_model_source:
                        # Ultralytics Models
                        config_ultralytics = config_autolabel["ultralytics"]
                        models_ultralytics = config_ultralytics["models"]
                        export_dataset_root = config_ultralytics["export_dataset_root"]

                        # Export data into necessary format
                        if "train" in mode:
                            try:
                                UltralyticsObjectDetection.export_data(
                                    self.dataset,
                                    self.dataset_info,
                                    export_dataset_root,
                                )
                            except Exception as e:
                                logging.error(
                                    f"Error during Ultralytics dataset export: {e}"
                                )

                        for model_name in (
                            pbar := tqdm(
                                models_ultralytics, desc="Ultralytics training"
                            )
                        ):
                            pbar.set_description(f"Ultralytics model {model_name}")
                            run_config = {
                                "mode": mode,
                                "model_name": model_name,
                                "v51_dataset_name": self.dataset_info["name"],
                                "epochs": epochs,
                                "patience": config_autolabel["early_stop_threshold"],
                                "batch_size": models_ultralytics[model_name][
                                    "batch_size"
                                ],
                                "img_size": models_ultralytics[model_name]["img_size"],
                                "export_dataset_root": export_dataset_root,
                                "inference_settings": config_autolabel[
                                    "inference_settings"
                                ],
                            }

                            workflow_auto_labeling_ultralytics(self.dataset, run_config)

                elif workflow == "auto_labeling_zero_shot":
                    config = WORKFLOWS["auto_labeling_zero_shot"]
                    workflow_zero_shot_object_detection(
                        self.dataset, self.dataset_info, config
                    )

                elif workflow == "ensemble_selection":
                    # Config
                    run_config = WORKFLOWS["ensemble_selection"]

                    # Workflow
                    workflow_ensemble_selection(
                        self.dataset, self.dataset_info, run_config
                    )

                elif workflow == "mask_teacher":
                    workflow_mask_teacher(self.dataset, self.dataset_info)

                else:
                    logging.error(
                        f"Workflow {workflow} not found. Check available workflows in config.py."
                    )
                    return False

                cleanup_memory()  # Clean after each workflow
                logging.info(f"Completed workflow {workflow} and cleaned up memory")

            except Exception as e:
                logging.error(f"Workflow {workflow}: An error occurred: {e}")
                wandb_close(exit_code=1)
                cleanup_memory()  # Clean up even after failure

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
    arrange_fields_in_groups(dataset)
    logging.info(f"Launching Voxel51 session for dataset {dataset_info['name']}.")

    # Dataset stats
    logging.debug(dataset)
    logging.debug(dataset.stats(include_media=True))

    # V51 UI launch
    session = fo.launch_app(
        dataset, address=V51_ADDRESS, port=V51_PORT, remote=V51_REMOTE
    )

    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")


if __name__ == "__main__":
    cleanup_memory()  # Clean before run
    main()
