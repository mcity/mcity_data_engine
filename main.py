import datetime
import logging
import os
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
from tqdm import tqdm

from config.config import (
    MSIGHT_CONFIG,
    SELECTED_DATASET,
    SELECTED_WORKFLOW,
    V51_ADDRESS,
    V51_PORT,
    V51_REMOTE,
    WORKFLOWS,
)
from utils.logging import configure_logging
from utils.sidebar_groups import arrange_fields_in_groups
from utils.wandb_helper import wandb_close, wandb_init
# from workflows.vitpose_keypoint import (
#     ViTPoseKeypointDetection,
#     RoIKeypointDetection,
#     download_vitpose_weights,
# )
from workflows.data_ingest import run_data_ingest

# NOTE: model-framework-specific imports (torch/transformers/ultralytics/anomalib/
# CoDETR/RFDETR, etc.) are done lazily inside the workflow functions/branches that
# actually use them below -- keeps `data_ingest`-only runs from paying the import
# cost of every other workflow's ML stack on each fresh `python main.py` subprocess.

wandb_run = None  # Init globally to make sure it is available

def signal_handler(sig, frame):
    """Handle Ctrl+C signal by cleaning up resources and exiting."""
    logging.error("You pressed Ctrl+C!")
    try:
        wandb_close(exit_code=1)
        #cleanup_memory()
    except:
        pass
    sys.exit(0)


def workflow_aws_download(parameters, wandb_activate=True):
    """Download and process data from AWS S3 bucket."""
    from workflows.aws_download import AwsDownloader

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
    from workflows.anomaly_detection import Anodec

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
    """Compute embeddings and find representative and rare images for dataset selection."""
    from workflows.embedding_selection import EmbeddingSelection

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
    """Auto-labeling workflow using Ultralytics models with optional training and inference."""
    from workflows.auto_labeling import UltralyticsObjectDetection

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
    """Auto-labeling using Hugging Face models on a dataset, including training and/or inference based on the provided configuration."""
    from workflows.auto_labeling import HuggingFaceObjectDetection

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
    """Auto labeling workflow using Co-DETR model supporting training and inference modes."""
    from workflows.auto_labeling import CustomCoDETRObjectDetection

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
        detector.convert_data()
        if "train" in mode:
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
    """Run zero-shot object detection on a dataset using models from Huggingface, supporting both single and multi-GPU inference."""
    import torch.multiprocessing as mp
    from utils.data_loader import FiftyOneTorchDatasetCOCO
    from utils.mp_distribution import ZeroShotDistributer
    from workflows.auto_labeling import ZeroShotObjectDetection

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


def workflow_auto_label_mask(dataset, dataset_info, config):
    from workflows.auto_label_mask import AutoLabelMask

    try:
        depth_config = config["depth_estimation"]
        seg_config = config["semantic_segmentation"]

        for architecture_name, architecture_info in depth_config.items():
            auto_labeler = AutoLabelMask(
                dataset=dataset,
                dataset_info=dataset_info,
                model_name=architecture_name,
                task_type="depth_estimation",
                model_config=architecture_info,
            )
            auto_labeler.run_inference()

        for architecture_name, architecture_info in seg_config.items():
            auto_labeler = AutoLabelMask(
                dataset=dataset,
                dataset_info=dataset_info,
                model_name=architecture_name,
                task_type="semantic_segmentation",
                model_config=architecture_info,
            )
            auto_labeler.run_inference()

    except Exception as e:
        logging.error(f"Auto-labeling mask workflow failed: {e}")
        raise


def workflow_ensemble_selection(dataset, dataset_info, run_config, wandb_activate=True):
    """Runs ensemble selection workflow on given dataset using provided configuration."""
    from workflows.ensemble_selection import EnsembleSelection

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


def workflow_class_mapping(
    dataset,
    dataset_info,
    run_config,
    wandb_activate=True,
    test_dataset_source=None,
    test_dataset_target=None,
):
    """Runs class mapping workflow to align labels between the source dataset and target dataset."""
    from workflows.class_mapping import ClassMapper

    try:
        wandb_exit_code = 0
        # Initialize a wandb run for class mapping
        wandb_run = wandb_init(
            run_name="class_mapping",
            project_name="Class Mapping",
            dataset_name=dataset_info["name"],
            config=run_config,
            wandb_activate=wandb_activate,
        )
        class_mapping_models = run_config["hf_models_zeroshot_classification"]

        for model_name in (
            pbar := tqdm(class_mapping_models, desc="Processing Class Mapping")
        ):
            pbar.set_description(f"Zero Shot Classification model {model_name}")
            mapper = ClassMapper(dataset, model_name, run_config)
            try:
                stats = mapper.run_mapping(test_dataset_source, test_dataset_target)
                # Display statistics only if mapping was successful
                source_class_counts = stats["total_processed"]
                logging.info("\nClassification Results for Source Dataset:")
                for source_class, count in stats["source_class_counts"].items():
                    percentage = (
                        (count / source_class_counts) * 100
                        if source_class_counts > 0
                        else 0
                    )
                    logging.info(
                        f"{source_class}: {count} samples processed ({percentage:.1f}%)"
                    )

                # Display statistics for tags added to Target Dataset
                logging.info("\nTag Addition Results (Target Dataset Tags):")
                logging.info(f"Total new tags added: {stats['changes_made']}")
                for target_class, tag_count in stats["tags_added_per_category"].items():
                    logging.info(f"{target_class} tags added: {tag_count}")

            except Exception as e:
                logging.error(f"Error during mapping with model {model_name}: {e}")

    except Exception as e:
        logging.error(f"Error in class_mapping workflow: {e}")
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


def _run_msight_localization(dataset: fo.Dataset) -> None:
    """Run MSight localization when MSIGHT_CONFIG['run_localization'] is True.

    Installs MSight/requirements.txt if msight_base is not yet importable,
    then localizes the configured detection_field and writes lat/lon detections
    and keypoints back to the dataset.
    """
    import subprocess
    import sys
    import importlib

    detection_field = MSIGHT_CONFIG.get("detection_field")
    loc_maps_path = MSIGHT_CONFIG.get("loc_maps")
    intrinsics_path = MSIGHT_CONFIG.get("intrinsics")

    if not detection_field or not loc_maps_path or not intrinsics_path:
        logging.error(
            "MSIGHT_CONFIG is missing one or more required keys: "
            "'detection_field', 'loc_maps', 'intrinsics'. Skipping localization."
        )
        return

    # The dataset holds no detections of this field, for example because the
    # workflow ran a different model than the one MSIGHT_CONFIG points to.
    # This is not an error: skip the step before the dependencies install.
    if detection_field not in dataset.get_field_schema():
        logging.warning(
            f"MSight localization skipped: dataset '{dataset.name}' has no field "
            f"'{detection_field}'. Set MSIGHT_CONFIG['detection_field'] to a "
            f"field of this dataset, or set 'run_localization' to False."
        )
        return

    # Auto-install dependencies if not present
    try:
        importlib.import_module("msight_base")
    except ModuleNotFoundError:
        logging.info("msight_base not found — running MSight/install.sh")
        install_script = os.path.join(os.path.dirname(__file__), "MSight", "install.sh")
        result = subprocess.run(
            ["bash", install_script], capture_output=True, text=True
        )
        if result.returncode != 0:
            logging.error(f"MSight install failed:\n{result.stderr}")
            return
        logging.info(result.stdout)
        importlib.invalidate_caches()

    try:
        from MSight.localize_dataset import run_localization
        from MSight.utils.load_locamaps import (
            build_pixel_localizer,
            load_intrinsics,
            load_locmaps,
        )
    except ImportError as exc:
        logging.error(f"Could not import MSight localization modules: {exc}")
        return

    msight_field = f"msight_{detection_field}"

    logging.info(f"MSight localization: '{detection_field}' -> '{msight_field}'")

    try:
        intrinsics = load_intrinsics(intrinsics_path)
        x0, y0 = intrinsics["x0"], intrinsics["y0"]
        logging.info(f"Camera intrinsics: f={intrinsics['f']}, x0={x0}, y0={y0}")

        lat_map, lon_map = load_locmaps(loc_maps_path)
        localizer = build_pixel_localizer(lat_map, lon_map)

        run_localization(
            dataset=dataset,
            detection_field=detection_field,
            msight_field=msight_field,
            localizer=localizer,
            x0=x0,
            y0=y0,
        )

        dataset.save()
        logging.info("MSight localization complete.")

    except Exception as exc:
        logging.error(f"MSight localization failed: {exc}")


class WorkflowExecutor:
    """Orchestrates the execution of multiple data processing workflows in sequence."""

    def __init__(
        self,
        workflows: List[str],
        selected_dataset: str,
        dataset: fo.Dataset,
        dataset_info: Dict,
    ):
        """Initializes with specified workflows, dataset selection, and dataset metadata."""
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
        failed_workflows: List[str] = []
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
                    from utils.anomaly_detection_data_preparation import (
                        AnomalyDetectionDataPreparation,
                    )

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
                        pbar.set_description(f"Anomalib model {MODEL_NAME}")

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
                        "hf_models_objectdetection",  # [0]
                        "ultralytics",                # [1]
                        "custom_codetr",              # [2]
                        "roboflow",                   # [3]
                        "roboflow_keypoint",          # [4]
                        "vitpose",                    # [5] standalone ViTPose fine-tuning
                        "roi_keypoint",               # [6] two-stage: RF-DETR + ViTPose
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
                        # Single GPU mode (https://github.com/huggingface/transformers/issues/28740)
                        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                        hf_models = config_autolabel["hf_models_objectdetection"]

                        # Dataset Conversion
                        try:
                            from utils.data_loader import (
                                FiftyOneTorchDatasetCOCO,
                                TorchToHFDatasetCOCO,
                            )

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
                        # Ultralytics Models
                        from workflows.auto_labeling import UltralyticsObjectDetection

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
                                "patience": config_autolabel["early_stop_patience"],
                                "batch_size": models_ultralytics[model_name][
                                    "batch_size"
                                ],
                                "img_size": models_ultralytics[model_name]["img_size"],
                                "export_dataset_root": export_dataset_root,
                                "inference_settings": config_autolabel[
                                    "inference_settings"
                                ],
                                "multi_scale": config_ultralytics["multi_scale"],
                                "cos_lr": config_ultralytics["cos_lr"],
                            }

                            workflow_auto_labeling_ultralytics(self.dataset, run_config)

                    if SUPPORTED_MODEL_SOURCES[2] in selected_model_source:
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

                        for config in (
                            pbar := tqdm(
                                codetr_configs, desc="Processing Co-DETR configurations"
                            )
                        ):
                            pbar.set_description(f"Co-DETR model {config}")
                            run_config["config"] = config
                            workflow_auto_labeling_custom_codetr(
                                self.dataset, self.dataset_info, run_config
                            )

                    if SUPPORTED_MODEL_SOURCES[3] in selected_model_source:
                        from workflows.auto_labeling import CustomRFDETRObjectDetection

                        config_rfdetr = config_autolabel["roboflow"]

                        # Shared config parameters
                        shared_config = {
                            "epochs": config_autolabel["epochs"],
                            "learning_rate": config_autolabel["learning_rate"],
                            "weight_decay": config_autolabel["weight_decay"],
                            "early_stop_patience": config_autolabel["early_stop_patience"],
                            "early_stop_threshold": config_autolabel["early_stop_threshold"],
                        }

                        run_config = {
                            "export_dataset_root": config_rfdetr["export_dataset_root"],
                            "mode": config_autolabel["mode"],
                            "inference_settings": config_autolabel["inference_settings"],
                            "config": None,
                            # RF-DETR specific parameters
                            "batch_size": config_rfdetr["batch_size"],
                            "grad_accum_steps": config_rfdetr["grad_accum_steps"],
                            "lr_encoder": config_rfdetr["lr_encoder"],
                            "resolution": config_rfdetr["resolution"],
                            "use_ema": config_rfdetr["use_ema"],
                            "gradient_checkpointing": config_rfdetr["gradient_checkpointing"],
                            "early_stopping_min_delta": config_rfdetr["early_stopping_min_delta"],
                            "early_stopping_use_ema": config_rfdetr["early_stopping_use_ema"],
                        }

                        rfdetr_configs = config_rfdetr["configs"]

                        for config in (
                            pbar := tqdm(rfdetr_configs, desc="Processing RF-DETR configurations")
                        ):
                            pbar.set_description(f"RF-DETR model {config}")
                            run_config["config"] = config

                            try:
                                wandb_exit_code = 0
                                wandb_run = wandb_init(
                                    run_name=config,
                                    project_name="RF-DETR Auto Labeling",
                                    dataset_name=self.dataset_info["name"],
                                    config=run_config,
                                    wandb_activate=True,
                                )

                                detector = CustomRFDETRObjectDetection(
                                    self.dataset, self.dataset_info, run_config
                                )

                                # Convert data to RF-DETR format (only needed for training)
                                if "train" in mode:
                                    detector.convert_data()

                                # Training
                                if "train" in mode:
                                    logging.info(f"Training RF-DETR model: {config}")
                                    detector.train(run_config, shared_config)

                                # Inference
                                if "inference" in mode:
                                    logging.info(f"Running inference for RF-DETR model: {config}")
                                    fallback_hf_map = config_rfdetr.get("fallback_hf_repo", {})
                                    fallback_file_map = config_rfdetr.get("fallback_hf_filename", {})
                                    rfdetr_inference_settings = {
                                        **config_autolabel["inference_settings"],
                                        "class_names": config_rfdetr.get("class_names"),
                                        "fallback_hf_repo": fallback_hf_map.get(config),
                                        "fallback_hf_filename": fallback_file_map.get(config),
                                    }
                                    detector.inference(
                                        inference_settings=rfdetr_inference_settings
                                    )

                            except Exception as e:
                                logging.error(f"Error during RF-DETR workflow with {config}: {e}")
                                wandb_exit_code = 1
                            finally:
                                wandb_close(wandb_exit_code)

                    if SUPPORTED_MODEL_SOURCES[4] in selected_model_source:
                        # RF-DETR with keypoint head
                        from workflows.rfdetr_keypoint import RFDETRKeypointDetection

                        config_rfdetr_kp = config_autolabel["roboflow_keypoint"]

                        shared_config = {
                            "epochs": config_autolabel["epochs"],
                            "learning_rate": config_autolabel["learning_rate"],
                            "weight_decay": config_autolabel["weight_decay"],
                            "early_stop_patience": config_autolabel["early_stop_patience"],
                            "early_stop_threshold": config_autolabel["early_stop_threshold"],
                        }

                        run_config_kp = {
                            "export_dataset_root": config_rfdetr_kp["export_dataset_root"],
                            "mode": config_autolabel["mode"],
                            "inference_settings": config_autolabel["inference_settings"],
                            "config": None,
                            # FiftyOne data-path selector — MUST be forwarded so that
                            # RFDETRKeypointDetection.train() uses the FO-native loader
                            # (prefetch_fo_split → FiftyOneKeypointDataset) instead of
                            # the COCO-export path.  Without this flag, fo_native
                            # defaults to False and all keypoint visibilities are 0
                            # (COCO fallback), causing OKS=nan for the entire run.
                            "fo_native": config_rfdetr_kp.get("fo_native", False),
                            "detection_field": config_rfdetr_kp.get("detection_field", "ground_truth"),
                            "target_label": config_rfdetr_kp.get("target_label", "pedestrian"),
                            "class_names": config_rfdetr_kp.get("class_names", ["pedestrian"]),
                            "num_classes": config_rfdetr_kp.get("num_classes", 1),
                            # Keypoint-specific
                            "keypoint_field": config_rfdetr_kp["keypoint_field"],
                            "keypoint_names": config_rfdetr_kp["keypoint_names"],
                            "num_keypoints": len(config_rfdetr_kp["keypoint_names"]),
                            "kp_xy_coef": config_rfdetr_kp.get("kp_xy_coef", 5.0),
                            "kp_vis_coef": config_rfdetr_kp.get("kp_vis_coef", 1.0),
                            "freeze_backbone_epochs": config_rfdetr_kp.get("freeze_backbone_epochs", 5),
                            "freeze_bbox_head": config_rfdetr_kp.get("freeze_bbox_head", False),
                            # RF-DETR parameters
                            "batch_size": config_rfdetr_kp.get("batch_size", 8),
                            "lr_encoder": config_rfdetr_kp.get("lr_encoder", None),
                            "resolution": config_rfdetr_kp.get("resolution", 560),
                            "pretrain_weights": config_rfdetr_kp.get("pretrain_weights", None),
                        }

                        for config in (
                            pbar := tqdm(
                                config_rfdetr_kp["configs"],
                                desc="Processing RF-DETR Keypoint configurations",
                            )
                        ):
                            pbar.set_description(f"RF-DETR Keypoint model {config}")
                            run_config_kp["config"] = config

                            try:
                                wandb_exit_code = 0
                                wandb_run = wandb_init(
                                    run_name=config,
                                    project_name="RF-DETR Keypoint Auto Labeling",
                                    dataset_name=self.dataset_info["name"],
                                    config=run_config_kp,
                                    wandb_activate=True,
                                )

                                detector_kp = RFDETRKeypointDetection(
                                    self.dataset, self.dataset_info, run_config_kp
                                )

                                detector_kp.convert_data()

                                if "train" in mode:
                                    logging.info(f"Training RF-DETR keypoint model: {config}")
                                    detector_kp.train(run_config_kp, shared_config)

                                if "inference" in mode:
                                    logging.info(f"Running keypoint inference: {config}")
                                    detector_kp.inference(
                                        inference_settings=config_autolabel["inference_settings"]
                                    )

                            except Exception as e:
                                logging.error(f"Error during RF-DETR keypoint workflow with {config}: {e}")
                                wandb_exit_code = 1
                            finally:
                                wandb_close(wandb_exit_code)


                    if SUPPORTED_MODEL_SOURCES[5] in selected_model_source:
                        # ── vitpose: standalone ViTPose-B fine-tuning ───────────────
                        config_vp = config_autolabel["vitpose"]
                        kp_names_vp = config_vp.get("keypoint_names", ["ankle_center"])

                        shared_config_vp = {
                            "epochs":              config_autolabel["epochs"],
                            "learning_rate":       config_autolabel["learning_rate"],
                            "weight_decay":        config_autolabel["weight_decay"],
                            "early_stop_patience": config_autolabel["early_stop_patience"],
                        }
                        run_config_vp = {
                            "mode":                     config_autolabel["mode"],
                            "vitpose_pretrain_weights": config_vp.get("vitpose_pretrain_weights", "usyd-community/vitpose-base-simple"),
                            "vitpose_save_dir":         config_vp.get("vitpose_save_dir", "output/models/vitpose/"),
                            "detection_field":          config_vp.get("detection_field", "ground_truth"),
                            "keypoint_field":           config_vp.get("keypoint_field", "pedestrian_points"),
                            "target_label":             config_vp.get("target_label", "pedestrian"),
                            "keypoint_names":           kp_names_vp,
                            "num_keypoints":            len(kp_names_vp),
                            "kp_sigma":                 config_vp.get("kp_sigma", 0.089),
                            "batch_size":               config_vp.get("batch_size", 32),
                            "freeze_backbone":          config_vp.get("freeze_backbone", True),
                            "freeze_backbone_epochs":   config_vp.get("freeze_backbone_epochs", 5),
                        }

                        try:
                            wandb_exit_code = 0
                            wandb_run = wandb_init(
                                run_name="vitpose-finetune",
                                project_name="ViTPose Fine-Tuning",
                                dataset_name=self.dataset_info["name"],
                                config=run_config_vp,
                                wandb_activate=True,
                            )
                            vp_detector = ViTPoseKeypointDetection(
                                self.dataset, self.dataset_info, run_config_vp
                            )
                            if "train" in mode:
                                logging.info("Downloading ViTPose-B pretrained weights…")
                                vp_detector.download_weights()
                                logging.info("Fine-tuning ViTPose-B on GT RoI crops…")
                                vp_detector.train(run_config_vp, shared_config_vp)
                            if "inference" in mode:
                                logging.info("ViTPose inference with GT boxes…")
                                vp_detector.inference(
                                    inference_settings=config_autolabel["inference_settings"]
                                )
                        except Exception as e:
                            logging.error(f"Error in vitpose workflow: {e}")
                            wandb_exit_code = 1
                        finally:
                            wandb_close(wandb_exit_code)

                    if SUPPORTED_MODEL_SOURCES[6] in selected_model_source:
                        # ── roi_keypoint: RF-DETR detect → ViTPose predict ──────────
                        config_roi   = config_autolabel["roi_keypoint"]
                        kp_names_roi = config_roi.get("keypoint_names", ["ankle_center"])

                        run_config_roi = {
                            "mode":                config_autolabel["mode"],
                            "rfdetr_model":        config_roi.get("rfdetr_model", "rfdetr_2xlarge"),
                            "pretrain_weights":    config_roi.get("pretrain_weights", None),
                            "detection_threshold": config_roi.get("detection_threshold", 0.3),
                            "vitpose_weights":     config_roi.get("vitpose_weights", None),
                            "target_label":        config_roi.get("target_label", "pedestrian"),
                            "keypoint_names":      kp_names_roi,
                            "num_keypoints":       len(kp_names_roi),
                            "kp_sigma":            config_roi.get("kp_sigma", 0.089),
                        }

                        try:
                            wandb_exit_code = 0
                            wandb_run = wandb_init(
                                run_name=f"roi_kp-{config_roi.get('rfdetr_model','rfdetr_2xlarge')}",
                                project_name="RoI Keypoint Inference",
                                dataset_name=self.dataset_info["name"],
                                config=run_config_roi,
                                wandb_activate=True,
                            )
                            roi_detector = RoIKeypointDetection(
                                self.dataset, self.dataset_info, run_config_roi
                            )
                            if "inference" in mode:
                                logging.info(
                                    "Running RoI Keypoint inference "
                                    "(RF-DETR detect → ViTPose predict)…"
                                )
                                roi_detector.inference(
                                    inference_settings=config_autolabel["inference_settings"]
                                )
                        except Exception as e:
                            logging.error(f"Error in roi_keypoint workflow: {e}")
                            wandb_exit_code = 1
                        finally:
                            wandb_close(wandb_exit_code)

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

                elif workflow == "auto_label_mask":
                    config = WORKFLOWS["auto_label_mask"]
                    workflow_auto_label_mask(self.dataset, self.dataset_info, config)

                elif workflow == "class_mapping":
                    # Config
                    run_config = WORKFLOWS["class_mapping"]

                    # Workflow
                    workflow_class_mapping(
                        self.dataset,
                        self.dataset_info,
                        run_config,
                        test_dataset_source=None,
                        test_dataset_target=None,
                    )
                elif workflow == "vitpose_download":
                    cfg = WORKFLOWS["vitpose_download"]
                    download_vitpose_weights(
                        model_name=cfg.get("vitpose_model", "usyd-community/vitpose-base-simple"),
                        save_dir=cfg.get("save_dir", "output/models/vitpose/"),
                    )

                elif workflow == "data_ingest":
                    dataset = run_data_ingest()

                    dataset_info = {
                        "name": "custom_dataset",
                        "v51_type": "FiftyOneDataset",
                        "splits": ["train", "val", "test"],
                    }

                    self.dataset = dataset
                    self.dataset_info = dataset_info
                    self.selected_dataset = "custom_dataset"

                    logging.info(f"Data ingestion completed successfully.")


                else:
                    logging.error(
                        f"Workflow {workflow} not found. Check available workflows in config.py."
                    )
                    return False

                #cleanup_memory()  # Clean after each workflow
                logging.info(f"Completed workflow {workflow} and cleaned up memory")

            except Exception as e:
                # logging.exception keeps the traceback -- logging.error dropped
                # it, which made these swallowed failures very hard to diagnose.
                logging.exception(f"Workflow {workflow}: An error occurred: {e}")
                failed_workflows.append(workflow)
                wandb_close(exit_code=1)
                #cleanup_memory()  # Clean up even after failure

        # Remaining workflows still run after one fails (unchanged), but the
        # failure is no longer hidden: this used to return True regardless.
        if failed_workflows:
            logging.error(f"Failed workflows: {failed_workflows}")
            return False

        return True


def main():
    """Executes the data processing workflow, loads dataset, and launches Voxel51 visualization interface."""
    time_start = time.time()
    configure_logging()

    # Signal handler for CTRL + C
    signal.signal(signal.SIGINT, signal_handler)

    # Execute workflows
    if "data_ingest" in SELECTED_WORKFLOW:
        executor = WorkflowExecutor(
            SELECTED_WORKFLOW,
            SELECTED_DATASET["name"],
            dataset=None,
            dataset_info=None,
        )
        workflows_ok = executor.execute()

        # FIX: Pull back outputs after ingestion
        dataset = executor.dataset
        dataset_info = executor.dataset_info

    else:
        from utils.dataset_loader import load_dataset

        dataset, dataset_info = load_dataset(SELECTED_DATASET)

        executor = WorkflowExecutor(
            SELECTED_WORKFLOW,
            SELECTED_DATASET["name"],
            dataset,
            dataset_info,
        )
        workflows_ok = executor.execute()


    if dataset is not None:
        dataset.reload()
        dataset.save()

        if MSIGHT_CONFIG.get("run_localization", False):
            _run_msight_localization(dataset)

        arrange_fields_in_groups(dataset)
        logging.info(f"Launching Voxel51 session for dataset {dataset_info['name']}.")

        # Dataset stats
        logging.debug(dataset)
        logging.debug(dataset.stats(include_media=True))

        # V51 UI launch
        session = fo.launch_app(
            dataset, address=V51_ADDRESS, port=V51_PORT, remote=V51_REMOTE
        )
    else:
        logging.info("Skipping Voxel51 session.")



    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")

    return workflows_ok


if __name__ == "__main__":
    cleanup_memory()
    # Exit non-zero when a workflow failed. Every MCP tool that runs main.py as
    # a subprocess decides success from the exit code, so returning 0 after a
    # caught workflow exception reported the failed run as a success.
    if not main():
        sys.exit(1)
