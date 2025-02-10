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
    ACCEPTED_SPLITS,
    GLOBAL_SEED,
    SELECTED_DATASET,
    SELECTED_WORKFLOW,
    V51_ADDRESS,
    V51_PORT,
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


def signal_handler(sig, frame):
    logging.error("You pressed Ctrl+C!")
    try:
        wandb.finish()
    except:
        pass
    sys.exit(0)


def workflow_aws_download():
    dataset = None
    dataset_name = None
    wandb_exit_code = 0
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

        wandb_run = wandb_init(
            run_name=dataset_name,
            project_name="AWS Download",
            dataset_name=dataset_name,
            log_dir=log_dir,
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

    except Exception as e:
        logging.error(f"AWS Download and Extraction failed: {e}")
        wandb_exit_code = 1

    finally:
        wandb_close(wandb_run, wandb_exit_code)

    return dataset, dataset_name


def workflow_anomaly_detection(
    dataset, dataset_info, eval_metrics, run_config, log_dir, mode
):
    ano_dec = Anodec(
        dataset=dataset,
        eval_metrics=eval_metrics,
        dataset_info=dataset_info,
        config=run_config,
        tensorboard_output=log_dir,
    )
    if mode == "train":
        ano_dec.train_and_export_model()
        ano_dec.run_inference()
        ano_dec.eval_v51()
    elif mode == "inference":
        ano_dec.run_inference()
    else:
        logging.error(f"Mode {mode} not suported.")

    return True


def workflow_embedding_selection(dataset, dataset_info, MODEL_NAME, log_dir, mode):
    embedding_selector = EmbeddingSelection(dataset, dataset_info, MODEL_NAME, log_dir)
    embedding_selector.compute_embeddings(mode)
    embedding_selector.compute_similarity()

    # Find representative and unique samples as center points for further selections
    embedding_selector.compute_representativeness()
    embedding_selector.compute_unique_images_greedy()
    embedding_selector.compute_unique_images_deterministic()

    # Select samples similar to the center points to enlarge the dataset
    embedding_selector.compute_similar_images()

    return True


def workflow_auto_labeling(dataset, hf_dataset, mode, run_config):

    detector = HuggingFaceObjectDetection(
        dataset=dataset,
        config=run_config,
    )
    if mode == "train":
        detector.train(hf_dataset)
        detector.inference(hf_dataset)
    elif mode == "inference":
        detector.inference(hf_dataset)
    else:
        logging.error(f"Mode {mode} is not supported.")


def workflow_zero_shot_object_detection(dataset, dataset_info):
    # Set multiprocessing mode for CUDA multiprocessing
    try:
        mp.set_start_method("spawn")
    except:
        pass
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

    return True


def workflow_ensemble_exploration():
    pass


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
        wandb_run = None
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
                            self.selected_dataset = dataset_name
                            logging.warning(
                                f"Selected dataset overwritten to {dataset_name}"
                            )

                elif workflow == "embedding_selection":
                    embedding_models = WORKFLOWS["embedding_selection"][
                        "embedding_models"
                    ]

                    for MODEL_NAME in (
                        pbar := tqdm(embedding_models, desc="Selection by Embedings")
                    ):
                        wandb_exit_code = 0
                        try:
                            pbar.set_description(
                                f"Selection by embeddings with model {MODEL_NAME}."
                            )

                            wandb_run, log_dir = wandb_init(
                                run_name=MODEL_NAME,
                                project_name="Selection by Embedding",
                                dataset_name=self.selected_dataset,
                            )

                            workflow_embedding_selection(
                                self.dataset, self.dataset_info, MODEL_NAME, log_dir
                            )
                        except Exception as e:
                            logging.error(
                                f"An error occurred with model {MODEL_NAME}: {e}"
                            )
                            wandb_exit_code = 1
                        finally:
                            wandb_close(wandb_run, wandb_exit_code)

                elif workflow == "anomaly_detection":

                    ano_dec_config = WORKFLOWS["anomaly_detection"]
                    mode = ano_dec_config["mode"]
                    anomalib_image_models = ano_dec_config["anomalib_image_models"]
                    eval_metrics = ano_dec_config["anomalib_eval_metrics"]

                    data_preparer = AnomalyDetectionDataPreparation(
                        self.dataset, self.selected_dataset
                    )

                    for MODEL_NAME in (
                        pbar := tqdm(anomalib_image_models, desc="Anomalib")
                    ):
                        wandb_exit_code = 0
                        try:
                            pbar.set_description(f"Anomalib model {MODEL_NAME}.")

                            run_config = {
                                "model_name": MODEL_NAME,
                                "image_size": anomalib_image_models[MODEL_NAME][
                                    "image_size"
                                ],
                                "batch_size": anomalib_image_models[MODEL_NAME][
                                    "batch_size"
                                ],
                                "epochs": ano_dec_config["epochs"],
                                "early_stop_patience": ano_dec_config[
                                    "early_stop_patience"
                                ],
                                "data_root": data_preparer.export_root,
                            }
                            wandb_run, log_dir = wandb_init(
                                run_name=MODEL_NAME,
                                project_name="Selection by Anomaly Detection",
                                dataset_name=self.selected_dataset,
                                config=run_config,
                            )

                            workflow_anomaly_detection(
                                data_preparer.dataset_ano_dec,
                                self.dataset_info,
                                eval_metrics,
                                run_config,
                                log_dir,
                                mode,
                            )
                        except Exception as e:
                            logging.error(f"Error in Anomaly Detection: {e}")
                            wandb_exit_code = 1
                        finally:
                            wandb_close(wandb_run=wandb_run, exit_code=wandb_exit_code)

                elif workflow == "auto_labeling":
                    mode = WORKFLOWS["auto_labeling"]["mode"]
                    selected_model_source = WORKFLOWS["auto_labeling"]["model_source"]

                    if selected_model_source == "hf_models_objectdetection":
                        hf_models = WORKFLOWS["auto_labeling"][
                            "hf_models_objectdetection"
                        ]

                        # Convert dataset
                        logging.info("Converting dataset into Hugging Face format.")
                        pytorch_dataset = FiftyOneTorchDatasetCOCO(self.dataset)
                        pt_to_hf_converter = TorchToHFDatasetCOCO(pytorch_dataset)
                        hf_dataset = pt_to_hf_converter.convert()

                        # Train models
                        for MODEL_NAME in (
                            pbar := tqdm(hf_models, desc="Auto Labeling Models")
                        ):
                            try:
                                wandb_exit_code = 0
                                pbar.set_description(f"Training model {MODEL_NAME}")

                                wandb_run = wandb_init(
                                    run_name=MODEL_NAME,
                                    project_name="Auto Labeling Hugging Face",
                                    dataset_name=self.selected_dataset,
                                )

                                run_config = {}

                                workflow_auto_labeling(dataset, hf_dataset, mode)

                            except Exception as e:
                                logging.error(
                                    f"An error occurred with model {MODEL_NAME}: {e}"
                                )
                                wandb_exit_code = 1

                            finally:
                                wandb_close(wandb_run, wandb_exit_code)

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

                    config["overrides"]["run_config"][
                        "v51_dataset_name"
                    ] = self.selected_dataset
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
    logging.info(dataset)
    fo.pprint(dataset.stats(include_media=True))
    session = fo.launch_app(dataset, address=V51_ADDRESS, port=V51_PORT, remote=True)

    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")


if __name__ == "__main__":
    main()
