import datetime
import json
import logging
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from difflib import get_close_matches
from functools import partial
from pathlib import Path
from typing import Union

import albumentations as A
import fiftyone as fo
import numpy as np
import psutil
import torch
import torch.multiprocessing as mp
from accelerate.test_utils.testing import get_backend
from datasets import Split
from fiftyone import ViewField as F
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForObjectDetection,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from ultralytics import YOLO

import wandb
from config.config import (
    ACCEPTED_SPLITS,
    GLOBAL_SEED,
    HF_DO_UPLOAD,
    HF_ROOT,
    NUM_WORKERS,
    WANDB_ACTIVE,
    WORKFLOWS,
)
from utils.dataset_loader import get_supported_datasets
from utils.logging import configure_logging
from utils.sample_field_operations import add_sample_field


def get_dataset_and_model_from_hf_id(hf_id: str):
    """Extract dataset and model name from HuggingFace ID by matching against supported datasets."""
    # HF ID follows structure organization/dataset_model
    # Both dataset and model can contain "_" as well

    # Remove organization (everything before the first "/")
    hf_id = hf_id.split("/", 1)[-1]

    # Find all dataset names that appear in hf_id
    supported_datasets = get_supported_datasets()
    matches = [
        dataset_name for dataset_name in supported_datasets if dataset_name in hf_id
    ]

    if not matches:
        logging.warning(
            f"Dataset name could not be extracted from Hugging Face ID {hf_id}"
        )
        dataset_name = "no_dataset_name"
    else:
        # Return the longest match (most specific)
        dataset_name = max(matches, key=len)

    # Get model name by removing dataset name from hf_id
    model_name = hf_id.replace(dataset_name, "").strip("_")
    if not model_name:
        logging.warning(
            f"Model name could not be extracted from Hugging Face ID {hf_id}"
        )
        model_name = "no_model_name"

    return dataset_name, model_name


# Handling timeouts
class TimeoutException(Exception):
    """Custom exception for handling dataloader timeouts."""

    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Dataloader creation timed out")


class ZeroShotInferenceCollateFn:
    """Collate function for zero-shot inference that prepares batches for model input."""

    def __init__(
        self,
        hf_model_config_name,
        hf_processor,
        batch_size,
        object_classes,
        batch_classes,
    ):
        """Initialize the auto labeling model with the Hugging Face model config, processor, batch size, object classes, and batch classes."""
        try:
            self.hf_model_config_name = hf_model_config_name
            self.processor = hf_processor
            self.batch_size = batch_size
            self.object_classes = object_classes
            self.batch_classes = batch_classes
        except Exception as e:
            logging.error(f"Error in collate init of DataLoader: {e}")

    def __call__(self, batch):
        """Processes a batch of data by preparing images and labels for model input."""
        try:
            images, labels = zip(*batch)
            target_sizes = [tuple(img.shape[1:]) for img in images]

            # Adjustments for final batch
            n_images = len(images)
            if n_images < self.batch_size:
                self.batch_classes = [self.object_classes] * n_images

            # Apply PIL transformation for specific models
            if self.hf_model_config_name == "OmDetTurboConfig":
                images = [to_pil_image(image) for image in images]

            inputs = self.processor(
                text=self.batch_classes,
                images=images,
                return_tensors="pt",
                padding=True,  # Allow for differently sized images
            )

            return inputs, labels, target_sizes, self.batch_classes
        except Exception as e:
            logging.error(f"Error in collate function of DataLoader: {e}")


class ZeroShotObjectDetection:
    """Zero-shot object detection using various HuggingFace models with multi-GPU support."""

    def __init__(
        self,
        dataset_torch: torch.utils.data.Dataset,
        dataset_info,
        config,
        detections_path="./output/detections/",
        log_root="./logs/",
    ):
        """Initialize the zero-shot object detection labeler with dataset, configuration, and path settings."""
        self.dataset_torch = dataset_torch
        self.dataset_info = dataset_info
        self.dataset_name = dataset_info["name"]
        self.object_classes = config["object_classes"]
        self.detection_threshold = config["detection_threshold"]
        self.detections_root = os.path.join(detections_path, self.dataset_name)
        self.tensorboard_root = os.path.join(
            log_root, "tensorboard/zeroshot_object_detection"
        )

        logging.info(f"Zero-shot models will look for {self.object_classes}")

    def exclude_stored_predictions(
        self, dataset_v51: fo.Dataset, config, do_exclude=False
    ):
        """Checks for existing predictions and loads them from disk if available."""
        dataset_schema = dataset_v51.get_field_schema()
        models_splits_dict = {}
        for model_name, value in config["hf_models_zeroshot_objectdetection"].items():
            model_name_key = re.sub(r"[\W-]+", "_", model_name)
            pred_key = re.sub(
                r"[\W-]+", "_", "pred_zsod_" + model_name
            )  # od for Object Detection
            # Check if data already stored in V51 dataset
            if pred_key in dataset_schema and do_exclude is True:
                logging.warning(
                    f"Skipping model {model_name}. Predictions already stored in Voxel51 dataset."
                )
            # Check if data already stored on disk
            elif (
                os.path.isdir(os.path.join(self.detections_root, model_name_key))
                and do_exclude is True
            ):
                try:
                    logging.info(f"Loading {model_name} predictions from disk.")
                    temp_dataset = fo.Dataset.from_dir(
                        dataset_dir=os.path.join(self.detections_root, model_name_key),
                        dataset_type=fo.types.COCODetectionDataset,
                        name="temp_dataset",
                        data_path="data.json",
                    )

                    # Copy all detections from stored dataset into our dataset
                    detections = temp_dataset.values("detections.detections")
                    add_sample_field(
                        dataset_v51,
                        pred_key,
                        fo.EmbeddedDocumentField,
                        embedded_doc_type=fo.Detections,
                    )
                    dataset_v51.set_values(f"{pred_key}.detections", detections)
                except Exception as e:
                    logging.error(
                        f"Data in {os.path.join(self.detections_root, model_name_key)} could not be loaded. Error: {e}"
                    )
                finally:
                    fo.delete_dataset("temp_dataset")
            # Assign model to be run
            else:
                models_splits_dict[model_name] = value

        logging.info(f"Models to be run: {models_splits_dict}")
        return models_splits_dict

    # Worker functions
    def update_queue_sizes_worker(
        self, queues, queue_sizes, largest_queue_index, max_queue_size
    ):
        """Monitor and manage multiple result queues for balanced processing."""
        experiment_name = f"queue_size_monitor_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        log_directory = os.path.join(
            self.tensorboard_root, self.dataset_name, experiment_name
        )
        wandb.tensorboard.patch(root_logdir=log_directory)
        if WANDB_ACTIVE:
            wandb.init(
                name=f"queue_size_monitor_{os.getpid()}",
                job_type="inference",
                project="Zero Shot Object Detection",
            )
        writer = SummaryWriter(log_dir=log_directory)

        step = 0

        while True:
            for i, queue in enumerate(queues):
                queue_sizes[i] = queue.qsize()
                writer.add_scalar(f"queue_size/items/{i}", queue_sizes[i], step)

            step += 1

            # Find the index of the largest queue
            max_size = max(queue_sizes)
            max_index = queue_sizes.index(max_size)

            # Calculate the total size of all queues
            total_size = sum(queue_sizes)

            # If total_size is greater than 0, calculate the probabilities
            if total_size > 0:
                # Normalize the queue sizes by the max_queue_size
                normalized_sizes = [size / max_queue_size for size in queue_sizes]

                # Calculate probabilities based on normalized sizes
                probabilities = [
                    size / sum(normalized_sizes) for size in normalized_sizes
                ]

                # Use random.choices with weights (probabilities)
                chosen_queue_index = random.choices(
                    range(len(queues)), weights=probabilities, k=1
                )[0]

                largest_queue_index.value = chosen_queue_index
            else:
                largest_queue_index.value = max_index

            time.sleep(0.1)

    def process_outputs_worker(
        self,
        result_queues,
        largest_queue_index,
        inference_finished,
        max_queue_size,
        wandb_activate=False,
    ):
        """Process model outputs from result queues and save to dataset."""
        configure_logging()
        logging.info(f"Process ID: {os.getpid()}. Results processing process started")
        dataset_v51 = fo.load_dataset(self.dataset_name)
        processing_successful = None

        # Logging
        experiment_name = f"post_process_{os.getpid()}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        log_directory = os.path.join(
            self.tensorboard_root, self.dataset_name, experiment_name
        )
        wandb.tensorboard.patch(root_logdir=log_directory)
        if WANDB_ACTIVE and wandb_activate:
            wandb.init(
                name=f"post_process_{os.getpid()}",
                job_type="inference",
                project="Zero Shot Object Detection",
            )
        writer = SummaryWriter(log_dir=log_directory)
        n_processed_images = 0

        logging.info(f"Post-Processor {os.getpid()} starting loop.")

        while True:
            results_queue = result_queues[largest_queue_index.value]
            writer.add_scalar(
                f"post_processing/selected_queue",
                largest_queue_index.value,
                n_processed_images,
            )

            if results_queue.qsize() == max_queue_size:
                logging.warning(
                    f"Queue full: {results_queue.qsize()}. Consider increasing number of post-processing workers."
                )

            # Exit only when inference is finished and the queue is empty
            if inference_finished.value and results_queue.empty():
                dataset_v51.save()
                logging.info(
                    f"Post-processing worker {os.getpid()} has finished all outputs."
                )
                break

            # Process results from the queue if available
            if not results_queue.empty():
                try:
                    time_start = time.time()

                    result = results_queue.get_nowait()

                    processing_successful = self.process_outputs(
                        dataset_v51,
                        result,
                        self.object_classes,
                        self.detection_threshold,
                    )

                    # Performance logging
                    n_images = len(result["labels"])
                    time_end = time.time()
                    duration = time_end - time_start
                    batches_per_second = 1 / duration
                    frames_per_second = batches_per_second * n_images
                    n_processed_images += n_images
                    writer.add_scalar(
                        f"post_processing/frames_per_second",
                        frames_per_second,
                        n_processed_images,
                    )

                    del result  # Explicit removal from device

                except Exception as e:
                    continue

            else:
                continue

        writer.close()
        wandb.finish(exit_code=0)
        return processing_successful  # Return last processing status

    def gpu_worker(
        self,
        gpu_id,
        cpu_cores,
        task_queue,
        results_queue,
        done_event,
        post_processing_finished,
        set_cpu_affinity=False,
    ):
        """Run model inference on specified GPU with dedicated CPU cores."""
        dataset_v51 = fo.load_dataset(
            self.dataset_name
        )  # NOTE Only for the case of sequential processing
        configure_logging()
        # Set CPU
        if set_cpu_affinity:
            # Allow only certain CPU cores
            psutil.Process().cpu_affinity(cpu_cores)
        logging.info(f"Available CPU cores: {psutil.Process().cpu_affinity()}")
        max_n_cpus = len(cpu_cores)
        torch.set_num_threads(max_n_cpus)

        # Set GPU
        logging.info(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        device = torch.device(f"cuda:{gpu_id}")

        run_successful = None
        with torch.cuda.device(gpu_id):
            while True:
                if post_processing_finished.value and task_queue.empty():
                    # Keep alive until post-processing is done
                    break

                if task_queue.empty():
                    done_event.set()

                if not task_queue.empty():
                    try:
                        task_metadata = task_queue.get(
                            timeout=5
                        )  # Timeout to prevent indefinite blocking
                    except Exception as e:
                        break  # Exit if no more tasks
                    run_successful = self.model_inference(
                        task_metadata,
                        device,
                        self.dataset_torch,
                        dataset_v51,
                        self.object_classes,
                        results_queue,
                        self.tensorboard_root,
                    )
                    logging.info(
                        f"Worker for GPU {gpu_id} finished run successful: {run_successful}"
                    )
                else:
                    continue
        return run_successful  # Return last processing status

    def eval_and_export_worker(self, models_ready_queue, n_models):
        """Evaluate model performance and export results for completed models."""
        configure_logging()
        logging.info(f"Process ID: {os.getpid()}. Eval-and-export process started")

        dataset = fo.load_dataset(self.dataset_name)
        run_successful = None
        models_done = 0

        while True:
            if not models_ready_queue.empty():
                try:
                    dict = models_ready_queue.get(
                        timeout=5
                    )  # Timeout to prevent indefinite blocking
                    model_name = dict["model_name"]
                    pred_key = re.sub(r"[\W-]+", "_", "pred_zsod_" + model_name)
                    eval_key = re.sub(r"[\W-]+", "_", "eval_zsod_" + model_name)
                    dataset.reload()
                    run_successful = self.eval_and_export(
                        dataset, model_name, pred_key, eval_key
                    )
                    models_done += 1
                    logging.info(
                        f"Evaluation and export of {models_done}/{n_models} models done."
                    )
                except Exception as e:
                    logging.error(f"Error in eval-and-export worker: {e}")
                    continue

            if models_done == n_models:
                break

        return run_successful

    # Functionality functions
    def model_inference(
        self,
        metadata: dict,
        device: str,
        dataset: torch.utils.data.Dataset,
        dataset_v51: fo.Dataset,
        object_classes: list,
        results_queue: Union[queue.Queue, mp.Queue],
        root_log_dir: str,
        persistent_workers: bool = False,
    ):
        """Model inference method running zero-shot object detection on provided dataset and device, returning success status."""
        writer = None
        run_successful = True
        processor, model, inputs, outputs, result, dataloader = (
            None,
            None,
            None,
            None,
            None,
            None,
        )  # For finally block

        # Timeout handler
        dataloader_timeout = 60
        signal.signal(signal.SIGALRM, timeout_handler)

        try:
            # Metadata
            run_id = metadata["run_id"]
            model_name = metadata["model_name"]
            dataset_name = metadata["dataset_name"]
            is_subset = metadata["is_subset"]
            batch_size = metadata["batch_size"]

            logging.info(
                f"Process ID: {os.getpid()}, Run ID: {run_id}, Device: {device}, Model: {model_name}"
            )

            # Load the model
            logging.info(f"Loading model {model_name}")
            processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            model = model.to(device, non_blocking=True)
            model.eval()
            hf_model_config = AutoConfig.from_pretrained(model_name)
            hf_model_config_name = type(hf_model_config).__name__
            batch_classes = [object_classes] * batch_size
            logging.info(f"Loaded model type {hf_model_config_name}")

            # Dataloader
            logging.info("Generating dataloader")
            if is_subset:
                chunk_index_start = metadata["chunk_index_start"]
                chunk_index_end = metadata["chunk_index_end"]
                logging.info(f"Length of dataset: {len(dataset)}")
                logging.info(f"Subset start index: {chunk_index_start}")
                logging.info(f"Subset stop index: {chunk_index_end}")
                dataset = Subset(dataset, range(chunk_index_start, chunk_index_end))

            zero_shot_inference_preprocessing = ZeroShotInferenceCollateFn(
                hf_model_config_name=hf_model_config_name,
                hf_processor=processor,
                object_classes=object_classes,
                batch_size=batch_size,
                batch_classes=batch_classes,
            )
            num_workers = WORKFLOWS["auto_labeling_zero_shot"]["n_worker_dataloader"]
            prefetch_factor = WORKFLOWS["auto_labeling_zero_shot"][
                "prefetch_factor_dataloader"
            ]
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                collate_fn=zero_shot_inference_preprocessing,
            )

            dataloader_length = len(dataloader)
            if dataloader_length < 1:
                logging.error(
                    f"Dataloader has insufficient data: {dataloader_length} entries. Please check your dataset and DataLoader configuration."
                )

            # Logging
            experiment_name = f"{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{device}"
            log_directory = os.path.join(root_log_dir, dataset_name, experiment_name)
            wandb.tensorboard.patch(root_logdir=log_directory)
            if WANDB_ACTIVE:
                wandb.init(
                    name=f"{model_name}_{device}",
                    job_type="inference",
                    project="Zero Shot Object Detection",
                    config=metadata,
                )
            writer = SummaryWriter(log_dir=log_directory)

            # Inference Loop
            logging.info(f"{os.getpid()}: Starting inference loop5")
            n_processed_images = 0
            for inputs, labels, target_sizes, batch_classes in tqdm(
                dataloader, desc="Inference Loop"
            ):
                signal.alarm(dataloader_timeout)
                try:
                    time_start = time.time()
                    n_images = len(labels)
                    inputs = inputs.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"), torch.inference_mode():
                        outputs = model(**inputs)

                    result = {
                        "inputs": inputs,
                        "outputs": outputs,
                        "processor": processor,
                        "target_sizes": target_sizes,
                        "labels": labels,
                        "model_name": model_name,
                        "hf_model_config_name": hf_model_config_name,
                        "batch_classes": batch_classes,
                    }

                    logging.debug(f"{os.getpid()}: Putting result into queue")

                    results_queue.put(
                        result, timeout=60
                    )  # Ditch data only after 60 seconds

                    # Logging
                    time_end = time.time()
                    duration = time_end - time_start
                    batches_per_second = 1 / duration
                    frames_per_second = batches_per_second * n_images
                    n_processed_images += n_images
                    logging.debug(
                        f"{os.getpid()}: Number of processes images: {n_processed_images}"
                    )
                    writer.add_scalar(
                        f"inference/frames_per_second",
                        frames_per_second,
                        n_processed_images,
                    )

                except TimeoutException:
                    logging.warning(
                        f"Dataloader loop got stuck. Continuing with next batch."
                    )
                    continue

                finally:
                    signal.alarm(0)  # Cancel the alarm

            # Flawless execution
            wandb_exit_code = 0

        except Exception as e:
            wandb_exit_code = 1
            run_successful = False
            logging.error(f"Error in Process {os.getpid()}: {e}")
        finally:
            try:
                wandb.finish(exit_code=wandb_exit_code)
            except:
                pass

            # Explicit removal from device
            del (
                processor,
                model,
                inputs,
                outputs,
                result,
                dataloader,
            )

            torch.cuda.empty_cache()
            wandb.tensorboard.unpatch()
            if writer:
                writer.close()
            return run_successful

    def process_outputs(self, dataset_v51, result, object_classes, detection_threshold):
        """Process outputs from object detection models, extracting bounding boxes and labels to save to the dataset."""
        try:
            inputs = result["inputs"]
            outputs = result["outputs"]
            target_sizes = result["target_sizes"]
            labels = result["labels"]
            model_name = result["model_name"]
            hf_model_config_name = result["hf_model_config_name"]
            batch_classes = result["batch_classes"]
            processor = result["processor"]

            # Processing output
            if hf_model_config_name == "GroundingDinoConfig":
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=detection_threshold,
                    text_threshold=detection_threshold,
                )
            elif hf_model_config_name in ["Owlv2Config", "OwlViTConfig"]:
                results = processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    threshold=detection_threshold,
                    target_sizes=target_sizes,
                    text_labels=batch_classes,
                )
            elif hf_model_config_name == "OmDetTurboConfig":
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    text_labels=batch_classes,
                    threshold=detection_threshold,
                    nms_threshold=detection_threshold,
                    target_sizes=target_sizes,
                )
            else:
                logging.error(f"Invalid model name: {hf_model_config_name}")

            if not len(results) == len(target_sizes) == len(labels):
                logging.error(
                    f"Lengths of results, target_sizes, and labels do not match: {len(results)}, {len(target_sizes)}, {len(labels)}"
                )
            for result, size, target in zip(results, target_sizes, labels):
                boxes, scores, labels = (
                    result["boxes"],
                    result["scores"],
                    result["text_labels"],
                )

                img_height = size[0]
                img_width = size[1]

                detections = []
                for box, score, label in zip(boxes, scores, labels):
                    processing_successful = True
                    if hf_model_config_name == "GroundingDinoConfig":
                        # Outputs do not comply with given labels
                        # Grounding DINO outputs multiple pairs of object boxes and noun phrases for a given (Image, Text) pair
                        # There can be either multiple labels per output ("bike van"), incomplete ones ("motorcyc"), or broken ones ("##cic")
                        processed_label = label.split()[
                            0
                        ]  # Assume first output is the best output
                        if processed_label in object_classes:
                            label = processed_label
                            top_left_x = box[0].item()
                            top_left_y = box[1].item()
                            box_width = (box[2] - box[0]).item()
                            box_height = (box[3] - box[1]).item()
                        else:
                            matches = get_close_matches(
                                processed_label, object_classes, n=1, cutoff=0.6
                            )
                            selected_label = matches[0] if matches else None
                            if selected_label:
                                logging.debug(
                                    f"Mapped output '{processed_label}' to class '{selected_label}'"
                                )
                                label = selected_label
                                top_left_x = box[0].item()
                                top_left_y = box[1].item()
                                box_width = (box[2] - box[0]).item()
                                box_height = (box[3] - box[1]).item()
                            else:
                                logging.debug(
                                    f"Skipped detection with {hf_model_config_name} due to unclear output: {label}"
                                )
                                processing_successful = False

                    elif hf_model_config_name in [
                        "Owlv2Config",
                        "OwlViTConfig",
                        "OmDetTurboConfig",
                    ]:
                        top_left_x = box[0].item() / img_width
                        top_left_y = box[1].item() / img_height
                        box_width = (box[2].item() - box[0].item()) / img_width
                        box_height = (box[3].item() - box[1].item()) / img_height

                    if (
                        processing_successful
                    ):  # Skip GroundingDinoConfig labels that could not be processed
                        detection = fo.Detection(
                            label=label,
                            bounding_box=[
                                top_left_x,
                                top_left_y,
                                box_width,
                                box_height,
                            ],
                            confidence=score.item(),
                        )
                        detection["bbox_area"] = (
                            detection["bounding_box"][2] * detection["bounding_box"][3]
                        )
                        detections.append(detection)

                # Attach label to V51 dataset
                pred_key = re.sub(
                    r"[\W-]+", "_", "pred_zsod_" + model_name
                )  # zsod Zero-Shot Object Deection
                sample = dataset_v51[target["image_id"]]
                sample[pred_key] = fo.Detections(detections=detections)
                sample.save()

        except Exception as e:
            logging.error(f"Error in processing outputs: {e}")
            processing_successful = False
        finally:
            return processing_successful

    def eval_and_export(self, dataset_v51, model_name, pred_key, eval_key):
        """Populate dataset with evaluation results (if ground_truth available)"""
        try:
            dataset_v51.evaluate_detections(
                pred_key,
                gt_field="ground_truth",
                eval_key=eval_key,
                compute_mAP=True,
            )
        except Exception as e:
            logging.warning(f"Evaluation not possible: {e}")

        # Store labels https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.export
        model_name_key = re.sub(r"[\W-]+", "_", model_name)
        dataset_v51.export(
            export_dir=os.path.join(self.detections_root, model_name_key),
            dataset_type=fo.types.COCODetectionDataset,
            data_path="data.json",
            export_media=None,  # "manifest",
            label_field=pred_key,
            progress=True,
        )
        return True


class UltralyticsObjectDetection:
    """Object detection using Ultralytics YOLO models with training and inference support."""

    def __init__(self, dataset, config):
        """Initialize with dataset, config, and setup paths for model and data."""
        self.dataset = dataset
        self.config = config
        self.ultralytics_data_path = os.path.join(
            config["export_dataset_root"], config["v51_dataset_name"]
        )

        self.hf_hub_model_id = (
            f"{HF_ROOT}/"
            + f"{config['v51_dataset_name']}_{config['model_name']}".replace("/", "_")
        )

        self.export_root = "output/models/ultralytics/"
        self.export_folder = os.path.join(
            self.export_root, self.config["v51_dataset_name"]
        )

        self.model_path = os.path.join(
            self.export_folder, self.config["model_name"], "weights", "best.pt"
        )

    @staticmethod
    def export_data(
        dataset, dataset_info, export_dataset_root, label_field="ground_truth"
    ):
        """Export dataset to YOLO format for Ultralytics training."""
        ultralytics_data_path = os.path.join(export_dataset_root, dataset_info["name"])
        # Delete export directory if it already exists
        if os.path.exists(ultralytics_data_path):
            shutil.rmtree(ultralytics_data_path)

        logging.info("Exporting data for training with Ultralytics")
        classes = dataset.distinct(f"{label_field}.detections.label")

        # Make directory
        os.makedirs(ultralytics_data_path, exist_ok=False)

        for split in ACCEPTED_SPLITS:
            split_view = dataset.match_tags(split)

            if split == "val" or split == "train":  # YOLO expects train and val
                split_view.export(
                    export_dir=ultralytics_data_path,
                    dataset_type=fo.types.YOLOv5Dataset,
                    label_field=label_field,
                    classes=classes,
                    split=split,
                )

    def train(self):
        """Train the YOLO model for object detection using Ultralytics and optionally upload to Hugging Face."""
        model = YOLO(self.config["model_name"], task="detect")
        # https://docs.ultralytics.com/modes/train/#train-settings

        # Use all available GPUs
        device = "0"  # Default to GPU 0
        if torch.cuda.device_count() > 1:
            device = ",".join(map(str, range(torch.cuda.device_count())))

        results = model.train(
            data=f"{self.ultralytics_data_path}/dataset.yaml",
            epochs=self.config["epochs"],
            project=self.export_folder,
            name=self.config["model_name"],
            patience=self.config["patience"],
            batch=self.config["batch_size"],
            imgsz=self.config["img_size"],
            multi_scale=self.config["multi_scale"],
            cos_lr=self.config["cos_lr"],
            seed=GLOBAL_SEED,
            optimizer="AdamW",  # "auto" as default
            pretrained=True,
            exist_ok=True,
            amp=True,
            device=device
        )
        metrics = model.val()
        logging.info(f"Model Performance: {metrics}")

        # Upload model to Hugging Face
        if HF_DO_UPLOAD:
            logging.info(f"Uploading model {self.model_path} to Hugging Face.")
            api = HfApi()
            api.create_repo(
                self.hf_hub_model_id, private=True, repo_type="model", exist_ok=True
            )
            api.upload_file(
                path_or_fileobj=self.model_path,
                path_in_repo="best.pt",
                repo_id=self.hf_hub_model_id,
                repo_type="model",
            )

    def inference(self, gt_field="ground_truth"):
        """Performs inference using YOLO model on a dataset, with options to evaluate results."""
        logging.info(f"Running inference on dataset {self.config['v51_dataset_name']}")
        inference_settings = self.config["inference_settings"]

        dataset_name = None
        model_name = self.config["model_name"]

        model_hf = inference_settings["model_hf"]
        if model_hf is not None:
            # Use model manually defined in config.
            # This way models can be used for inference which were trained on a different dataset
            dataset_name, _ = get_dataset_and_model_from_hf_id(model_hf)

            # Set up directories
            download_dir = os.path.join(
                self.export_root, dataset_name, model_name, "weights"
            )
            os.makedirs(os.path.join(download_dir), exist_ok=True)

            self.model_path = os.path.join(download_dir, "best.pt")

            # Create directories if they don't exist

            file_path = hf_hub_download(
                repo_id=model_hf,
                filename="best.pt",
                local_dir=download_dir,
            )
        else:
            # Automatically determine model based on dataset
            dataset_name = self.config["v51_dataset_name"]

            try:
                if os.path.exists(self.model_path):
                    file_path = self.model_path
                    logging.info(f"Loading model {model_name} from disk: {file_path}")
                else:
                    download_dir = self.model_path.replace("best.pt", "")
                    os.makedirs(download_dir, exist_ok=True)
                    logging.info(
                        f"Downloading model {self.hf_hub_model_id} from Hugging Face to {download_dir}"
                    )
                    file_path = hf_hub_download(
                        repo_id=self.hf_hub_model_id,
                        filename="best.pt",
                        local_dir=download_dir,
                    )
            except Exception as e:
                logging.error(f"Failed to load or download model: {str(e)}.")
                return False

        pred_key = f"pred_od_{model_name}-{dataset_name}"
        logging.info(f"Using model {self.model_path} for inference.")
        model = YOLO(self.model_path)

        detection_threshold = inference_settings["detection_threshold"]
        if inference_settings["inference_on_test"] is True:
            dataset_eval_view = self.dataset.match_tags("test")
            if len(dataset_eval_view) == 0:
                logging.error("Dataset misses split 'test'")
            dataset_eval_view.apply_model(
                model, label_field=pred_key, confidence_thresh=detection_threshold
            )
        else:
            self.dataset.apply_model(
                model, label_field=pred_key, confidence_thresh=detection_threshold
            )

        if inference_settings["do_eval"]:
            eval_key = f"eval_{self.config['model_name']}_{dataset_name}"

            if inference_settings["inference_on_test"] is True:
                dataset_view = self.dataset.match_tags(["test"])
            else:
                dataset_view = self.dataset

            dataset_view.evaluate_detections(
                pred_key,
                gt_field=gt_field,
                eval_key=eval_key,
                compute_mAP=True,
            )


def transform_batch_standalone(
    batch,
    image_processor,
    do_convert_annotations=True,
    return_pixel_mask=False,
):
    """Apply format annotations in COCO format for object detection task. Outside of class so it can be pickled."""
    images = []
    annotations = []

    for image_path, annotation in zip(batch["image_path"], batch["objects"]):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        images.append(image_np)

        coco_annotations = []
        for i, bbox in enumerate(annotation["bbox"]):

            # Conversion from HF dataset bounding boxes to DETR:
            # Input: HF dataset bbox is COCO (top_left_x, top_left_y, width, height) in absolute coordinates
            # Output:
            # DETR expects COCO (top_left_x, top_left_y, width, height) in absolute coordinates if 'do_convert_annotations == True'
            # DETR expects YOLO (center_x, center_y, width, height) in relative coordinates between [0,1] if 'do_convert_annotations == False'

            if do_convert_annotations == False:
                x, y, w, h = bbox
                img_height, img_width = image_np.shape[:2]
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                bbox = [center_x, center_y, width, height]

                # Ensure bbox values are within the expected range
                assert all(0 <= coord <= 1 for coord in bbox), f"Invalid bbox: {bbox}"

                logging.debug(
                    f"Converted {[x, y, w, h]} to {[center_x, center_y, width, height]} with 'do_convert_annotations' = {do_convert_annotations}"
                )

            coco_annotation = {
                "image_id": annotation["image_id"],
                "bbox": bbox,
                "category_id": annotation["category_id"][i],
                "area": annotation["area"][i],
                "iscrowd": 0,
            }
            coco_annotations.append(coco_annotation)
        detr_annotation = {
            "image_id": annotation["image_id"],
            "annotations": coco_annotations,
        }
        annotations.append(detr_annotation)

        # Apply the image processor transformations: resizing, rescaling, normalization
        result = image_processor(
            images=images, annotations=annotations, return_tensors="pt"
        )

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


class HuggingFaceObjectDetection:
    """Object detection using HuggingFace models with support for training and inference."""

    def __init__(
        self,
        dataset,
        config,
        output_model_path="./output/models/object_detection_hf",
        output_detections_path="./output/detections/",
        gt_field="ground_truth",
    ):
        """Initialize with dataset, config, and optional output paths."""
        self.dataset = dataset
        self.config = config
        self.model_name = config["model_name"]
        self.model_name_key = re.sub(r"[\W-]+", "_", self.model_name)
        self.dataset_name = config["v51_dataset_name"]
        self.do_convert_annotations = True  # HF can convert (top_left_x, top_left_y, bottom_right_x, bottom_right_y) in abs. coordinates to (x_min, y_min, width, height) in rel. coordinates https://github.com/huggingface/transformers/blob/v4.48.2/src/transformers/models/conditional_detr/image_processing_conditional_detr.py#L1497

        self.detections_root = os.path.join(
            output_detections_path, self.dataset_name, self.model_name_key
        )

        self.model_root = os.path.join(
            output_model_path, self.dataset_name, self.model_name_key
        )

        self.hf_hub_model_id = (
            f"{HF_ROOT}/" + f"{self.dataset_name}_{self.model_name}".replace("/", "_")
        )

        self.categories = dataset.distinct(f"{gt_field}.detections.label")
        self.id2label = {index: x for index, x in enumerate(self.categories, start=0)}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def collate_fn(self, batch):
        """Collate function for batching data during training and inference."""
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        if "pixel_mask" in batch[0]:
            data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
        return data

    def train(self, hf_dataset, overwrite_output=True):
        """Train models for object detection tasks with support for custom image sizes and transformations."""
        torch.cuda.empty_cache()
        img_size_target = self.config.get("image_size", None)
        if img_size_target is None:
            image_processor = AutoProcessor.from_pretrained(
                self.model_name,
                do_resize=False,
                do_pad=True,
                use_fast=True,
                do_convert_annotations=self.do_convert_annotations,
            )
        else:
            logging.warning(f"Resizing images to target size {img_size_target}.")
            image_processor = AutoProcessor.from_pretrained(
                self.model_name,
                do_resize=True,
                size={
                    "max_height": img_size_target[1],
                    "max_width": img_size_target[0],
                },
                do_pad=True,
                pad_size={"height": img_size_target[1], "width": img_size_target[0]},
                use_fast=True,
                do_convert_annotations=self.do_convert_annotations,
            )

        train_transform_batch = partial(
            transform_batch_standalone,
            image_processor=image_processor,
            do_convert_annotations=self.do_convert_annotations,
        )
        val_test_transform_batch = partial(
            transform_batch_standalone,
            image_processor=image_processor,
            do_convert_annotations=self.do_convert_annotations,
        )

        hf_dataset[Split.TRAIN] = hf_dataset[Split.TRAIN].with_transform(
            train_transform_batch
        )
        hf_dataset[Split.VALIDATION] = hf_dataset[Split.VALIDATION].with_transform(
            val_test_transform_batch
        )
        hf_dataset[Split.TEST] = hf_dataset[Split.TEST].with_transform(
            val_test_transform_batch
        )

        hf_model_config = AutoConfig.from_pretrained(self.model_name)
        hf_model_config_name = type(hf_model_config).__name__

        if type(hf_model_config) in AutoModelForObjectDetection._model_mapping:
            model = AutoModelForObjectDetection.from_pretrained(
                self.model_name,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )
        else:
            model = None
            logging.error(
                "Hugging Face AutoModel does not support " + str(type(hf_model_config))
            )

        if (
            overwrite_output == True
            and os.path.exists(self.model_root)
            and os.listdir(self.model_root)
        ):
            logging.warning(
                f"Training will overwrite existing results in {self.model_root}"
            )

        training_args = TrainingArguments(
            run_name=self.model_name,
            output_dir=self.model_root,
            overwrite_output_dir=overwrite_output,
            num_train_epochs=self.config["epochs"],
            fp16=True,
            per_device_train_batch_size=self.config["batch_size"],
            auto_find_batch_size=True,
            dataloader_num_workers=min(self.config["n_worker_dataloader"], NUM_WORKERS),
            learning_rate=self.config["learning_rate"],
            lr_scheduler_type="cosine",
            weight_decay=self.config["weight_decay"],
            max_grad_norm=self.config["max_grad_norm"],
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="best",
            save_total_limit=1,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            save_safetensors=False,  # Does not work with all models
            hub_model_id=self.hf_hub_model_id,
            hub_private_repo=True,
            push_to_hub=HF_DO_UPLOAD,
            seed=GLOBAL_SEED,
            data_seed=GLOBAL_SEED,
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=self.config["early_stop_patience"],
            early_stopping_threshold=self.config["early_stop_threshold"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset[Split.TRAIN],
            eval_dataset=hf_dataset[Split.VALIDATION],
            tokenizer=image_processor,
            data_collator=self.collate_fn,
            callbacks=[early_stopping_callback],
            # compute_metrics=eval_compute_metrics_fn,
        )

        logging.info(f"Starting training of model {self.model_name}.")
        trainer.train()
        if HF_DO_UPLOAD:
            trainer.push_to_hub()

        metrics = trainer.evaluate(eval_dataset=hf_dataset[Split.TEST])
        logging.info(f"Model training completed. Evaluation results: {metrics}")

    def inference(self, inference_settings, load_from_hf=True, gt_field="ground_truth"):
        """Performs model inference on a dataset, loading from Hugging Face or disk, and optionally evaluates detection results."""

        model_hf = inference_settings["model_hf"]
        dataset_name = None
        if model_hf is not None:
            self.hf_hub_model_id = model_hf
            dataset_name, model_name = get_dataset_and_model_from_hf_id(model_hf)
        else:
            dataset_name = self.dataset_name
        torch.cuda.empty_cache()
        # Load trained model from Hugging Face
        load_from_hf_successful = None
        if load_from_hf:
            try:
                logging.info(f"Loading model from Hugging Face: {self.hf_hub_model_id}")
                image_processor = AutoProcessor.from_pretrained(self.hf_hub_model_id)
                model = AutoModelForObjectDetection.from_pretrained(
                    self.hf_hub_model_id
                )
                load_from_hf_successful = True
            except Exception as e:
                load_from_hf_successful = False
                logging.warning(
                    f"Model {self.model_name} could not be loaded from Hugging Face {self.hf_hub_model_id}. Attempting loading from disk."
                )
        if load_from_hf == False or load_from_hf_successful == False:
            try:
                # Select folder in self.model_root that include 'checkpoint-'
                checkpoint_dirs = [
                    d
                    for d in os.listdir(self.model_root)
                    if "checkpoint-" in d
                    and os.path.isdir(os.path.join(self.model_root, d))
                ]

                if not checkpoint_dirs:
                    logging.error(
                        f"No checkpoint directory found in {self.model_root}!"
                    )
                    model_path = None
                else:
                    # Sort by modification time (latest first)
                    checkpoint_dirs.sort(
                        key=lambda d: os.path.getmtime(
                            os.path.join(self.model_root, d)
                        ),
                        reverse=True,
                    )

                    if len(checkpoint_dirs) > 1:
                        logging.warning(
                            f"Multiple checkpoint directories found: {checkpoint_dirs}. Selecting the latest one: {checkpoint_dirs[0]}."
                        )

                    selected_checkpoint = checkpoint_dirs[0]
                    logging.info(
                        f"Loading model from disk: {self.model_root}/{selected_checkpoint}"
                    )
                    model_path = os.path.join(self.model_root, selected_checkpoint)

                image_processor = AutoProcessor.from_pretrained(model_path)
                model = AutoModelForObjectDetection.from_pretrained(model_path)
            except Exception as e:
                logging.error(
                    f"Model {self.model_name} could not be loaded from folder {self.model_root}/{selected_checkpoint}. Inference not possible."
                )

        device, _, _ = get_backend()
        logging.info(f"Using device {device} for inference.")
        model = model.to(device)
        model.eval()

        pred_key = f"pred_od_{self.model_name_key}-{dataset_name}"

        if inference_settings["inference_on_test"] is True:
            INFERENCE_SPLITS = ["test"]
            dataset_eval_view = self.dataset.match_tags(INFERENCE_SPLITS)
        else:
            dataset_eval_view = self.dataset

        detection_threshold = inference_settings["detection_threshold"]

        with torch.amp.autocast("cuda"), torch.inference_mode():
            for sample in dataset_eval_view.iter_samples(progress=True, autosave=True):
                image_width = sample.metadata.width
                image_height = sample.metadata.height
                img_filepath = sample.filepath

                image = Image.open(img_filepath)
                inputs = image_processor(images=[image], return_tensors="pt")
                outputs = model(**inputs.to(device))
                target_sizes = torch.tensor([[image.size[1], image.size[0]]])

                results = image_processor.post_process_object_detection(
                    outputs, threshold=detection_threshold, target_sizes=target_sizes
                )[0]

                detections = []
                for score, label, box in zip(
                    results["scores"], results["labels"], results["boxes"]
                ):
                    # Bbox is in absolute coordinates x, y, x2, y2
                    box = box.tolist()
                    text_label = model.config.id2label[label.item()]

                    # Voxel51 requires relative coordinates between 0 and 1
                    top_left_x = box[0] / image_width
                    top_left_y = box[1] / image_height
                    box_width = (box[2] - box[0]) / image_width
                    box_height = (box[3] - box[1]) / image_height
                    detection = fo.Detection(
                        label=text_label,
                        bounding_box=[
                            top_left_x,
                            top_left_y,
                            box_width,
                            box_height,
                        ],
                        confidence=score.item(),
                    )
                    detections.append(detection)

                sample[pred_key] = fo.Detections(detections=detections)

        if inference_settings["do_eval"] is True:
            eval_key = re.sub(
                r"[\W-]+", "_", "eval_" + self.model_name + "_" + self.dataset_name
            )

            if inference_settings["inference_on_test"] is True:
                dataset_view = self.dataset.match_tags(["test"])
            else:
                dataset_view = self.dataset

            dataset_view.evaluate_detections(
                pred_key,
                gt_field=gt_field,
                eval_key=eval_key,
                compute_mAP=True,
            )


class CustomCoDETRObjectDetection:
    """Interface for running Co-DETR object detection model training and inference in containers"""

    def __init__(self, dataset, dataset_info, run_config):
        """Initialize Co-DETR interface with dataset and configuration"""
        self.root_codetr = "./custom_models/CoDETR/Co-DETR"
        self.root_codetr_models = "output/models/codetr"
        self.dataset = dataset
        self.dataset_name = dataset_info["name"]
        self.export_dir_root = run_config["export_dataset_root"]
        self.config_key = os.path.splitext(os.path.basename(run_config["config"]))[0]
        self.hf_repo_name = f"{HF_ROOT}/{self.dataset_name}_{self.config_key}"

    def convert_data(self):
        """Convert dataset to COCO format required by Co-DETR"""

        export_dir = os.path.join(self.export_dir_root, self.dataset_name, "coco")

        # Check if folder already exists
        if not os.path.exists(export_dir):
            # Make directory
            os.makedirs(export_dir, exist_ok=True)
            logging.info(f"Exporting data to {export_dir}")
            splits = [
                "train",
                "val",
                "test",
            ]  # CoDETR expects data in 'train' and 'val' folder
            for split in splits:
                split_view = self.dataset.match_tags(split)
                split_view.export(
                    dataset_type=fo.types.COCODetectionDataset,
                    data_path=os.path.join(export_dir, f"{split}2017"),
                    labels_path=os.path.join(
                        export_dir, "annotations", f"instances_{split}2017.json"
                    ),
                    label_field="ground_truth",
                )
        else:
            logging.warning(
                f"Folder {export_dir} already exists, skipping data export."
            )

    def update_config_file(self, dataset_name, config_file, max_epochs):
        """Update Co-DETR config file with dataset-specific parameters"""

        config_path = os.path.join(self.root_codetr, config_file)

        # Get classes from exported data
        annotations_json = os.path.join(
            self.export_dir_root,
            dataset_name,
            "coco/annotations/instances_train2017.json",
        )
        # Read the JSON file
        with open(annotations_json, "r") as file:
            data = json.load(file)

        # Extract the value associated with the key "categories"
        categories = data.get("categories")
        class_names = tuple(category["name"] for category in categories)
        num_classes = len(class_names)

        # Update configuration file
        # This assumes that 'classes = '('a','b',...)' are already defined and will be overwritten.
        with open(config_path, "r") as file:
            content = file.read()

        # Update the classes tuple
        content = re.sub(r"classes\s*=\s*\(.*?\)", f"classes = {class_names}", content)

        # Update all instances of num_classes
        content = re.sub(r"num_classes=\d+", f"num_classes={num_classes}", content)

        # Update all instances of max_epochs
        content = re.sub(r"max_epochs=\d+", f"max_epochs={max_epochs}", content)

        with open(config_path, "w") as file:
            file.write(content)

        logging.warning(
            f"Updated {config_path} with classes={class_names} and num_classes={num_classes} and max_epochs={max_epochs}"
        )

    def train(self, param_config, param_n_gpus, container_tool, param_function="train"):
        """Train Co-DETR model using containerized environment"""

        # Check if model already exists
        output_folder_codetr = os.path.join(self.root_codetr, "output")
        os.makedirs(output_folder_codetr, exist_ok=True)
        param_config_name = os.path.splitext(os.path.basename(param_config))[0]
        best_models_dir = os.path.join(output_folder_codetr, "best")
        os.makedirs(best_models_dir, exist_ok=True)
        # Best model files follow the naming scheme "config_dataset.pth"
        pth_model_files = (
            [f for f in os.listdir(best_models_dir) if f.endswith(".pth")]
            if os.path.exists(best_models_dir) and os.path.isdir(best_models_dir)
            else []
        )

        # Best model files are stored in the format "config_dataset.pth"
        matching_files = [
            f
            for f in pth_model_files
            if f.startswith(param_config_name)
            and self.dataset_name in f
            and f.endswith(".pth")
        ]
        if len(matching_files) > 0:
            logging.warning(
                f"Model {param_config_name} already trained on dataset {self.dataset_name}. Skipping training."
            )
            if len(matching_files) > 1:
                logging.warning(f"Multiple weights found: {matching_files}")
        else:
            logging.info(
                f"Launching training for Co-DETR config {param_config} and dataset {self.dataset_name}."
            )
            volume_data = os.path.join(self.export_dir_root, self.dataset_name)

            # Train model, store checkpoints in 'output_folder_codetr'
            train_result = self._run_container(
                volume_data=volume_data,
                param_function=param_function,
                param_config=param_config,
                param_n_gpus=param_n_gpus,
                container_tool=container_tool,
            )

            # Find the best_bbox checkpoint file
            checkpoint_files = [
                f
                for f in os.listdir(output_folder_codetr)
                if "best_bbox" in f and f.endswith(".pth")
            ]
            if not checkpoint_files:
                logging.error(
                    "Co-DETR was not trained, model pth file missing. No checkpoint file with 'best_bbox' found."
                )
            else:
                if len(checkpoint_files) > 1:
                    logging.warning(
                        f"Found {len(checkpoint_files)} checkpoint files. Selecting {checkpoint_files[0]}."
                    )
                checkpoint = checkpoint_files[0]
                checkpoint_path = os.path.join(output_folder_codetr, checkpoint)
                logging.info("Co-DETR was trained successfully.")

                # Upload best model to Hugging Face
                if HF_DO_UPLOAD == True:
                    logging.info("Uploading Co-DETR model to Hugging Face.")
                    api = HfApi()
                    api.create_repo(
                        self.hf_repo_name,
                        private=True,
                        repo_type="model",
                        exist_ok=True,
                    )
                    api.upload_file(
                        path_or_fileobj=checkpoint_path,
                        path_in_repo="model.pth",
                        repo_id=self.hf_repo_name,
                        repo_type="model",
                    )

                # Move best model file and clear output folder
                self._run_container(
                    volume_data=volume_data,
                    param_function="clear-output",
                    param_config=param_config,
                    param_dataset_name=self.dataset_name,
                    container_tool=container_tool,
                )

    @staticmethod
    def _find_file_iteratively(start_path, filename):
        """Direct access or recursively search for a file in a directory structure."""
        # Convert start_path to a Path object
        start_path = Path(start_path)

        # Check if the file exists in the start_path directly (very fast)
        file_path = start_path / filename
        if file_path.exists():
            return str(file_path)

        # Start with the highest directory and go up iteratively
        current_dir = start_path
        checked_dirs = set()

        while current_dir != current_dir.root:
            # Check if the file is in the current directory
            file_path = current_dir / filename
            if file_path.exists():
                return str(file_path)

            # If we haven't checked the sibling directories, check them as well
            parent_dir = current_dir.parent
            if parent_dir not in checked_dirs:
                # Check sibling directories
                for sibling in parent_dir.iterdir():
                    if sibling != current_dir and sibling.is_dir():
                        sibling_file_path = sibling / filename
                        if sibling_file_path.exists():
                            return str(sibling_file_path)
                checked_dirs.add(parent_dir)

            # Otherwise, go one level up
            current_dir = current_dir.parent

        # If file is not found after traversing all levels, return None
        logging.error(f"File {filename} could not be found.")
        return None

    def run_inference(
        self,
        dataset,
        param_config,
        param_n_gpus,
        container_tool,
        inference_settings,
        param_function="inference",
        inference_output_folder="custom_models/CoDETR/Co-DETR/output/inference/",
        gt_field="ground_truth",
    ):
        """Run inference using trained Co-DETR model and convert results to FiftyOne format"""

        logging.info(f"Launching inference for Co-DETR config {param_config}.")
        volume_data = os.path.join(self.export_dir_root, self.dataset_name)

        if inference_settings["inference_on_test"] is True:
            folder_inference = os.path.join("coco", "test2017")
        else:
            folder_inference = os.path.join("coco")

        # Get model from Hugging Face
        dataset_name = None
        config_key = None
        try:
            if inference_settings["model_hf"] is None:
                hf_path = self.hf_repo_name
            else:
                hf_path = inference_settings["model_hf"]

            dataset_name, config_key = get_dataset_and_model_from_hf_id(hf_path)

            download_folder = os.path.join(
                self.root_codetr_models, dataset_name, config_key
            )

            logging.info(
                f"Downloading model {hf_path} from Hugging Face into {download_folder}"
            )
            os.makedirs(download_folder, exist_ok=True)

            file_path = hf_hub_download(
                repo_id=hf_path,
                filename="model.pth",
                local_dir=download_folder,
            )
        except Exception as e:
            logging.error(f"An error occured during model download: {e}")

        model_path = os.path.join(dataset_name, config_key, "model.pth")
        logging.info(f"Starting inference for model {model_path}")

        inference_result = self._run_container(
            volume_data=volume_data,
            param_function=param_function,
            param_config=param_config,
            param_n_gpus=param_n_gpus,
            container_tool=container_tool,
            param_inference_dataset_folder=folder_inference,
            param_inference_model_checkpoint=model_path,
        )

        # Convert results from JSON output into V51 dataset
        # Files follow format inference_results_{timestamp}.json (run_inference.py)
        os.makedirs(inference_output_folder, exist_ok=True)
        output_files = [
            f
            for f in os.listdir(inference_output_folder)
            if f.startswith("inference_results_") and f.endswith(".json")
        ]
        logging.debug(f"Found files with inference content: {output_files}")

        if not output_files:
            logging.error(
                f"No inference result files found in {inference_output_folder}"
            )

        # Get full path for each file
        file_paths = [os.path.join(inference_output_folder, f) for f in output_files]

        # Extract timestamp from the filename and sort based on the timestamp
        file_paths_sorted = sorted(
            file_paths,
            key=lambda f: datetime.datetime.strptime(
                f.split("_")[-2] + "_" + f.split("_")[-1].replace(".json", ""),
                "%Y%m%d_%H%M%S",
            ),
            reverse=True,
        )

        # Use the most recent file based on timestamp
        latest_file = file_paths_sorted[0]
        logging.info(f"Using inference results from: {latest_file}")
        with open(latest_file, "r") as file:
            data = json.load(file)

        # Get conversion for annotated classes
        annotations_path = os.path.join(
            volume_data, "coco", "annotations", "instances_train2017.json"
        )

        with open(annotations_path, "r") as file:
            data_annotations = json.load(file)

        class_ids_and_names = [
            (category["id"], category["name"])
            for category in data_annotations["categories"]
        ]

        # Match sample filepaths (from exported Co-DETR COCO format) to V51 filepaths
        sample = dataset.first()
        root_dir_samples = sample.filepath

        # Convert results into V51 file format
        detection_threshold = inference_settings["detection_threshold"]
        pred_key = f"pred_od_{config_key}-{dataset_name}"
        for key, value in tqdm(data.items(), desc="Processing Co-DETR detection"):
            try:
                # Get filename
                filepath = CustomCoDETRObjectDetection._find_file_iteratively(
                    root_dir_samples, os.path.basename(key)
                )
                sample = dataset[filepath]

                img_width = sample.metadata.width
                img_height = sample.metadata.height

                detections_v51 = []
                for class_id, class_detections in enumerate(data[key]):  # Starts with 0
                    if len(class_detections) > 0:
                        objects_class = class_ids_and_names[class_id]
                        for detection in class_detections:
                            confidence = detection[4]
                            detection_v51 = fo.Detection(
                                label=objects_class[1],
                                bounding_box=[
                                    detection[0] / img_width,
                                    detection[1] / img_height,
                                    (detection[2] - detection[0]) / img_width,
                                    (detection[3] - detection[1]) / img_height,
                                ],
                                confidence=confidence,
                            )
                            if confidence >= detection_threshold:
                                detections_v51.append(detection_v51)

                sample[pred_key] = fo.Detections(detections=detections_v51)
                sample.save()
            except Exception as e:
                logging.error(
                    f"An error occured during the conversion of Co-DETR inference results to the V51 dataset: {e}"
                )

        # Run V51 evaluation
        if inference_settings["do_eval"] is True:
            eval_key = pred_key.replace("pred_", "eval_").replace("-", "_")

            if inference_settings["inference_on_test"] is True:
                dataset_view = dataset.match_tags(["test"])
            else:
                dataset_view = dataset

            logging.info(
                f"Starting evaluation for {pred_key} in evaluation key {eval_key}."
            )
            dataset_view.evaluate_detections(
                pred_key,
                gt_field=gt_field,
                eval_key=eval_key,
                compute_mAP=True,
            )

    def _run_container(
        self,
        volume_data,
        param_function,
        param_config="",
        param_n_gpus="1",
        param_dataset_name="",
        param_inference_dataset_folder="",
        param_inference_model_checkpoint="",
        image="dbogdollresearch/codetr",
        workdir="/launch",
        container_tool="docker",
    ):
        """Execute Co-DETR container with specified parameters using Docker or Singularity"""

        try:
            # Convert relative paths to absolute paths (necessary under WSL2)
            root_codetr_abs = os.path.abspath(self.root_codetr)
            volume_data_abs = os.path.abspath(volume_data)
            root_codetr_models_abs = os.path.abspath(self.root_codetr_models)

            # Check if using Docker or Singularity and define the appropriate command
            if container_tool == "docker":
                command = [
                    "docker",
                    "run",
                    "--gpus",
                    "all",
                    "--workdir",
                    workdir,
                    "--volume",
                    f"{root_codetr_abs}:{workdir}",
                    "--volume",
                    f"{volume_data_abs}:{workdir}/data:ro",
                    "--volume",
                    f"{root_codetr_models_abs}:{workdir}/hf_models:ro",
                    "--shm-size=8g",
                    image,
                    param_function,
                    param_config,
                    param_n_gpus,
                    param_dataset_name,
                    param_inference_dataset_folder,
                    param_inference_model_checkpoint,
                ]
            elif container_tool == "singularity":
                command = [
                    "singularity",
                    "run",
                    "--nv",
                    "--pwd",
                    workdir,
                    "--bind",
                    f"{self.root_codetr}:{workdir}",
                    "--bind",
                    f"{volume_data}:{workdir}/data:ro",
                    "--bind",
                    f"{self.root_codetr_models}:{workdir}/hf_models:ro",
                    f"docker://{image}",
                    param_function,
                    param_config,
                    param_n_gpus,
                    param_dataset_name,
                    param_inference_dataset_folder,
                    param_inference_model_checkpoint,
                ]
            else:
                raise ValueError(
                    f"Invalid container tool specified: {container_tool}. Choose 'docker' or 'singularity'."
                )

            # Start the process and stream outputs to the console
            logging.info(f"Launching terminal command {command}")
            with subprocess.Popen(
                command, stdout=sys.stdout, stderr=sys.stderr, text=True
            ) as proc:
                proc.wait()  # Wait for the process to complete
            return True
        except Exception as e:
            logging.error(f"Error during Co-DETR container run: {e}")
            return False
