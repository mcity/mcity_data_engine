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
from typing import Union

import albumentations as A
import fiftyone as fo
import numpy as np
import psutil
import torch
import torch.multiprocessing as mp
import wandb
from accelerate.test_utils.testing import get_backend
from datasets import Split
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

from config.config import (
    GLOBAL_SEED,
    HF_DO_UPLOAD,
    HF_ROOT,
    NUM_WORKERS,
    WANDB_ACTIVE,
    WORKFLOWS,
)
from utils.logging import configure_logging


# Handling timeouts
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Dataloader creation timed out")


class ZeroShotInferenceCollateFn:
    def __init__(
        self,
        hf_model_config_name,
        hf_processor,
        batch_size,
        object_classes,
        batch_classes,
    ):
        try:
            self.hf_model_config_name = hf_model_config_name
            self.processor = hf_processor
            self.batch_size = batch_size
            self.object_classes = object_classes
            self.batch_classes = batch_classes
        except Exception as e:
            logging.error(f"Error in collate init of DataLoader: {e}")

    def __call__(self, batch):
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
    def __init__(
        self,
        dataset_torch: torch.utils.data.Dataset,
        dataset_info,
        config,
        detections_path="./output/detections/",
        log_root="./logs/",
    ):
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
                    dataset_v51.add_sample_field(
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
        # Measure the sizes of multiple result queues (one per worker process)
        # Logging
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
        result_queues_sizes,
        largest_queue_index,
        inference_finished,
        queue_warning_threshold=5,
    ):
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
        if WANDB_ACTIVE:
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

            # if results_queue.qsize() > queue_warning_threshold:
            # logging.warning(f"Queue size of {results_queue.qsize()}. Consider increasing number of post-processing workers.")
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
                    # result = results_queue.get(block=True, timeout=0.5)

                    processing_successful = self.process_outputs(
                        dataset_v51, result, self.object_classes
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

    def process_outputs(
        self, dataset_v51, result, object_classes, detection_threshold=0.2
    ):
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
        # Populate dataset with evaluation results (if ground_truth available)
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


class HuggingFaceObjectDetection:
    def __init__(
        self,
        dataset,
        config,
        output_model_path="./output/models/object_detection_hf",
        output_detections_path="./output/detections/",
    ):
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

        self.categories = dataset.default_classes
        self.id2label = {index: x for index, x in enumerate(self.categories, start=0)}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def transform_batch(
        self,
        batch,
        image_processor,
        return_pixel_mask=False,
    ):
        """Apply format annotations in COCO format for object detection task"""
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

                if self.do_convert_annotations == False:
                    x, y, w, h = bbox
                    img_height, img_width = image_np.shape[:2]
                    center_x = (x + w / 2) / img_width
                    center_y = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height
                    bbox = [center_x, center_y, width, height]

                    # Ensure bbox values are within the expected range
                    assert all(
                        0 <= coord <= 1 for coord in bbox
                    ), f"Invalid bbox: {bbox}"

                    logging.debug(
                        f"Converted {[x, y, w, h]} to {[center_x, center_y, width, height]} with 'do_convert_annotations' = {self.do_convert_annotations}"
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

    def collate_fn(self, batch):
        """
        Collates a batch of data into a single dictionary suitable for model input.

        Args:
            batch (list of dict): A list of dictionaries where each dictionary contains
                                  the keys "pixel_values", "labels", and optionally "pixel_mask".

        Returns:
            dict: A dictionary with the following keys:
                - "pixel_values" (torch.Tensor): A tensor containing stacked pixel values from the batch.
                - "labels" (list): A list of labels from the batch.
                - "pixel_mask" (torch.Tensor, optional): A tensor containing stacked pixel masks from the batch,
                                                         if "pixel_mask" is present in the input batch.
        """
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        if "pixel_mask" in batch[0]:
            data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
        return data

    def train(self, hf_dataset, overwrite_output=True):
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
            self.transform_batch,
            # transform=None,  # TODO train_augmentation_and_transform,
            image_processor=image_processor,
        )
        val_test_transform_batch = partial(
            self.transform_batch,
            # transform=None,  # TODO validation_transform,
            image_processor=image_processor,
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

        if overwrite_output == True:
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

    def inference(self, detection_threshold=0.2, load_from_hf=True):
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

        pred_key = re.sub(r"[\W-]+", "_", "pred_od_" + self.model_name)

        # TODO Improve GPU utilization similar to ZeroShotInference
        with torch.amp.autocast("cuda"), torch.inference_mode():
            for sample in self.dataset.iter_samples(progress=True, autosave=True):
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


class CustomCoDETRObjectDetection:
    def __init__(self, dataset, dataset_name, splits, export_dir_root):
        self.root_codetr = "./custom_models/CoDETR/Co-DETR"
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.splits = splits

        if "train" not in splits or "test" not in splits:
            logging.error("Co-DETR training requires a train and test split.")

        self.export_dir_root = export_dir_root

    def convert_data(self):

        export_dir = os.path.join(self.export_dir_root, self.dataset_name, "coco")

        # Check if folder already exists
        if not os.path.exists(export_dir):
            logging.info(f"Exporting data to {export_dir}")
            splits = ["train", "val"]  # Expects train and val tags in FiftyOne dataset
            for split in splits:
                split_view = self.dataset.match_tags(split)
                # Utilize expected 'val' split for export
                if split == "test":
                    split = "val"
                split_view.export(
                    dataset_type=fo.types.COCODetectionDataset,
                    data_path=os.path.join(export_dir, f"{split}2017"),
                    labels_path=os.path.join(
                        export_dir, "annotations", f"instances_{split}2017.json"
                    ),
                    label_field="ground_truth",
                )
        else:
            logging.info(f"Folder {export_dir} already exists, skipping data export.")

    def update_config_file(self, dataset_name, config_file, max_epochs=12):
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
        content = re.sub(r"classes = \([^\)]+\)", f"classes = {class_names}", content)

        # Update all instances of num_classes
        content = re.sub(r"num_classes=\d+", f"num_classes={num_classes}", content)

        # Update all instances of max_epochs
        content = re.sub(r"max_epochs=\d+", f"max_epochs={max_epochs}", content)

        with open(config_path, "w") as file:
            file.write(content)

        logging.info(
            f"Updated {config_path} with classes={class_names} and num_classes={num_classes}"
        )

    def train(self, param_config, param_n_gpus, container_tool, param_function="train"):

        volume_data = os.path.join(self.export_dir_root, self.dataset_name)
        train_result = self._run_container(
            volume_data=volume_data,
            param_function=param_function,
            param_config=param_config,
            param_n_gpus=param_n_gpus,
            container_tool=container_tool,
        )

        # Find the best_bbox checkpoint file
        output_folder_codetr = os.path.join(self.root_codetr, "output")
        checkpoint_files = [
            f
            for f in os.listdir(output_folder_codetr)
            if "best_bbox" in f and f.endswith(".pth")
        ]
        if not checkpoint_files:
            logging.error(
                "CoDETR was not trained, model pth file missing. No checkpoint file with 'best_bbox' found."
            )
        else:
            if len(checkpoint_files) > 1:
                logging.warning(
                    f"Found {len(checkpoint_files)} checkpoint files. Selecting {checkpoint_files[0]}."
                )
            checkpoint = checkpoint_files[0]
            checkpoint_path = os.path.join(output_folder_codetr, checkpoint)

            # Train model, store checkpoints in 'output_folder_codetr'
            train_result = self._run_container(
                volume_data=volume_data,
                param_function=param_function,
                param_config=param_config,
                param_n_gpus=param_n_gpus,
                container_tool=container_tool,
            )

            # Move model file
            param_config_clean = os.path.splitext(os.path.basename(param_config))[0]
            output_dir = os.path.join(
                "output",
                "models",
                "object_detection",
                "codetr",
                self.dataset_name,
                param_config_clean,
            )
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(checkpoint_path, output_dir)

    def run_inference(
        self,
        volume_data,
        param_config,
        param_n_gpus,
        container_tool,
        param_function="inference",
    ):
        inference_result = self._run_container(
            volume_data=volume_data,
            param_function=param_function,
            param_config=param_config,
            param_n_gpus=param_n_gpus,
            container_tool=container_tool,
        )

    def _run_container(
        self,
        volume_data,
        param_function,
        param_config,
        param_n_gpus="1",
        image="dbogdollresearch/codetr",
        workdir="/launch",
        container_tool="docker",
    ):
        try:
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
                    f"{self.root_codetr}:{workdir}",
                    "--volume",
                    f"{volume_data}:{workdir}/data",
                    "--shm-size=8g",
                    image,
                    param_function,
                    param_config,
                    param_n_gpus,
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
                    f"{volume_data}:{workdir}/data",
                    f"docker://{image}",
                    param_function,
                    param_config,
                    param_n_gpus,
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
