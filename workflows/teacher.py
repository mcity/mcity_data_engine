import datetime
import logging
import os
import queue
import re
import time
from difflib import get_close_matches
from functools import partial
from typing import List, Union

import fiftyone as fo
import numpy as np
import psutil
import torch
import torch.multiprocessing as mp
import wandb
from datasets import Split
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForObjectDetection,
                          AutoModelForZeroShotObjectDetection, AutoProcessor,
                          EarlyStoppingCallback, Trainer, TrainingArguments)
from transformers.utils.generic import is_torch_device
from transformers.utils.import_utils import (is_torch_available,
                                             requires_backends)

from config.config import NUM_WORKERS
from utils.data_loader import FiftyOneTorchDatasetCOCO, TorchToHFDatasetCOCO
from utils.logging import configure_logging


def transform_batch(examples, image_processor, return_pixel_mask=False):
    """Apply format annotations in COCO format for object detection task"""
    # TODO Include augmentations
    # https://albumentations.ai/docs/integrations/fiftyone/

    images = []
    annotations = []

    for image_path, annotation in zip(examples["image"], examples["target"]):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        images.append(image_np)

        # Annotation needs to be in COCO style annotation per bounding box
        coco_annotations = []
        for i, bbox in enumerate(annotation["bbox"]):

            # Convert bbox x_min, y_min, w, h to YOLO format x_center, y_center, w, h
            bbox[0] = bbox[0] + bbox[2] / 2.0
            bbox[1] = bbox[1] + bbox[3] / 2.0

            # Ensure bbox values are within the expected range
            assert all(0 <= coord <= 1 for coord in bbox), f"Invalid bbox: {bbox}"

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


class ZeroShotInferenceCollateFn:
    def __init__(self, hf_model_config_name, hf_processor, batch_size, object_classes, batch_classes):
        self.hf_model_config_name = hf_model_config_name
        self.processor = hf_processor
        self.batch_size = batch_size
        self.object_classes = object_classes
        self.batch_classes = batch_classes

    def __call__(self, batch):
        try:
            images, labels = zip(*batch)
            target_sizes = [tuple(img.shape[1:]) for img in images]

            # Adjustments for final batch
            n_images = len(images)
            if n_images < self.batch_size:
                self.batch_classes = ZeroShotTeacher._get_batch_classes(self.hf_model_config_name, self.object_classes, n_images)                

            # Apply PIL transformation for specific models
            if self.hf_model_config_name == "OmDetTurboConfig":
                images = [to_pil_image(image) for image in images]

            inputs = self.processor(
                text=self.batch_classes,
                images=images,
                return_tensors="pt",
                padding=True    # Allow for differently sized images
            )

            return inputs, labels, target_sizes, self.batch_classes
        except Exception as e:
            logging.error(f"Error in collate function of DataLoader: {e}")

class ZeroShotTeacher:
    def __init__(self, dataset_torch: torch.utils.data.Dataset, dataset_info, config, detections_path="./output/detections/"):
        # Can only store objects that are pickable for multiprocessing
        self.dataset_torch = dataset_torch
        self.dataset_info = dataset_info
        self.dataset_name = dataset_info["name"]
        self.object_classes = config["object_classes"]
        self.detection_threshold = config["detection_threshold"]
        self.detections_root = os.path.join(detections_path, self.dataset_name)

        logging.info(f"Zero-shot models will look for {self.object_classes}")

    @staticmethod   # Also utilized in ZeroShotInferenceCollateFn
    def _get_batch_classes(hf_model_config_name, object_classes, batch_size):
        if hf_model_config_name == "GroundingDinoConfig":
                classes = " . ".join(object_classes) + " . "
                batch_classes = [classes] * batch_size
        elif hf_model_config_name == "OmDetTurboConfig":
            batch_classes = [object_classes] * batch_size
        elif hf_model_config_name == "Owlv2Config" or hf_model_config_name == "OwlViTConfig":
            batch_classes = object_classes * batch_size
        else:
            logging.error(f"Invalid model name: {hf_model_config_name}")
        
        return batch_classes

    def exclude_stored_predictions(self, dataset_v51: fo.Dataset, config):
        dataset_schema = dataset_v51.get_field_schema()
        models_splits_dict = {}
        for model_name, value in config["hf_models_zeroshot_objectdetection"].items():
            model_name_key = re.sub(r"[\W-]+", "_", model_name)
            pred_key = re.sub(r"[\W-]+", "_", "pred_" + model_name)
            # Check if data already stored in V51 dataset
            if pred_key in dataset_schema:
                logging.info(f"Model {model_name} predictions already stored in Voxel51 dataset.")
            # Check if data already stored on disk
            elif os.path.isdir(os.path.join(self.detections_root, model_name_key)):
                try:
                    temp_dataset = fo.Dataset.from_dir(
                        dataset_dir=self.detections_root,
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
                    logging.info(f"Model {model_name} predictions loaded from disk.")
                except Exception as e:
                    logging.error(
                        f"Data in {self.detections_root} could not be loaded. Error: {e}"
                    )
                finally:
                    fo.delete_dataset("temp_dataset")
            # Assign model to be run             
            else:
                models_splits_dict[model_name] = value

        print(f"Models to be run: {models_splits_dict}")
        return models_splits_dict

    # Worker functions
    def process_outputs_worker(self, results_queue, inference_finished, queue_warning_threshold=5):
        configure_logging()
        logging.info(f"Process ID: {os.getpid()}. Results processing process started")    
        dataset_v51 = fo.load_dataset(self.dataset_name)
        processing_successful = None
        while True:
            if results_queue.qsize() > queue_warning_threshold:
                logging.warning(f"Queue size of {results_queue.qsize()}. Consider increasing number of post-processing workers.")
            # Exit only when inference is finished and the queue is empty
            if inference_finished.value and results_queue.empty():
                logging.info(f"Post-processing worker {os.getpid()} has finished all outputs.")
                break
            # Process results from the queue if available
            if not results_queue.empty():
                try:
                    result = results_queue.get_nowait()
                    processing_successful = self.process_outputs(dataset_v51, result, self.object_classes)
                except Exception as e:
                    continue
            else:
                continue
        return processing_successful    # Return last processing status
    
    def gpu_worker(self, gpu_id, cpu_cores, task_queue, results_queue, done_event, post_processing_finished, set_cpu_affinity=False):
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

        # Set WandB directory
        root_log_dir = "logs/tensorboard/teacher_zeroshot"  # FIXME Use paths from init
        
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
                        task_metadata = task_queue.get(timeout=5)  # Timeout to prevent indefinite blocking
                    except Exception as e:
                        break  # Exit if no more tasks

                    run_successful = self.model_inference(task_metadata, device, self.dataset_torch, self.object_classes, results_queue, root_log_dir)
                    logging.info(f"Worker for GPU {gpu_id} finished run successful: {run_successful}")
                else:
                    continue
        return run_successful   # Return last processing status

    # Functionality functions
    def model_inference(self, metadata: dict, device: str, dataset: torch.utils.data.Dataset, object_classes: list, results_queue: Union[queue.Queue, mp.Queue], root_log_dir: str, num_workers: int = 4, persistent_workers: bool = False, prefetch_factor: int = 4, detection_threshold: int = 0.2):
        writer = None
        run_successful = True
        model = None

        try:
            # Metadata
            run_id = metadata["run_id"]
            model_name = metadata["model_name"]
            dataset_name = metadata["dataset_name"]
            is_subset = metadata["is_subset"]
            batch_size = metadata["batch_size"]
            
            logging.info(f"Process ID: {os.getpid()}, Run ID: {run_id}, Device: {device}, Model: {model_name}")        

            # Load the model
            logging.info(f"Loading model {model_name}")
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            model = model.to(device)
            model.eval()
            hf_model_config = AutoConfig.from_pretrained(model_name)
            hf_model_config_name = type(hf_model_config).__name__
            logging.info(f"Loaded model type {hf_model_config_name}")
            
            batch_classes = ZeroShotTeacher._get_batch_classes(hf_model_config_name=hf_model_config_name, object_classes=object_classes, batch_size=batch_size)

            # Dataloader
            logging.info("Generating dataloader")
            if is_subset:
                chunk_index_start = metadata["chunk_index_start"]
                chunk_index_end = metadata["chunk_index_end"]
                logging.info(f"Length of dataset: {len(dataset)}")
                logging.info(f"Subset start index: {chunk_index_start}")
                logging.info(f"Subset stop index: {chunk_index_end}")
                dataset = Subset(dataset, range(chunk_index_start, chunk_index_end))
            
            zero_shot_inference_preprocessing = ZeroShotInferenceCollateFn(hf_model_config_name=hf_model_config_name, hf_processor=processor, object_classes=object_classes, batch_size=batch_size, batch_classes=batch_classes)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers, pin_memory=True, prefetch_factor=prefetch_factor, collate_fn=zero_shot_inference_preprocessing)
            
            # Logging
            experiment_name = f"{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{device}"
            log_directory = os.path.join(root_log_dir, dataset_name, experiment_name)
            wandb.tensorboard.patch(root_logdir=log_directory)
            wandb.init(
                name=f"{model_name}_{device}",
                job_type="inference",
                project="Zero Shot Teacher",
                config=metadata,
            )
            writer = SummaryWriter(log_dir=log_directory)

            # Inference Loop
            logging.info("Starting inference")
            n_processed_images = 0
            for inputs, labels, target_sizes, batch_classes in dataloader:
                time_start = time.time()
                n_images = len(labels)
                inputs = inputs.to(device)  # TODO Add non_blocking=True after HF PR is merged: https://github.com/huggingface/transformers/pull/34883

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
                    "batch_classes": batch_classes
                }

                results_queue.put(result)

                time_end = time.time()
                duration = time_end - time_start
                batches_per_second = 1 / duration
                frames_per_second = batches_per_second * n_images
                n_processed_images += n_images
                writer.add_scalar(f'inference/frames_per_second', frames_per_second, n_processed_images)    
            
            # Flawless execution
            wandb.finish(exit_code=0)
        
        except Exception as e:
            run_successful = False
            wandb.finish(exit_code=1)
            logging.error(f"Error in Process {os.getpid()}: {e}")
        finally:
            wandb.tensorboard.unpatch()
            if writer:
                writer.close()
            torch.cuda.empty_cache()
            return run_successful


        
    def process_outputs(self, dataset_v51, result, object_classes, detection_threshold=0.2):
        processing_successful = True
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
                results = processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=detection_threshold,
                    target_sizes=target_sizes,
                )
            elif hf_model_config_name == "OmDetTurboConfig":
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    classes=batch_classes,
                    score_threshold=detection_threshold,
                    nms_threshold=detection_threshold,
                    target_sizes=target_sizes,
                )
            else:
                logging.error(f"Invalid model name: {hf_model_config_name}")

            if not len(results) == len(target_sizes) == len(labels):
                logging.error(f"Lengths of results, target_sizes, and labels do not match: {len(results)}, {len(target_sizes)}, {len(labels)}")
            for result, size, target in zip(results, target_sizes, labels):
                boxes, scores = result["boxes"], result["scores"]

                img_height = size[0]
                img_width = size[1]

                if "labels" in result:
                    labels = result["labels"]
                elif "classes" in result:  # OmDet deviates from the other models
                    labels = result["classes"]

                detections = []
                for box, score, label in zip(boxes, scores, labels):
                    processing_successful = True
                    if hf_model_config_name == "GroundingDinoConfig":
                        # Outputs do not comply with given labels
                        # Grounding DINO outputs multiple pairs of object boxes and noun phrases for a given (Image, Text) pair
                        # There can be either multiple labels per output ("bike van"), incomplete ones ("motorcyc"), or broken ones ("##cic")
                        processed_label = label.split()[0]          # Assume first output is the best output
                        if processed_label in object_classes:
                            label = processed_label
                            top_left_x = box[0].item()
                            top_left_y = box[1].item()
                            box_width = (box[2] - box[0]).item()
                            box_height = (box[3] - box[1]).item()
                        else:
                            matches = get_close_matches(processed_label, object_classes, n=1, cutoff=0.6)
                            selected_label = matches[0] if matches else None
                            if selected_label:
                                processed_label = selected_label
                            else:
                                logging.warning(object_classes)
                                logging.warning(f"Skipped detection with {hf_model_config_name} due to unclear output: {label}")
                                processing_successful = False
                                

                    elif hf_model_config_name in [
                        "Owlv2Config",
                        "OwlViTConfig",
                    ]:
                        label = object_classes[label]   # Label is ID
                        top_left_x = box[0].item() / img_width
                        top_left_y = box[1].item() / img_height
                        box_width = (box[2].item() - box[0].item()) / img_width
                        box_height = (box[3].item() - box[1].item()) / img_height

                    elif hf_model_config_name == "OmDetTurboConfig":
                        top_left_x = box[0].item() / img_width
                        top_left_y = box[1].item() / img_height
                        box_width = (box[2].item() - box[0].item()) / img_width
                        box_height = (box[3].item() - box[1].item()) / img_height

                    if processing_successful:   # Skip GroundingDinoConfig labels that could not be processed
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
                pred_key = re.sub(r"[\W-]+", "_", "pred_" + model_name)
                sample = dataset_v51[target["image_id"]]
                sample[pred_key] = fo.Detections(detections=detections)
                sample.save()
        except Exception as e:
            logging.error(f"Error in processing outputs: {e}")
            processing_successful = False
        finally:
            return processing_successful

class Teacher:
    def __init__(self, dataset, config=None, detections_path="./output/detections/"):
        self.dataset = dataset
        self.config = config
        self.model_name = config["model_name"]
        self.model_name_key = re.sub(r"[\W-]+", "_", self.model_name)
        self.dataset_name = config["v51_dataset_name"]

        self.detections_root = os.path.join(
            detections_path, self.dataset_name, self.model_name_key
        )
        self.categories = dataset.default_classes
        self.id2label = {index: x for index, x in enumerate(self.categories, start=0)}
        self.label2id = {v: k for k, v in self.id2label.items()}

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

    def train(self):
        pytorch_dataset = FiftyOneTorchDatasetCOCO(self.dataset)
        pt_to_hf_converter = TorchToHFDatasetCOCO(pytorch_dataset)
        hf_dataset = pt_to_hf_converter.convert()

        image_processor = AutoProcessor.from_pretrained(
            self.model_name,
            do_resize=False,
            do_pad=False,  # Assumes all images have the same size
            do_convert_annotations=False,  # expects YOLO (center_x, center_y, width, height) between [0,1]
        )

        hf_model_config = AutoConfig.from_pretrained(self.model_name)
        train_transform_batch = partial(
            transform_batch,
            image_processor=image_processor,
        )
        validation_transform_batch = partial(
            transform_batch,
            image_processor=image_processor,
        )

        hf_dataset[Split.TRAIN] = hf_dataset[Split.TRAIN].with_transform(
            train_transform_batch
        )
        hf_dataset[Split.VALIDATION] = hf_dataset[Split.VALIDATION].with_transform(
            validation_transform_batch
        )

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
                "HuggingFace AutoModel does not support " + str(type(hf_model_config))
            )

        training_args = TrainingArguments(
            run_name=self.model_name,
            output_dir="output/models/teacher/" + self.model_name,
            num_train_epochs=self.config["epochs"],
            fp16=False,
            per_device_train_batch_size=self.config["batch_size"],
            auto_find_batch_size=True,  # Automates the lowering process if CUDA OOM
            dataloader_num_workers=NUM_WORKERS,
            learning_rate=self.config["learning_rate"],
            lr_scheduler_type="cosine",
            weight_decay=self.config["weight_decay"],
            max_grad_norm=self.config["max_grad_norm"],
            metric_for_best_model="eval_loss",  # eval_map,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            save_safetensors=False,  # TODO Might have caused error for facebook/deformable-detr-detic
            push_to_hub=False,
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=self.config["early_stop_patience"],
            early_stopping_threshold=0.0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset[Split.TRAIN],
            eval_dataset=hf_dataset[Split.VALIDATION],
            tokenizer=image_processor,
            data_collator=self.collate_fn,
            callbacks=[early_stopping_callback],
            # compute_metrics=eval_compute_metrics_fn, # TODO Write eval function
        )

        trainer.train()
