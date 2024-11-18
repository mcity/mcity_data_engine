import logging
import os
import re
import time
from difflib import get_close_matches
from functools import partial
from typing import List, Union

import fiftyone as fo
import numpy as np
import torch
from datasets import Split
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForObjectDetection,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    BatchEncoding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.models.owlv2.processing_owlv2 import Owlv2Processor
from transformers.utils.generic import is_torch_device
from transformers.utils.import_utils import is_torch_available, requires_backends

from config.config import NUM_WORKERS
from utils.data_loader import FiftyOneTorchDatasetCOCO, TorchToHFDatasetCOCO


class CustomBatchEncoding(BatchEncoding):
    """
    Improved HF batch encoding with non_blocking=True option.
    https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L192
    """

    def to(
        self, device: Union[str, "torch.device"], non_blocking: bool = False
    ) -> "CustomBatchEncoding":
        """
        Send all values to device by calling `v.to(device, non_blocking=non_blocking)` (PyTorch only).

        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
            non_blocking (`bool`): Whether to perform the copy asynchronously with respect to the host.

        Returns:
            [`CustomBatchEncoding`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])
        import torch

        if (
            isinstance(device, str)
            or is_torch_device(device)
            or isinstance(device, int)
        ):
            self.data = {
                k: v.to(device=device, non_blocking=non_blocking)
                for k, v in self.data.items()
                if isinstance(v, torch.Tensor)
            }
        else:
            logging.warning(
                f"Attempting to cast a CustomBatchEncoding to type {str(device)}. This is not supported."
            )
        return self


class CustomOwlv2Processor(Owlv2Processor):
    """
    Improved HF processor using CustomBatchEncoding encoding.
    """

    def __call__(
        self,
        text=None,
        images=None,
        query_images=None,
        padding="max_length",
        return_tensors="np",
        **kwargs,
    ):
        if text is None and query_images is None and images is None:
            raise ValueError(
                "You have to specify at least one text or query image or image. All three cannot be none."
            )

        if text is not None:
            if isinstance(text, str) or (
                isinstance(text, List) and not isinstance(text[0], List)
            ):
                encodings = [
                    self.tokenizer(
                        text, padding=padding, return_tensors=return_tensors, **kwargs
                    )
                ]

            elif isinstance(text, List) and isinstance(text[0], List):
                encodings = []

                # Maximum number of queries across batch
                max_num_queries = max([len(t) for t in text])

                # Pad all batch samples to max number of text queries
                for t in text:
                    if len(t) != max_num_queries:
                        t = t + [" "] * (max_num_queries - len(t))

                    encoding = self.tokenizer(
                        t, padding=padding, return_tensors=return_tensors, **kwargs
                    )
                    encodings.append(encoding)
            else:
                raise TypeError(
                    "Input text should be a string, a list of strings or a nested list of strings"
                )

            if return_tensors == "np":
                input_ids = np.concatenate(
                    [encoding["input_ids"] for encoding in encodings], axis=0
                )
                attention_mask = np.concatenate(
                    [encoding["attention_mask"] for encoding in encodings], axis=0
                )

            elif return_tensors == "pt" and is_torch_available():
                import torch

                input_ids = torch.cat(
                    [encoding["input_ids"] for encoding in encodings], dim=0
                )
                attention_mask = torch.cat(
                    [encoding["attention_mask"] for encoding in encodings], dim=0
                )

            else:
                raise ValueError("Target return tensor type could not be returned")

            encoding = CustomBatchEncoding()
            encoding["input_ids"] = input_ids
            encoding["attention_mask"] = attention_mask

        if query_images is not None:
            encoding = CustomBatchEncoding()
            query_pixel_values = self.image_processor(
                query_images, return_tensors=return_tensors, **kwargs
            ).pixel_values
            encoding["query_pixel_values"] = query_pixel_values

        if images is not None:
            image_features = self.image_processor(
                images, return_tensors=return_tensors, **kwargs
            )

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif query_images is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None or query_images is not None:
            return encoding
        else:
            return CustomBatchEncoding(
                data=dict(**image_features), tensor_type=return_tensors
            )


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

    def _initialize_zero_shot_processor(
        self, hf_model_config, model_name, batch_size, object_classes, device
    ):
        """
        Initializes and returns the processor, batch classes, tokenized text, and batch tasks
        for zero-shot learning based on the provided HuggingFace model configuration.

        Args:
            hf_model_config (object): The HuggingFace model configuration object.
            model_name (str): The name of the pre-trained model.
            batch_size (int): The size of the batch.
            object_classes (list): A list of class names.
            device (str): The device to which tensors should be moved (e.g., 'cpu' or 'cuda').

        Returns:
            tuple: A tuple containing:
                - processor (object or None): The processor object initialized from the pre-trained model.
                - batch_classes (list or None): The list of batch classes.
                - tokenized_text (torch.Tensor or None): The tokenized text tensor.
                - batch_tasks (list or None): The list of batch tasks (only for OmDetTurboConfig).
        """
        processor, batch_classes, tokenized_text, batch_tasks = None, None, None, None
        if type(hf_model_config).__name__ == "GroundingDinoConfig":
            processor = AutoProcessor.from_pretrained(model_name)  # , do_rescale=False
            # https://huggingface.co/docs/transformers/v4.45.2/en/model_doc/grounding-dino
            classes = " . ".join(object_classes) + " . "
            batch_classes = [classes] * batch_size
            tokenized_text = processor.tokenizer(
                batch_classes,
                padding="max_length",
                return_tensors="pt",
                max_length=256,  # Adjust max_length to match vision hidden state
            ).to(device)

        elif type(hf_model_config).__name__ == "Owlv2Config":
            processor = CustomOwlv2Processor.from_pretrained(model_name)
            batch_classes = object_classes * batch_size
            tokenized_text = processor.tokenizer(
                batch_classes, padding="max_length", return_tensors="pt"
            ).to(device)
        elif type(hf_model_config).__name__ == "OwlViTConfig":
            processor = AutoProcessor.from_pretrained(model_name)
            batch_classes = object_classes * batch_size
            tokenized_text = processor.tokenizer(
                batch_classes, padding="max_length", return_tensors="pt"
            ).to(device)
        elif type(hf_model_config).__name__ == "OmDetTurboConfig":
            processor = AutoProcessor.from_pretrained(model_name)
            batch_classes = [object_classes] * batch_size
            task = "Detect {}.".format(", ".join(object_classes))
            batch_tasks = [task] * batch_size
        else:
            logging.error(
                "HuggingFace AutoModel does not support " + str(type(hf_model_config))
            )

        return processor, batch_classes, tokenized_text, batch_tasks

    def zero_shot_inference(
        self,
        pytorch_dataset,
        batch_size=16,
        detection_threshold=0.2,
        object_classes=None,
    ):
        """
        Perform zero-shot inference with a specified batch size and detection threshold.

        This method attempts to run inference with the given batch size. If a CUDA out of memory
        error occurs, the batch size is halved and the inference is retried until a successful run
        is achieved or the batch size is reduced to less than 1.

        Args:
            batch_size (int, optional): The initial batch size for inference. Defaults to 16.
            detection_threshold (float, optional): The threshold for detection. Defaults to 0.2.

        Raises:
            RuntimeError: If an error other than "CUDA out of memory" occurs during inference.
        """

        successful_run = False

        # Run inference with maximum batch size
        while batch_size >= 1 and successful_run == False:
            try:
                self._zero_shot_inference(
                    pytorch_dataset=pytorch_dataset,
                    batch_size=batch_size,
                    detection_threshold=detection_threshold,
                    object_classes=object_classes,
                )
                successful_run = True

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    batch_size //= 2
                    logging.info("Batch size reduced to " + str(batch_size))
                else:
                    logging.error(f"Runtime error: {e}")
                    raise e
            finally:
                torch.cuda.empty_cache()

        if batch_size < 1:
            logging.error("The model failed to run with batch_size = 1.")

    def _zero_shot_inference(
        self,
        pytorch_dataset,
        batch_size=16,
        detection_threshold=0.2,
        object_classes=None,
    ):
        """
        Performs zero-shot inference on the dataset using a specified model.

        This method either loads precomputed detections from a directory or runs
        inference using a zero-shot model to generate detections. The results are
        stored in the dataset and evaluated.

        Args:
            batch_size (int, optional): The number of samples to process in each batch. Default is 16.
            detection_threshold (float, optional): The confidence threshold for detections. Default is 0.2.

        Raises:
            Exception: If there is an error loading precomputed detections from the directory.

        Notes:
            - If precomputed detections are available in the specified directory, they are loaded and added to the dataset.
            - If precomputed detections are not available, the method initializes a zero-shot model, runs inference, and saves the results.
            - The method supports multiple model configurations, including GroundingDinoConfig, Owlv2Config, OwlViTConfig, and OmDetTurboConfig.
            - The results are stored in the dataset and evaluated using mean Average Precision (mAP).
            - The method logs inference performance metrics to TensorBoard.
        """
        pred_key = re.sub(r"[\W-]+", "_", "pred_" + self.model_name)
        eval_key = re.sub(r"[\W-]+", "_", "eval_" + self.model_name)

        # Read labels from file if already saved
        if os.path.isdir(self.detections_root):
            try:
                temp_dataset = fo.Dataset.from_dir(
                    dataset_dir=self.detections_root,
                    dataset_type=fo.types.COCODetectionDataset,
                    name="temp_dataset",
                    data_path="data.json",
                )

                # Copy all detections from stored dataset into our dataset
                for temp_sample in tqdm(
                    temp_dataset,
                    desc="Loading stored detections and evaluation results",
                ):
                    filepath = temp_sample.filepath
                    sample = self.dataset[filepath]
                    if (
                        "detections" in temp_sample
                        and temp_sample["detections"] is not None
                    ):
                        sample[pred_key] = temp_sample["detections"]
                        sample.save()
            except Exception as e:
                logging.error(
                    f"Data in {self.detections_root} could not be loaded. Error: {e}"
                )
            finally:
                fo.delete_dataset("temp_dataset")

        else:  # Load zero shot model, run inference, and save results
            hf_model_config = AutoConfig.from_pretrained(self.model_name)

            data_loader = DataLoader(
                pytorch_dataset,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                collate_fn=lambda batch: list(zip(*batch)),
            )
            device = torch.device("cuda")

            if object_classes is None:
                object_classes = self.dataset.default_classes

            # Process combined label types like "motorbike/cycler"
            processed_classes = [
                part for classname in object_classes for part in classname.split("/")
            ]
            class_parts_dict = {
                part: classname
                for classname in object_classes
                for part in classname.split("/")
            }
            object_classes = processed_classes

            processor, batch_classes, tokenized_text, batch_tasks = (
                self._initialize_zero_shot_processor(
                    hf_model_config=hf_model_config,
                    model_name=self.model_name,
                    batch_size=batch_size,
                    object_classes=object_classes,
                    device=device,
                )
            )

            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_name
            ).to(device)
            writer = SummaryWriter(log_dir="logs/tensorboard/teacher_zeroshot")

            for step, (images, targets) in enumerate(
                tqdm(data_loader, desc="Zero Shot Teacher Model " + self.model_name)
            ):
                start_time = time.time()
                if len(images) != batch_size:  # For final batch, if batch not full
                    processor, batch_classes, tokenized_text, batch_tasks = (
                        self._initialize_zero_shot_processor(
                            hf_model_config,
                            self.model_name,
                            len(images),  # Key difference
                            object_classes,
                            device,
                        )
                    )

                target_sizes = [tuple(img.shape[1:]) for img in images]
                img_width = target_sizes[0][
                    1
                ]  # FIXME Assumption that all images have the same size
                img_height = target_sizes[0][0]
                if type(hf_model_config).__name__ == "OmDetTurboConfig":
                    images = [to_pil_image(image) for image in images]
                else:
                    images = [(image).to(device, non_blocking=True) for image in images]

                # Process inputs
                if type(hf_model_config).__name__ == "GroundingDinoConfig":
                    inputs = processor(
                        text=None, images=images, return_tensors="pt"
                    ).to(device)
                    inputs.update(tokenized_text)

                elif type(hf_model_config).__name__ == "Owlv2Config":
                    inputs = processor(
                        text=None, images=images, return_tensors="pt"
                    ).to(device, non_blocking=True)
                    inputs.update(tokenized_text)
                elif type(hf_model_config).__name__ == "OwlViTConfig":
                    inputs = processor(
                        text=batch_classes, images=images, return_tensors="pt"
                    ).to(device)
                    # inputs.update(tokenized_text)
                elif type(hf_model_config).__name__ == "OmDetTurboConfig":
                    inputs = processor(
                        text=batch_classes,
                        images=images,
                        task=batch_tasks,
                        return_tensors="pt",
                    ).to(device)

                # Model inference
                with torch.amp.autocast("cuda"):
                    with torch.no_grad():
                        outputs = model(**inputs)

                # Process results
                if type(hf_model_config).__name__ == "GroundingDinoConfig":
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=detection_threshold,
                        text_threshold=detection_threshold,
                    )
                elif type(hf_model_config).__name__ in ["Owlv2Config", "OwlViTConfig"]:
                    results = processor.post_process_object_detection(
                        outputs=outputs,
                        threshold=detection_threshold,
                        target_sizes=target_sizes,
                    )
                elif type(hf_model_config).__name__ == "OmDetTurboConfig":
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        classes=batch_classes,
                        score_threshold=detection_threshold,
                        nms_threshold=detection_threshold,
                        target_sizes=target_sizes,
                    )

                # Store results in V51 dataset
                for result, target in zip(results, targets):
                    boxes, scores = result["boxes"], result["scores"]

                    if "labels" in result:
                        labels = result["labels"]
                    elif "classes" in result:  # OmDet deviates from the other models
                        labels = result["classes"]

                    detections = []
                    for box, score, label in zip(boxes, scores, labels):
                        if type(hf_model_config).__name__ == "GroundingDinoConfig":
                            # Outputs do not comply with given labels
                            # Grounding DINO outputs multiple pairs of object boxes and noun phrases for a given (Image, Text) pair
                            # There can be either multiple labels per output ("bike van") or incomplete ones ("motorcyc")
                            # TODO Improve by not just concidering only the first label
                            processed_label = label.split()[0]
                            if processed_label not in object_classes:
                                matches = get_close_matches(
                                    processed_label, object_classes, n=1, cutoff=0.6
                                )
                                processed_label = matches[0] if matches else None
                            if processed_label == None:
                                logging.info(
                                    "Skipped detection with model "
                                    + type(hf_model_config).__name__
                                    + " due to unclear detection label: "
                                    + label
                                )
                                continue
                            label = class_parts_dict[
                                processed_label
                            ]  # Original label for eval
                            top_left_x = box[0].item()
                            top_left_y = box[1].item()
                            box_width = (box[2] - box[0]).item()
                            box_height = (box[3] - box[1]).item()

                        elif type(hf_model_config).__name__ in [
                            "Owlv2Config",
                            "OwlViTConfig",
                        ]:
                            label = class_parts_dict[object_classes[label]]
                            top_left_x = box[0].item() / img_width
                            top_left_y = box[1].item() / img_height
                            box_width = (box[2].item() - box[0].item()) / img_width
                            box_height = (box[3].item() - box[1].item()) / img_height
                        elif type(hf_model_config).__name__ == "OmDetTurboConfig":
                            label = class_parts_dict[label]
                            top_left_x = box[0].item() / img_width
                            top_left_y = box[1].item() / img_height
                            box_width = (box[2].item() - box[0].item()) / img_width
                            box_height = (box[3].item() - box[1].item()) / img_height

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
                    sample = self.dataset[target["image_id"]]
                    sample[pred_key] = fo.Detections(detections=detections)
                    sample.save()

                # Log inference performance
                end_time = time.time()
                batch_duration = end_time - start_time
                batches_per_second = 1 / batch_duration
                frames_per_second = batches_per_second * batch_size
                writer.add_scalar(
                    "inference/frames_per_second", frames_per_second, step
                )

            # Populate dataset with evaluation results
            try:
                self.dataset.evaluate_detections(
                    pred_key,
                    gt_field="ground_truth",
                    eval_key=eval_key,
                    compute_mAP=True,
                )
            except Exception as e:
                logging.warning(f"Evaluation not possible. Error: {e}")

            # Store labels https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.export
            self.dataset.export(
                export_dir=self.detections_root,
                dataset_type=fo.types.COCODetectionDataset,
                data_path="data.json",
                export_media=None,  # "manifest",
                label_field=pred_key,
                progress=True,
            )

            writer.close()

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
