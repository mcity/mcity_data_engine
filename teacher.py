import logging
import wandb
import random

from functools import partial

import numpy as np
import torch

from config.config import WORKFLOWS
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)

import datasets
from fiftyone.utils.image import read
from config.config import NUM_WORKERS


def fiftyone_to_yolo(top_left_x, top_left_y, width, height):
    """
    Input: Bbox in relative values [0,1]: top-left-x, top-left-y, w,h (V51)
    Return: Bbox in relative values [0,1]: center-x, center-y, w,h (YOLO)
    """
    center_x = top_left_x + width / 2
    center_y = top_left_y + height / 2

    return center_x, center_y, width, height


class V51Transform:
    def __init__(self, v51_instance, adapt_function):
        self.v51_instance = v51_instance
        self.adapt_function = adapt_function

    def __getitem__(self, index):
        original_data = self.v51_instance[index]
        return self.adapt_function(original_data)


class Teacher:
    def __init__(self, dataset, dataset_info, config, wandb_project):
        self.dataset = dataset
        self.config = config
        self.model_name = config["model_name"]
        self.img_width = (
            dataset.take(1).first().metadata.width
        )  # Assumption: All images have same size
        self.img_height = dataset.take(1).first().metadata.height

        self.categories = dataset.default_classes
        self.id2label = {index: x for index, x in enumerate(self.categories, start=0)}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def convert_to_coco_format_w_yolo_bbox(self, image_id, annotations):
        """
        V51 bbox format is [top-left-x, top-left-y, width, height]

        HuggingFace DETR needs COCO format with normalized YOLO bboxes:
        https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/detr#transformers.DetrImageProcessor
        """
        coco_annotations = []
        bounding_boxes = [annotation.bounding_box for annotation in annotations]
        labels = [annotation.label for annotation in annotations]
        for bbox, label in zip(bounding_boxes, labels):
            top_left_x, top_left_y, width, height = bbox
            center_x, center_y, width, height = fiftyone_to_yolo(
                top_left_x, top_left_y, width, height
            )

            label_id = self.dataset.default_classes.index(label)
            coco_annotation = {
                "image_id": image_id,
                "bbox": [center_x, center_y, width, height],
                "category_id": label_id,
                "area": width
                * height
                * self.img_width
                * self.img_height,  # width * height in pixel values
                "iscrowd": 0,  # Assuming no crowd annotations
            }
            coco_annotations.append(coco_annotation)

        return {"image_id": image_id, "annotations": coco_annotations}

    def transform_data(self, data, image_processor, return_pixel_mask=False):
        """Apply format annotations in COCO format for object detection task"""
        images = []
        annotations = []

        for sample in data:
            logging.warning(sample)
            image_np = read(sample.filepath)  # read image as np array
            images.append(image_np)
            annotations_sample = sample.ground_truth.detections
            image_id = random.randint()
            formatted_annotations = self.convert_to_coco_format_w_yolo_bbox(
                image_id, annotations_sample
            )
            annotations.append(formatted_annotations)

        result = image_processor(
            images=images, annotations=annotations, return_tensors="pt"
        )

        if not return_pixel_mask:
            result.pop("pixel_mask", None)

        return result

    def apply_transform(self, dataset, transform_fn):
        transformed_data = []
        for element in dataset:
            transformed_element = transform_fn(element)
            transformed_data.append(transformed_element)
        return transformed_data

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

    def train(self, finetune):
        train_data_v51 = self.dataset.match_tags("train")
        val_data_v51 = self.dataset.match_tags("val")
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            do_resize=True,
            size={"max_height": self.img_height, "max_width": self.img_width},
            do_pad=True,
            pad_size={"height": self.img_height, "width": self.img_width},
            do_convert_annotations=False,  # expects YOLO (center_x, center_y, width, height) between [0,1]
        )

        # do_convert_annotations (bool, optional, defaults to True)
        # Controls whether to convert the annotations to the format expected by the DETR model.
        # Converts the bounding boxes to the format (center_x, center_y, width, height) and in the range [0, 1].
        # Can be overridden by the do_convert_annotations parameter in the preprocess method.

        # Process train and val data
        # train_data = self.transform_data(train_data_v51, image_processor)
        # al_data = self.transform_data(val_data_v51, image_processor)

        # This calls the function whenever an element of train_data is being accessed
        train_data = self.apply_transform(
            train_data_v51, lambda x: self.transform_data(x, image_processor)
        )
        val_data = self.apply_transform(
            val_data_v51, lambda x: self.transform_data(x, image_processor)
        )

        # for sample in dataset.iter_samples(progress=True, autosave=True):
        # https://docs.voxel51.com/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.iter_samples

        train_data = V51Transform(train_data_v51, self.transform_data)
        val_data = V51Transform(val_data_v51, self.transform_data)

        for sample in train_data:
            logging.warning(sample)

        # Maybe this? https://albumentations.ai/docs/integrations/fiftyone/

        model = AutoModelForObjectDetection.from_pretrained(
            self.model_name,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        training_args = TrainingArguments(
            output_dir="output/models/teacher/" + self.model_name,
            num_train_epochs=self.config["epochs"],
            fp16=False,
            per_device_train_batch_size=self.config["batch_size"],
            dataloader_num_workers=NUM_WORKERS,
            learning_rate=self.config["learning_rate"],
            lr_scheduler_type="cosine",
            weight_decay=self.config["weight_decay"],
            max_grad_norm=self.config["max_grad_norm"],
            metric_for_best_model="eval_map",
            greater_is_better=True,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=image_processor,
            data_collator=self.collate_fn,
            # compute_metrics=eval_compute_metrics_fn,
        )

        trainer.train()
