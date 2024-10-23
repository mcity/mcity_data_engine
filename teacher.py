import logging

from PIL import Image

from functools import partial

import numpy as np
import torch
from torch.cuda import Stream
from torch.utils.data import DataLoader
from torchvision import transforms

from config.config import WORKFLOWS, NUM_WORKERS
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from tqdm import tqdm

from utils.data_loader import FiftyOneTorchDatasetCOCO, TorchToHFDatasetCOCO
from datasets import Split
from config.config import NUM_WORKERS

from transformers import BatchEncoding
from transformers.models.owlvit.processing_owlvit import OwlViTProcessor
from transformers.utils.import_utils import requires_backends, is_torch_available
from transformers.utils.generic import is_torch_device
from typing import List, Union


class CustomBatchEncoding(BatchEncoding):
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


class CustomOwlViTProcessor(OwlViTProcessor):
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


def zeroshot_collate_fn(batch):
    return list(zip(*batch))


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
    def __init__(self, dataset, config=None):
        self.dataset = dataset
        self.config = config
        self.model_name = config["model_name"]

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

    def zero_shot_inference(self):
        transform = transforms.Compose([transforms.ToTensor()])

        self.dataset = self.dataset.take(200)
        pytorch_dataset = FiftyOneTorchDatasetCOCO(
            self.dataset,
            transforms=transform,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        classes = [
            "car",
            "truck",
            "bus",
            "trailer",
            "motorbike/cycler",  # TODO Maybe split into two and then merge again for eval?
            "pedestrian",
            "van",
            "pickup",
        ]

        hf_model_config = AutoConfig.from_pretrained(self.model_name)
        if type(hf_model_config) in AutoModelForZeroShotObjectDetection._model_mapping:
            processor = CustomOwlViTProcessor.from_pretrained(
                self.model_name, do_rescale=False
            )
            # processor = AutoProcessor.from_pretrained(self.model_name, do_rescale=False)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_name
            ).to(device)
        else:
            model = None
            logging.error(
                "HuggingFace AutoModel does not support " + str(type(hf_model_config))
            )

        data_loader = DataLoader(
            pytorch_dataset,
            batch_size=8,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=zeroshot_collate_fn,
            prefetch_factor=8,  # Prefetch data
            persistent_workers=True,  # Keep workers alive
        )

        batch_classes = classes * data_loader.batch_size

        tokenized_text = processor.tokenizer(
            batch_classes, padding="max_length", return_tensors="pt"
        ).to(device)

        # Create a separate CUDA stream for data transfer
        data_transfer_stream = Stream()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # Disable stack tracing to reduce memory usage
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "./logs/tensorboard/zeroshot_profile/"
            ),
        ) as prof:
            for images, targets in tqdm(data_loader):
                with torch.profiler.record_function("image loading"):
                    images = [(image).to(device, non_blocking=True) for image in images]
                with torch.profiler.record_function("inputs_by_processor"):
                    # Use the separate CUDA stream for data transfer
                    # with torch.cuda.stream(data_transfer_stream):
                    inputs = processor(
                        text=None, images=images, return_tensors="pt"
                    ).to(device, non_blocking=True)
                    # Ensure the main stream waits for the data transfer to complete
                    # torch.cuda.synchronize()
                    ## https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlv2/processing_owlv2.py#L49

                    # inputs = processor(
                    #    text=None, images=images, return_tensors="pt"
                    # ).to(device, non_blocking=True)
                # with torch.profiler.record_function("inputs_to_device"):
                #    inputs = {
                #        k: v.to(device, non_blocking=True) for k, v in inputs.items()
                #    }
                with torch.profiler.record_function("inputs_update"):
                    inputs.update(tokenized_text)

                with torch.profiler.record_function("inference"):
                    with torch.amp.autocast("cuda"):
                        with torch.no_grad():
                            outputs = model(**inputs)

                with torch.profiler.record_function("post-processing"):
                    results = processor.post_process_object_detection(
                        outputs=outputs, threshold=0.2
                    )

                with torch.profiler.record_function("processing"):
                    for result in results:
                        print(result)

                prof.step()  # Notify profiler of the end of a step

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
