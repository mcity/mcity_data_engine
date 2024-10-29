import objgraph
import os
import torch.multiprocessing as mp

# from memory_profiler import profile

# import tracemalloc
import gc

from pympler import muppy, summary, tracker

import logging

from difflib import get_close_matches

from utils.dataset_loader import *

import re
import time
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from config.config import NUM_WORKERS
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

import fiftyone as fo

from utils.data_loader import FiftyOneTorchDatasetCOCO
from config.config import NUM_WORKERS

from transformers import BatchEncoding
from transformers.models.owlv2.processing_owlv2 import Owlv2Processor
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
                "Attempting to cast a CustomBatchEncoding to type %s. This is not supported.",
                str(device),
            )
        return self


class CustomOwlv2Processor(Owlv2Processor):
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

    def _initialize_zero_shot_processor(
        self, hf_model_config, model_name, batch_size, classes_v51, device
    ):
        """
        Initializes and returns the processor, batch classes, tokenized text, and batch tasks
        for zero-shot learning based on the provided HuggingFace model configuration.

        Args:
            hf_model_config (object): The HuggingFace model configuration object.
            model_name (str): The name of the pre-trained model.
            batch_size (int): The size of the batch.
            classes_v51 (list): A list of class names.
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
            processor = AutoProcessor.from_pretrained(model_name, do_rescale=False)
            # https://huggingface.co/docs/transformers/v4.45.2/en/model_doc/grounding-dino
            classes = " . ".join(classes_v51) + " . "
            batch_classes = [classes] * batch_size
            # tokenized_text = processor.tokenizer(
            #    batch_classes,
            #    padding="max_length",
            #    return_tensors="pt",
            #    max_length=256,  # Adjust max_length to match vision hidden state
            # ).to(device)

        elif type(hf_model_config).__name__ == "Owlv2Config":
            processor = CustomOwlv2Processor.from_pretrained(
                model_name, do_rescale=False
            )
            batch_classes = classes_v51 * batch_size
            # tokenized_text = processor.tokenizer(
            #    batch_classes, padding="max_length", return_tensors="pt"
            # ).to(device)
        elif type(hf_model_config).__name__ == "OwlViTConfig":
            processor = AutoProcessor.from_pretrained(model_name, do_rescale=False)
            batch_classes = classes_v51 * batch_size
            # tokenized_text = processor.tokenizer(
            #    batch_classes, padding="max_length", return_tensors="pt"
            # ).to(device)
        elif type(hf_model_config).__name__ == "OmDetTurboConfig":
            processor = AutoProcessor.from_pretrained(model_name, do_rescale=False)
            batch_classes = [classes_v51] * batch_size
            task = "Detect {}.".format(", ".join(classes_v51))
            batch_tasks = [task] * batch_size
        else:
            logging.error(
                "HuggingFace AutoModel does not support " + str(type(hf_model_config))
            )

        return processor, batch_classes, tokenized_text, batch_tasks

    def zero_shot_inference(self, batch_size=16, detection_threshold=0.2):

        successful_run = False
        # Run inference with maximum batch size
        while batch_size >= 1 and successful_run == False:
            try:
                self._zero_shot_inference(
                    batch_size=batch_size, detection_threshold=detection_threshold
                )
                successful_run = True

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    batch_size //= 2
                    logging.info("Batch size reduced to " + str(batch_size))
                else:
                    logging.error("Runtime error: %s", e)
                    raise e

    # @profile
    def _zero_shot_inference(self, batch_size=16, detection_threshold=0.2):
        # tracemalloc.start()
        pred_key = re.sub(r"[\W-]+", "_", "pred_" + self.model_name)
        eval_key = re.sub(r"[\W-]+", "_", "eval_" + self.model_name)

        hf_model_config = AutoConfig.from_pretrained(self.model_name)

        transform = (
            None
            if type(hf_model_config).__name__ == "OmDetTurboConfig"
            else transforms.Compose([transforms.ToTensor()])
        )

        # self.dataset = self.dataset.take(640)
        print_memory("Config and transform loaded")

        pytorch_dataset = FiftyOneTorchDatasetCOCO(
            self.dataset,
            transforms=transform,
        )
        print_memory("Pytorch dataset loaded")

        data_loader = DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            collate_fn=zeroshot_collate_fn,
        )

        # data_loader = DataLoader(
        #    pytorch_dataset,
        #    batch_size=batch_size,
        #    num_workers=NUM_WORKERS,
        #    pin_memory=True,
        #    collate_fn=zeroshot_collate_fn,
        #    persistent_workers=False,
        # )
        print_memory("Dataloader loaded")
        device = torch.device("cuda")

        classes_v51 = self.dataset.default_classes

        # Process combined label types like "motorbike/cycler"
        processed_classes = [
            part for classname in classes_v51 for part in classname.split("/")
        ]
        class_parts_dict = {
            part: classname
            for classname in classes_v51
            for part in classname.split("/")
        }
        classes_v51 = processed_classes
        print_memory("Classnames processed")

        processor, batch_classes, tokenized_text, batch_tasks = (
            self._initialize_zero_shot_processor(
                hf_model_config=hf_model_config,
                model_name=self.model_name,
                batch_size=batch_size,
                classes_v51=classes_v51,
                device=device,
            )
        )
        print_memory("Initialization done")

        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(
            device
        )
        print_memory("Model loaded")

        for images, targets in data_loader:

            print_memory("Dataloader starts")
            # Get target sizes
            if type(hf_model_config).__name__ == "OmDetTurboConfig":
                target_sizes = [img.size[::-1] for img in images]
            else:
                target_sizes = [tuple(img.shape[1:]) for img in images]
                images = [(image).to(device, non_blocking=True) for image in images]

            print_memory("Target sizes computed")
            # Process inputs
            if type(hf_model_config).__name__ == "GroundingDinoConfig":
                inputs = processor(
                    text=batch_classes, images=images, return_tensors="pt"
                ).to(device)
                # inputs.update(tokenized_text)

            elif type(hf_model_config).__name__ == "Owlv2Config":
                inputs = processor(
                    text=batch_classes, images=images, return_tensors="pt"
                ).to(device, non_blocking=True)
                # inputs.update(tokenized_text)
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

            print_memory("Inputs processed")
            # Model inference
            with torch.amp.autocast("cuda"):
                with torch.no_grad():
                    outputs = model(**inputs)

            print_memory("Model inference done")
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

            print_memory("Results processed")
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
                        processed_label = label.split()[0]
                        if processed_label not in classes_v51:
                            matches = get_close_matches(
                                processed_label, classes_v51, n=1, cutoff=0.6
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
                        # Get image size (ID is stored in annotation)
                        img_path = pytorch_dataset.img_paths[target["image_id"].item()]
                        sample = self.dataset[img_path]
                        img_width = sample.metadata.width
                        img_height = sample.metadata.height

                        # Convert bbox to V51 type
                        label = class_parts_dict[classes_v51[label]]
                        top_left_x = box[0].item() / img_width
                        top_left_y = box[1].item() / img_height
                        box_width = (box[2].item() - box[0].item()) / img_width
                        box_height = (box[3].item() - box[1].item()) / img_height
                    elif type(hf_model_config).__name__ == "OmDetTurboConfig":
                        # Get image size
                        img_path = pytorch_dataset.img_paths[target["image_id"].item()]
                        sample = self.dataset[img_path]
                        img_width = sample.metadata.width
                        img_height = sample.metadata.height

                        # Convert bbox to V51 type
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

                    detections.append(detection)

                # Attach label to V51 dataset
                img_path = pytorch_dataset.img_paths[target["image_id"].item()]
                sample = self.dataset[img_path]
                sample[pred_key] = fo.Detections(detections=detections)
                sample.save()

            print_memory("Results stored in V51 dataset")
            break  # Only 1 batch

        print_memory("Dataloader ended")

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics("lineno")

        # for stat in top_stats[:10]:
        #    print(stat)


def print_memory(description=""):
    total_memory, used_memory, free_memory = map(
        lambda x: int(x) / 1024, os.popen("free -t -m").readlines()[-1].split()[1:]
    )
    print(f"{used_memory:.2f} - {description}")

    # memory_info = psutil.virtual_memory()
    # used_memory_bytes = memory_info.used
    # used_memory_gb = used_memory_bytes / (1024 * 1024 * 1024)
    # print(str(used_memory_gb) + " - " + description)


def run_inference(dataset, config):
    teacher = Teacher(dataset=dataset, config=config)
    print_memory("Teacher initialized")
    teacher.zero_shot_inference(batch_size=16, detection_threshold=0.2)
    print_memory("Zero Shot Inference done")

    # Clean up
    del teacher
    # print_memory("Teacher deleted")
    del dataset
    # print_memory("Dataset deleted")
    del config
    # print_memory("Config deleted")
    gc.collect()
    # print_memory("After garbage collection")
    torch.cuda.empty_cache()
    # print_memory("After torch empty cache release")
    torch.cuda.ipc_collect()
    # print_memory("After torch ipc release")
    torch.cuda.reset_peak_memory_stats()
    # print_memory("After torch peak memory stats reset")
    print_memory("After cleanup")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    SELECTED_DATASET = "mcity_fisheye_2000"
    MODEL_NAME = "google/owlv2-base-patch16"

    mp.set_start_method("fork")  # Default, other two very slow

    dataset_info = load_dataset_info(SELECTED_DATASET)
    loader_function = dataset_info.get("loader_fct")
    dataset = globals()[loader_function](dataset_info)
    print_memory("V51 dataset loaded")

    config_file_path = "wandb_runs/teacher_zero_shot_config.json"
    with open(config_file_path, "r") as file:
        config = json.load(file)

    config["overrides"]["run_config"]["model_name"] = MODEL_NAME
    config["overrides"]["run_config"]["v51_dataset_name"] = SELECTED_DATASET
    config = config["overrides"]["run_config"]

    for i in range(2):
        logging.warning("Loop " + str(i))
        run_inference(dataset, config)


if __name__ == "__main__":
    print_memory("START")
    main()
    print_memory("END")
