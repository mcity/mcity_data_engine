# https://docs.python.org/2/library/multiprocessing.html#sharing-state-between-processes
# https://pytorch.org/docs/stable/multiprocessing.html
import logging

import fiftyone.utils.coco as fouc
import torch
from datasets import Dataset, Split
from torch.multiprocessing import Manager
from torchvision.io import decode_image
from tqdm import tqdm

from utils.dataset_loader import get_split


class FiftyOneTorchDatasetCOCO(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading and processing a FiftyOne dataset in COCO format.
    This class handles multiprocessing to allow loading data with num_workers > 0 and
    converts the dataset into a format compatible with PyTorch's DataLoader.

    References:
        - https://github.com/voxel51/fiftyone-examples/blob/master/examples/pytorch_detection_training.ipynb
        - https://github.com/voxel51/fiftyone/issues/1302
        - https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        - https://github.com/pytorch/pytorch/issues/13246
        - https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
    """

    def __init__(self, fiftyone_dataset, transforms=None, gt_field="ground_truth"):
        """Initialize dataset from Voxel51 (fiftyone) dataset with optional transforms and ground truth field name."""
        logging.info(f"Collecting data for torch dataset conversion.")
        self.transforms = transforms
        self.classes = fiftyone_dataset.default_classes
        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}
        self.dataset_length = len(fiftyone_dataset)

        # Multiprocessing for data loader with num_workers > 0
        # https://docs.python.org/3/library/multiprocessing.html
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-612396143
        manager = Manager()
        self.img_paths = manager.list()
        self.ids = manager.list()
        self.metadata = manager.list()

        self.labels = manager.dict()
        self.splits = manager.dict()

        # Use values() to directly get the required fields from the dataset
        img_paths = fiftyone_dataset.values("filepath")
        ids = fiftyone_dataset.values("id")
        metadata = fiftyone_dataset.values("metadata")

        try:
            ground_truths = fiftyone_dataset.values(gt_field)
        except:
            logging.info(f"Voxel51 dataset has no field named '{gt_field}'")
            ground_truths = None
        tags = fiftyone_dataset.values("tags")

        # Process all samples with values() in place of the loop
        for i, sample_id in tqdm(
            enumerate(ids),
            total=len(ids),
            desc="Generating torch dataset from Voxel51 dataset",
        ):
            self.img_paths.append(img_paths[i])
            self.ids.append(sample_id)  # Store the sample ID
            self.metadata.append(metadata[i])

            # Extract labels and splits for each sample
            if (
                ground_truths and ground_truths[i]
            ):  # Check if the ground truth exists for the sample
                # Store detections as (top_left_x, top_left_y, width, height) in rel. coordinates between [0,1]
                self.labels[sample_id] = ground_truths[i].detections
            if tags[i]:  # Check if the tags exist for the sample
                self.splits[sample_id] = get_split(tags[i])

    def __getitem__(self, idx):
        """Returns transformed image and its target dictionary containing bounding boxes, category IDs, image ID, areas and crowd flags."""
        img_path = self.img_paths[idx]
        img_id = self.ids[idx]
        metadata = self.metadata[idx]
        detections = self.labels.get(img_id, [])
        img = decode_image(img_path, mode="RGB")
        boxes = []
        labels = []
        area = []
        iscrowd = []

        for det in detections:
            category_id = self.labels_map_rev[det.label]
            # https://docs.voxel51.com/api/fiftyone.utils.coco.html#fiftyone.utils.coco.COCOObject
            coco_obj = fouc.COCOObject.from_label(
                det,
                metadata,
                category_id=category_id,
            )
            x_min, y_min, w, h = coco_obj.bbox  # Absolute coordinates
            boxes.append([x_min, y_min, w, h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {
            "bbox": torch.as_tensor(boxes, dtype=torch.float32),
            "category_id": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": img_id,
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __getitems__(self, indices):
        """Returns a list of items at the specified indices using __getitem__ for each index."""
        return [self.__getitem__(idx) for idx in indices]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.dataset_length

    def get_classes(self):
        """Return the list of classes available in the dataset."""
        return self.classes

    def get_splits(self):
        """Returns a set of all unique split labels in the dataset."""
        return set(self.splits.values())


class TorchToHFDatasetCOCO:
    """Convert PyTorch COCO-style dataset to Hugging Face dataset format.

    This class facilitates the conversion of PyTorch COCO-style datasets to the Hugging Face
    dataset format, handling split management and data generation.
    """

    split_mapping = {
        "train": Split.TRAIN,
        "test": Split.TEST,
        "validation": Split.VALIDATION,
        "val": Split.VALIDATION,
    }

    def __init__(self, torch_dataset):
        """Initialize a data loader wrapper around a PyTorch dataset."""
        self.torch_dataset = torch_dataset

    def convert(self):
        """Converts a PyTorch dataset to a Hugging Face dataset dictionary with mapped splits."""
        try:
            default_split_hf = "test"
            splits = self.torch_dataset.get_splits()
            if len(splits) == 0:
                logging.warning(
                    f"Hugging Face Datasets expects splits, but none are provided. Setting '{default_split_hf}' as the default split."
                )
                splits = [default_split_hf]
            hf_dataset = {
                self.split_mapping[split]: Dataset.from_generator(
                    gen_factory(self.torch_dataset, split, default_split_hf),
                    split=self.split_mapping[split],
                )
                for split in splits
            }
            return hf_dataset
        except Exception as e:
            logging.error(
                f"Error in dataset conversion from Torch to Hugging Face: {e}"
            )


def gen_factory(torch_dataset, split_name, default_split_hf):
    """
    Factory function to create a generator function for the Hugging Face dataset.

    This function ensures that all objects used within the generator function are picklable.
    The FiftyOne dataset is iterated to collect sample data, which is then used within the generator function.
    """
    img_paths = torch_dataset.img_paths
    img_ids = torch_dataset.ids
    splits = torch_dataset.splits
    metadata = torch_dataset.metadata
    labels = torch_dataset.labels
    labels_map_rev = torch_dataset.labels_map_rev

    def _gen():
        """Yields dictionaries containing image paths, object targets, and dataset splits for each image in the dataset."""
        for idx, (img_path, img_id) in enumerate(zip(img_paths, img_ids)):
            split = splits.get(img_id, None)

            # If no split is provided, set default split
            if split is None:
                split = default_split_hf

            # Only select samples of the split we are currently looking for
            if split != split_name:
                continue

            sample_data = {
                "metadata": metadata[idx],
                "detections": labels.get(img_id, None),
            }
            target = create_target(sample_data, labels_map_rev, idx)
            yield {
                "image_path": img_path,
                "objects": target,
                "split": split,
            }

    return _gen


def create_target(sample_data, labels_map_rev, idx, convert_to_coco=True):
    """Convert detection data to COCO format, transforming relative coordinates to absolute if specified."""

    detections = sample_data.get("detections", [])
    img_width = sample_data["metadata"]["width"]
    img_height = sample_data["metadata"]["height"]

    # Handle empty or missing detections
    if not detections:
        logging.warning(f"No detections found for sample {idx}")
        return {
            "bbox": [],
            "category_id": [],
            "image_id": idx,
            "area": [],
            "iscrowd": [],
        }

    boxes = []
    areas = []

    if convert_to_coco:
        # From rel. coordinates between [0,1] to abs. coordinates (COCO)
        for det in detections:
            x_min = det.bounding_box[0] * img_width
            y_min = det.bounding_box[1] * img_height
            width = det.bounding_box[2] * img_width
            height = det.bounding_box[3] * img_height
            boxes.append([x_min, y_min, width, height])
            area = width * height
            areas.append(area)
    else:
        boxes = [det.bounding_box for det in detections]
        areas = [det.bounding_box[2] * det.bounding_box[3] for det in detections]

    labels = [labels_map_rev[det.label] for det in detections]
    iscrowd = [0 for _ in detections]

    return {
        "bbox": boxes,
        "category_id": labels,
        "image_id": idx,
        "area": areas,
        "iscrowd": iscrowd,
    }
