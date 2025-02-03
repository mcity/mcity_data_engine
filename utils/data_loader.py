# https://docs.python.org/2/library/multiprocessing.html#sharing-state-between-processes
# https://pytorch.org/docs/stable/multiprocessing.html
import logging

import fiftyone.utils.coco as fouc
import numpy as np
import torch
from datasets import Dataset, Split
from torch.multiprocessing import Manager
from torchvision.io import decode_image
from tqdm import tqdm


class FiftyOneTorchDatasetCOCO(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading and processing a FiftyOne dataset in COCO format.
    This class handles multiprocessing to allow loading data with num_workers > 0 and
    converts the dataset into a format compatible with PyTorch's DataLoader.
    Attributes:
        transforms (callable, optional): A function/transform to apply to the images.
        classes (list): List of class names in the dataset.
        labels_map_rev (dict): A dictionary mapping class names to their corresponding indices.
        dataset_length (int): The length of the dataset.
        img_paths (multiprocessing.Manager().list): A list of image file paths.
        ids (multiprocessing.Manager().list): A list of sample IDs.
        metadata (multiprocessing.Manager().list): A list of metadata for each sample.
        labels (multiprocessing.Manager().dict): A dictionary mapping sample IDs to their ground truth detections.
        splits (multiprocessing.Manager().dict): A dictionary mapping sample IDs to their split tags.
    Methods:
        __init__(self, fiftyone_dataset, transforms=None, gt_field="ground_truth"):
            Initializes the dataset with the given FiftyOne dataset and optional transforms.
        __getitem__(self, idx):
            Retrieves the image and target at the specified index.
        __getitems__(self, indices):
            Retrieves a list of (image, target) pairs for the specified indices.
        __len__(self):
            Returns the length of the dataset.
        get_classes(self):
            Returns the list of class names in the dataset.
        get_splits(self):
            Returns a set of unique split tags in the dataset.

    References:
        - https://github.com/voxel51/fiftyone-examples/blob/master/examples/pytorch_detection_training.ipynb
        - https://github.com/voxel51/fiftyone/issues/1302
        - https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        - https://github.com/pytorch/pytorch/issues/13246
        - https://github.com/pytorch/pytorch/issues/13246#issuecomment-715050814
    """

    def __init__(self, fiftyone_dataset, transforms=None, gt_field="ground_truth"):
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
        for i, sample_id in tqdm(enumerate(ids), total=len(ids), desc="Generating torch dataset from Voxel51 dataset"):
            self.img_paths.append(img_paths[i])
            self.ids.append(sample_id)  # Store the sample ID
            self.metadata.append(metadata[i])

            # Extract labels and splits for each sample
            if ground_truths and ground_truths[i]:  # Check if the ground truth exists for the sample
                self.labels[sample_id] = ground_truths[i].detections
            if tags[i]:  # Check if the tags exist for the sample
                self.splits[sample_id] = tags[i][0]  # FIXME Assume first tag is split

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_id = self.ids[idx]
        metadata = self.metadata[idx]
        detections = self.labels.get(idx, [])
        img = decode_image(img_path, mode="RGB")
        boxes = []
        labels = []
        area = []
        iscrowd = []

        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det,
                metadata,
                category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, w, h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {
            "bbox": torch.as_tensor(boxes, dtype=torch.float32),
            "category_id": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": img_id,
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)
        }
        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

    def __len__(self):
        return self.dataset_length

    def get_classes(self):
        return self.classes

    def get_splits(self):
        return set(self.splits.values())

class TorchToHFDatasetCOCO:
    """
    A class to convert a PyTorch dataset to a Hugging Face dataset in COCO format.

    Attributes:
    -----------
    torch_dataset : object
        The PyTorch dataset to be converted.

    Methods:
    --------
    __init__(torch_dataset):
        Initializes the TorchToHFDatasetCOCO with a PyTorch dataset.

    convert():
        Converts the PyTorch dataset to a Hugging Face dataset.
    """

    split_mapping = {
        "train": Split.TRAIN,
        "test": Split.TEST,
        "validation": Split.VALIDATION,
        "val": Split.VALIDATION,
    }

    def __init__(self, torch_dataset):
        self.torch_dataset = torch_dataset

    def convert(self):
        try:
            splits = self.torch_dataset.get_splits()
            hf_dataset = {
                self.split_mapping[split]: Dataset.from_generator(
                    gen_factory(self.torch_dataset, split),
                    split=self.split_mapping[split],
                )
                for split in splits
            }
            return hf_dataset
        except Exception as e:
            logging.error(f"Error in dataset conversion from Torch to Hugging Face: {e}")


def gen_factory(torch_dataset, split_name):
    """
    Factory function to create a generator function for the Hugging Face dataset.

    Args:
    -----
    torch_dataset : FiftyOneTorchDatasetCOCO
        The PyTorch dataset to be converted.
    split_name : str
        The name of the split to filter the data.

    Returns:
    --------
    function
        A generator function that yields data samples for the specified split.

    Note:
    -----
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
        for idx, (img_path, img_id) in enumerate(zip(img_paths, img_ids)):
            split = splits[img_id]
            if split != split_name:
                continue

            sample_data = {
                "metadata": metadata[idx],
                "detections": labels[img_id],
            }
            target = create_target(sample_data, labels_map_rev, idx)
            yield {
                "image": img_path,
                "target": target,
                "split": split,
            }

    return _gen


def create_target(sample_data, labels_map_rev, idx):
    """
    Creates a target dictionary for a given sample.

    Args:
    -----
    sample_data : dict
        The data of the sample, including detections and metadata.
    labels_map_rev : dict
        A dictionary mapping class names to indices.
    idx : int
        The index of the sample.

    Returns:
    --------
    dict
        A dictionary containing bounding boxes, category IDs, image ID, area, and iscrowd flags.
    """
    detections = sample_data["detections"]

    boxes = [det.bounding_box for det in detections]
    labels = [labels_map_rev[det.label] for det in detections]
    area = [det.bounding_box[2] * det.bounding_box[3] for det in detections]
    iscrowd = [0 for _ in detections]

    return {
        "bbox": boxes,
        "category_id": labels,
        "image_id": idx,
        "area": area,
        "iscrowd": iscrowd,
    }
