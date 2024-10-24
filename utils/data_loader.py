import torch
from PIL import Image
import fiftyone.utils.coco as fouc

import logging

from datasets import Dataset, Split


class FiftyOneTorchDatasetCOCO(torch.utils.data.Dataset):
    """
    FiftyOneTorchDatasetCOCO is a custom PyTorch Dataset for loading COCO-style datasets from FiftyOne.

    Attributes:
        transforms (callable, optional): A function/transform that takes in an image and returns a transformed version.
        gt_field (str): The name of the ground truth field in the FiftyOne dataset.
        img_paths (list): List of image file paths from the FiftyOne dataset.
        labels (dict): Dictionary mapping image file paths to their corresponding detections.
        metadata (dict): Dictionary mapping image file paths to their corresponding metadata.
        splits (dict): Dictionary mapping image file paths to their corresponding split tags.
        classes (list): List of distinct labels in the dataset.
        labels_map_rev (dict): Dictionary mapping class labels to their corresponding indices.

    Methods:
        __init__(self, fiftyone_dataset, transforms=None, gt_field="ground_truth", classes=None):
            Initializes the FiftyOneTorchDatasetCOCO with the given parameters.

        __getitem__(self, idx):
            Returns the image and target at the specified index.

        __getitems__(self, indices):
            Returns a list of samples for the specified indices.

        __len__(self):
            Returns the number of samples in the dataset.

        get_classes(self):
            Returns the list of classes in the dataset.

        get_splits(self):
            Returns the set of split tags in the dataset.

    References:
        - https://github.com/voxel51/fiftyone-examples/blob/master/examples/pytorch_detection_training.ipynb
        - https://github.com/voxel51/fiftyone/issues/1302
    """

    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    ):
        self.transforms = transforms
        self.gt_field = gt_field
        self.img_paths = [sample.filepath for sample in fiftyone_dataset]
        self.labels = {
            sample.filepath: sample[gt_field].detections
            for sample in fiftyone_dataset
            if gt_field in sample
        }
        self.metadata = {
            sample.filepath: sample.metadata for sample in fiftyone_dataset
        }
        self.splits = {
            sample.filepath: sample.tags[0]
            for sample in fiftyone_dataset
            if sample.tags
        }
        self.classes = classes
        if not self.classes:
            # Check if any sample has the gt_field
            if any(gt_field in sample for sample in fiftyone_dataset):
                # Get list of distinct labels that exist in the view
                self.classes = fiftyone_dataset.distinct(f"{gt_field}.detections.label")
            else:
                self.classes = []
        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        metadata = self.metadata[img_path]
        detections = self.labels.get(img_path, [])
        img = Image.open(img_path).convert("RGB")
        boxes = []
        labels = []
        area = []
        iscrowd = []

        if detections:
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

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __getitems__(self, indices):
        samples = [self.__getitem__(idx) for idx in indices]
        return samples

    def __len__(self):
        return len(self.img_paths)

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
        splits = self.torch_dataset.get_splits()
        hf_dataset = {
            self.split_mapping[split]: Dataset.from_generator(
                gen_factory(self.torch_dataset, split),
                split=self.split_mapping[split],
            )
            for split in splits
        }
        return hf_dataset


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
    labels_map_rev = torch_dataset.labels_map_rev
    splits = torch_dataset.splits
    metadata = torch_dataset.metadata
    labels = torch_dataset.labels

    def _gen():
        for idx, img_path in enumerate(img_paths):
            split = splits[img_path]
            if split != split_name:
                continue

            sample_data = {
                "metadata": metadata[img_path],
                "detections": labels[img_path],
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
