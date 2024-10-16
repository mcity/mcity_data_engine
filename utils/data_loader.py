import torch
from PIL import Image
import fiftyone.utils.coco as fouc

import logging

from datasets import Dataset, NamedSplit, Split


class FiftyOneTorchDatasetCOCO(torch.utils.data.Dataset):
    """A PyTorch Dataset class for loading data from a FiftyOne dataset in COCO format.

        fiftyone_dataset (fiftyone.core.dataset.DatasetView): A FiftyOne dataset or view to be used for training or testing.
        transforms (callable, optional): A function/transform that takes in an image and target and returns a transformed version.
        gt_field (str, optional): The name of the field in fiftyone_dataset that contains the desired labels to load. Default is "ground_truth".
        classes (list of str, optional): A list of class strings to define the mapping between class names and indices. If None, all classes present in the given fiftyone_dataset will be used.

    Attributes:
        samples (fiftyone.core.dataset.DatasetView): The FiftyOne dataset or view.
        transforms (callable, optional): The transform function to apply to images and targets.
        gt_field (str): The name of the field in fiftyone_dataset that contains the desired labels.
        img_paths (list of str): List of image file paths from the FiftyOne dataset.
        classes (list of str): List of class names.
        labels_map_rev (dict): A dictionary mapping class names to indices.

    Methods:
        __getitem__(idx):
            Retrieves the image and target at the specified index.

        __getitems__(indices):
            Retrieves a list of samples for the specified indices.

        __len__():
            Returns the number of samples in the dataset.

        get_classes():
            Returns the list of class names.

        get_splits():
            Returns a set of unique split names from the dataset.
    """

    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct("%s.detections.label" % gt_field)

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
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
            img, target = self.transforms(img, target)

        return img, target

    def __getitems__(self, indices):
        samples = [self.__getitem__(idx) for idx in indices]
        return samples

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

    def get_splits(self):
        splits = set()
        for sample in self.samples.iter_samples():
            split = sample.tags[0]  # Assuming the split is the first tag
            splits.add(split)
        return splits


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
    gt_field = torch_dataset.gt_field
    labels_map_rev = torch_dataset.labels_map_rev
    samples_data = {
        sample.filepath: {
            "tags": sample.tags,
            "metadata": sample.metadata,
            "detections": sample[gt_field].detections,
        }
        for sample in torch_dataset.samples.iter_samples()
    }

    def _gen():
        for idx, img_path in enumerate(img_paths):
            sample_data = samples_data[img_path]
            split = sample_data["tags"][0]
            if split != split_name:
                continue

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
