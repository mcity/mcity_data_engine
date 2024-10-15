import torch
from PIL import Image
import fiftyone.utils.coco as fouc

import datasets
from datasets import Dataset, NamedSplit


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

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

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
            boxes.append([x, y, x + w, y + h])
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
        torch_dataset (Dataset): The PyTorch dataset to be converted.

    Methods:
        convert():
            Converts the PyTorch dataset to a Hugging Face dataset.
        
        gen_factory(dataset, split_name):
            Generates a generator function for a specific split of the dataset.
        
        create_target(sample, dataset, idx):
            Creates the target dictionary for a given sample.
    """

        """
        Initializes the TorchToHFDatasetCOCO with the given PyTorch dataset.

        Args:
            torch_dataset (Dataset): The PyTorch dataset to be converted.
        """

        """
        Converts the PyTorch dataset to a Hugging Face dataset.

        Returns:
            dict: A dictionary where keys are split names and values are Hugging Face datasets.
        """

        """
        Generates a generator function for a specific split of the dataset.

        Args:
            dataset (Dataset): The PyTorch dataset.
            split_name (str): The name of the split.

        Returns:
            function: A generator function that yields samples for the specified split.
        """



        """
        Creates the target dictionary for a given sample.

        Args:
            sample (Sample): A sample from the dataset.
            dataset (Dataset): The PyTorch dataset.
            idx (int): The index of the sample.

        Returns:
            dict: A dictionary containing bounding boxes, labels, image ID, area, and iscrowd information.
        """


    def __init__(self, torch_dataset):
        self.torch_dataset = torch_dataset

    def convert(self):
        splits = self.torch_dataset.get_splits()
        hf_dataset = {
            split: Dataset.from_generator(
                self.gen_factory(self.torch_dataset, split),
                split=NamedSplit(split),
            )
            for split in splits
        }
        return hf_dataset

    def gen_factory(self, dataset, split_name):
        def gen():
            for idx, img_path in enumerate(dataset.img_paths):
                sample = dataset.samples[img_path]
                split = sample.tags[0]
                if split != split_name:
                    continue

                target = self.create_target(sample, dataset, idx)
                yield {"image": img_path, "target": target, "split": split}

        return gen

    def create_target(self, sample, dataset, idx):
        metadata = sample.metadata
        detections = sample[dataset.gt_field].detections

        boxes = [
            [
                det.bounding_box[0],
                det.bounding_box[1],
                det.bounding_box[0] + det.bounding_box[2],
                det.bounding_box[1] + det.bounding_box[3],
            ]
            for det in detections
        ]
        labels = [dataset.labels_map_rev[det.label] for det in detections]
        area = [det.bounding_box[2] * det.bounding_box[3] for det in detections]
        iscrowd = [0 for _ in detections]  # Assuming iscrowd is not available

        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd,
        }
