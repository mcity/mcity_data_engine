import random

import fiftyone as fo
import pytest
import torch
from fiftyone.utils.huggingface import load_from_hub
from torch.utils.data import DataLoader

from utils.data_loader import FiftyOneTorchDatasetCOCO, TorchToHFDatasetCOCO


@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "dbogdollumich/mcity_fisheye_v51"
    dataset_name = "mcity_fisheye_v51_pytest"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=100, name=dataset_name
        )
        for sample in dataset:
            sample.tags = random.choice([["train"], ["val"]])
    except:
        dataset = fo.load_dataset(dataset_name)
    return dataset


def test_dataset_v51(dataset_v51):
    assert dataset_v51 is not None


# Tests for torch dataset
@pytest.fixture
def torch_dataset(dataset_v51):
    """Fixture to create a FiftyOneTorchDatasetCOCO instance."""
    return FiftyOneTorchDatasetCOCO(dataset_v51)


def test_torch_dataset_length(torch_dataset):
    """Test the length of the torch dataset."""
    assert len(torch_dataset) == 100


@pytest.mark.parametrize("index", [0, 1, 2])
def test_torch_dataset_getitem(torch_dataset, index):
    """Test getting an item from the torch dataset."""
    img, target = torch_dataset[index]
    assert isinstance(img, torch.Tensor)
    assert "bbox" in target
    assert "category_id" in target
    assert "image_id" in target
    assert "area" in target
    assert "iscrowd" in target


def test_torch_dataset_getitem_invalid_index(torch_dataset):
    """Test getting an item with an invalid index from the torch dataset."""
    with pytest.raises(IndexError):
        torch_dataset[1000]


def test_torch_dataset_getitems(torch_dataset):
    """Test getting multiple items from the torch dataset."""
    samples = torch_dataset.__getitems__([0, 1, 2])
    assert len(samples) == 3
    for img, target in samples:
        assert isinstance(img, torch.Tensor)
        assert "bbox" in target


def test_torch_dataset_getitems_invalid_indices(torch_dataset):
    """Test getting multiple items with invalid indices from the torch dataset."""
    with pytest.raises(IndexError):
        torch_dataset.__getitems__([1000, 1001])


def test_torch_dataset_get_classes(torch_dataset):
    """Test getting classes from the torch dataset."""
    classes = torch_dataset.get_classes()
    assert isinstance(classes, list)


def test_torch_dataset_get_splits(torch_dataset):
    """Test getting splits from the torch dataset."""
    splits = torch_dataset.get_splits()
    assert isinstance(splits, set)
    assert "train" in splits or "val" in splits


# Tests for torch dataloader
@pytest.fixture
def dataloader(torch_dataset):
    """Fixture to create a DataLoader instance."""
    return DataLoader(
        torch_dataset,
        batch_size=4,
        collate_fn=lambda batch: list(zip(*batch)),
        shuffle=True,
    )


def test_dataloader_length(dataloader, torch_dataset):
    """Test the length of the dataloader."""
    assert len(dataloader) == (len(torch_dataset) + 3) // 4


def test_dataloader_batch(dataloader):
    """Test getting a batch from the dataloader."""
    for batch in dataloader:
        imgs, targets = batch
        assert len(imgs) == 4
        assert len(targets) == 4
        for img, target in zip(imgs, targets):
            assert isinstance(img, torch.Tensor)
            assert "bbox" in target
            assert "category_id" in target
            assert "image_id" in target
            assert "area" in target
            assert "iscrowd" in target


# Tests for HF dataset
@pytest.fixture
def converter_torch_hf(torch_dataset):
    """Fixture to create a TorchToHFDatasetCOCO instance."""
    return TorchToHFDatasetCOCO(torch_dataset)


def test_hf_dataset_conversion(converter_torch_hf):
    """Test converting the torch dataset to HF dataset."""
    hf_dataset = converter_torch_hf.convert()
    assert "train" in hf_dataset or "val" in hf_dataset


def test_hf_dataset_sample(converter_torch_hf):
    """Test getting a sample from the HF dataset."""
    hf_dataset = converter_torch_hf.convert()
    if "train" in hf_dataset:
        sample = hf_dataset["train"][0]
        assert "image" in sample
        assert "target" in sample
        assert "split" in sample
    if "val" in hf_dataset:
        sample = hf_dataset["val"][0]
        assert "image" in sample
        assert "target" in sample
        assert "split" in sample


def test_hf_dataset_dataloader(converter_torch_hf):
    """Test creating a DataLoader from the HF dataset."""
    hf_dataset = converter_torch_hf.convert()
    if "train" in hf_dataset:
        hf_dataset["train"] = hf_dataset["train"]
        dataloader = DataLoader(
            hf_dataset["train"],
            batch_size=4,
            collate_fn=lambda batch: (
                [item["image"] for item in batch],
                [item["target"] for item in batch],
                [item["split"] for item in batch],
            ),
        )
        for batch in dataloader:
            images, targets, splits = batch
            for img, target, split in zip(images, targets, splits):
                assert isinstance(img, str)
                assert isinstance(target["bbox"], list)
                assert isinstance(target["category_id"], list)
                assert isinstance(split, str)
    if "val" in hf_dataset:
        hf_dataset["val"] = hf_dataset["val"]
        dataloader = DataLoader(
            hf_dataset["val"],
            batch_size=4,
            collate_fn=lambda batch: (
                [item["image"] for item in batch],
                [item["target"] for item in batch],
                [item["split"] for item in batch],
            ),
        )
        for batch in dataloader:
            images, targets, splits = batch
            for img, target, split in zip(images, targets, splits):
                assert isinstance(img, str)
                assert isinstance(target["bbox"], list)
                assert isinstance(target["category_id"], list)
                assert isinstance(split, str)


def test_hf_dataset_with_format(converter_torch_hf):
    """Test setting the format of the HF dataset."""
    hf_dataset = converter_torch_hf.convert()
    if "train" in hf_dataset:
        hf_dataset["train"] = hf_dataset["train"].with_format("torch")
        sample = hf_dataset["train"][0]
        assert isinstance(sample["image"], str)  # Includes filepath
        assert isinstance(sample["target"]["bbox"], torch.Tensor)
        assert isinstance(sample["target"]["category_id"], torch.Tensor)
    if "val" in hf_dataset:
        hf_dataset["val"] = hf_dataset["val"].with_format("torch")
        sample = hf_dataset["val"][0]
        assert isinstance(sample["image"], str)
        assert isinstance(sample["target"]["bbox"], torch.Tensor)
        assert isinstance(sample["target"]["category_id"], torch.Tensor)


# Tests for incomplete datasets
@pytest.fixture
def empty_dataset():
    """Fixture to create an empty FiftyOne dataset."""
    try:
        dataset = fo.Dataset(name="empty_dataset")
    except:
        dataset = fo.load_dataset("empty_dataset")
    return dataset


@pytest.fixture
def no_annotations_dataset():
    """Fixture to create a FiftyOne dataset with no annotations."""
    try:
        dataset = fo.Dataset(name="no_annotations_dataset")
        dataset.add_sample(fo.Sample(filepath="image1.jpg"))
        dataset.add_sample(fo.Sample(filepath="image2.jpg"))
    except:
        dataset = fo.load_dataset("no_annotations_dataset")

    return dataset


def test_empty_dataset(empty_dataset):
    """Test creating a torch dataset from an empty FiftyOne dataset."""
    dataset = FiftyOneTorchDatasetCOCO(empty_dataset)
    assert len(dataset) == 0


def test_no_annotations_dataset(no_annotations_dataset):
    """Test creating a torch dataset from a FiftyOne dataset with no annotations."""
    dataset = FiftyOneTorchDatasetCOCO(no_annotations_dataset)
    assert len(dataset) == 2
