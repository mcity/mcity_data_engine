import pytest
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
from utils.data_loader import FiftyOneTorchDatasetCOCO, TorchToHFDatasetCOCO
from PIL import Image
from torch.utils.data import DataLoader
from datasets import Dataset, Features, Array2D, ClassLabel
import torch


@pytest.fixture
def dataset_v51():
    dataset_name = "dbogdollumich/mcity_fisheye_v51"
    try:
        dataset = load_from_hub(dataset_name, max_samples=100)
    except:
        dataset = fo.load_dataset(dataset_name)
    return dataset


@pytest.fixture
def torch_dataset(dataset_v51):
    return FiftyOneTorchDatasetCOCO(dataset_v51)


def test_torch_dataset_length(torch_dataset):
    assert len(torch_dataset) == 100


def test_torch_dataset_getitem(torch_dataset):
    img, target = torch_dataset[0]
    assert isinstance(img, Image.Image)
    assert "boxes" in target
    assert "labels" in target
    assert "image_id" in target
    assert "area" in target
    assert "iscrowd" in target


def test_torch_dataset_getitems(torch_dataset):
    samples = torch_dataset.__getitems__([0, 1, 2])
    assert len(samples) == 3
    for img, target in samples:
        assert isinstance(img, Image.Image)
        assert "boxes" in target


@pytest.fixture
def dataset_v51():
    dataset_name = "dbogdollumich/mcity_fisheye_v51"
    try:
        dataset = load_from_hub(dataset_name, max_samples=100)
    except:
        dataset = fo.load_dataset(dataset_name)
    return dataset


@pytest.fixture
def torch_dataset(dataset_v51):
    return FiftyOneTorchDatasetCOCO(dataset_v51)


def test_torch_dataset_length(torch_dataset):
    assert len(torch_dataset) == 100


def test_torch_dataset_getitem(torch_dataset):
    img, target = torch_dataset[0]
    assert isinstance(img, Image.Image)
    assert "boxes" in target
    assert "labels" in target
    assert "image_id" in target
    assert "area" in target
    assert "iscrowd" in target


def test_torch_dataset_getitems(torch_dataset):
    samples = torch_dataset.__getitems__([0, 1, 2])
    assert len(samples) == 3
    for img, target in samples:
        assert isinstance(img, Image.Image)
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target
        assert "area" in target
        assert "iscrowd" in target


def test_torch_dataset_get_classes(torch_dataset):
    classes = torch_dataset.get_classes()
    assert isinstance(classes, list)


def test_torch_dataset_get_splits(torch_dataset):
    splits = torch_dataset.get_splits()
    assert isinstance(splits, set)
    assert "train" in splits or "val" in splits


@pytest.fixture
def converter_torch_hf(torch_dataset):
    return TorchToHFDatasetCOCO(torch_dataset)


def test_hf_dataset_conversion(converter_torch_hf):
    hf_dataset = converter_torch_hf.convert()

    @pytest.fixture
    def dataset_v51():
        dataset_name = "dbogdollumich/mcity_fisheye_v51"
        try:
            dataset = load_from_hub(dataset_name, max_samples=100)
        except:
            dataset = fo.load_dataset(dataset_name)
        return dataset

    @pytest.fixture
    def torch_dataset(dataset_v51):
        return FiftyOneTorchDatasetCOCO(dataset_v51)

    def test_torch_dataset_length(torch_dataset):
        assert len(torch_dataset) == 100

    def test_torch_dataset_getitem(torch_dataset):
        img, target = torch_dataset[0]
        assert isinstance(img, Image.Image)
        assert "boxes" in target
        assert "labels" in target
        assert "image_id" in target
        assert "area" in target
        assert "iscrowd" in target

    def test_torch_dataset_getitems(torch_dataset):
        samples = torch_dataset.__getitems__([0, 1, 2])
        assert len(samples) == 3
        for img, target in samples:
            assert isinstance(img, Image.Image)
            assert "boxes" in target


def test_torch_dataset_get_classes(torch_dataset):
    classes = torch_dataset.get_classes()
    assert isinstance(classes, list)


def test_torch_dataset_get_splits(torch_dataset):
    splits = torch_dataset.get_splits()
    assert isinstance(splits, set)
    assert "train" in splits or "val" in splits


@pytest.fixture
def converter_torch_hf(torch_dataset):
    return TorchToHFDatasetCOCO(torch_dataset)


def test_hf_dataset_conversion(converter_torch_hf):
    hf_dataset = converter_torch_hf.convert()
    assert "train" in hf_dataset or "val" in hf_dataset


def test_hf_dataset_sample(converter_torch_hf):
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


@pytest.fixture
def dataloader(torch_dataset):
    return DataLoader(torch_dataset, batch_size=4, shuffle=True)


def test_dataloader_length(dataloader, torch_dataset):
    assert len(dataloader) == (len(torch_dataset) + 3) // 4


def test_dataloader_batch(dataloader):
    for batch in dataloader:
        imgs, targets = batch
        assert len(imgs) == 4
        assert len(targets) == 4
        for img, target in zip(imgs, targets):
            assert isinstance(img, Image.Image)
            assert "boxes" in target
            assert "labels" in target
            assert "image_id" in target
            assert "area" in target
            assert "iscrowd" in target
            break  # Only test the first batch

    def test_hf_dataset_with_format(converter_torch_hf):
        hf_dataset = converter_torch_hf.convert()
        if "train" in hf_dataset:
            hf_dataset["train"] = hf_dataset["train"].with_format("torch")
            sample = hf_dataset["train"][0]
            assert isinstance(sample["image"], torch.Tensor)
            assert isinstance(sample["target"]["bbox"], torch.Tensor)
            assert isinstance(sample["target"]["category_id"], torch.Tensor)
        if "val" in hf_dataset:
            hf_dataset["val"] = hf_dataset["val"].with_format("torch")
            sample = hf_dataset["val"][0]
            assert isinstance(sample["image"], torch.Tensor)
            assert isinstance(sample["target"]["bbox"], torch.Tensor)
            assert isinstance(sample["target"]["category_id"], torch.Tensor)

    def test_hf_dataset_dataloader(converter_torch_hf):
        hf_dataset = converter_torch_hf.convert()
        if "train" in hf_dataset:
            hf_dataset["train"] = hf_dataset["train"].with_format("torch")
            dataloader = DataLoader(hf_dataset["train"], batch_size=4)
            for batch in dataloader:
                assert isinstance(batch["image"], torch.Tensor)
                assert isinstance(batch["target"]["bbox"], torch.Tensor)
                assert isinstance(batch["target"]["category_id"], torch.Tensor)
                break  # Only test the first batch
        if "val" in hf_dataset:
            hf_dataset["val"] = hf_dataset["val"].with_format("torch")
            dataloader = DataLoader(hf_dataset["val"], batch_size=4)
            for batch in dataloader:
                assert isinstance(batch["image"], torch.Tensor)
                assert isinstance(batch["target"]["bbox"], torch.Tensor)
                assert isinstance(batch["target"]["category_id"], torch.Tensor)
        break  # Only test the first batch
