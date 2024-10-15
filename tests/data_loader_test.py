import pytest
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
from utils.data_loader import FiftyOneTorchDatasetCOCO, TorchToHFDatasetCOCO
from PIL import Image


@pytest.fixture
def dataset_v51():
    return load_from_hub("dbogdollumich/mcity_fisheye_v51", max_samples=100)


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
    assert "background" in classes


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
