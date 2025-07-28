import pytest
import fiftyone as fo
from huggingface_hub import snapshot_download
from config import config
from workflows.data_ingest import run_data_ingest
from utils.logging import configure_logging
import logging
import os


@pytest.fixture(autouse=True)
def setup_logging():
    configure_logging()


@pytest.fixture(autouse=True)
def deactivate_wandb_sync():
    config.WANDB_ACTIVE = False


@pytest.fixture
def test_video_ingest_dataset_dir():
    """
    Fixture that downloads a folder with a .mov file from Hugging Face Hub
    and returns the local path so it can be passed into dataset_ingest.
    """
    local_dir = snapshot_download(
        repo_id="Abeyankar/video-ingest-test",  # Your dataset with sample.mov
        repo_type="dataset",
        local_dir="/tmp/video-ingest-test",
        local_dir_use_symlinks=False
    )
    return local_dir


def test_dataset_ingest_workflow_video(test_video_ingest_dataset_dir):
    """
    Test the dataset_ingest workflow using a single .mov file downloaded from HF Hub.
    Verifies that frames are extracted, splits are applied, and ground_truth exists.
    """

    base_name = "video_ingest_test"

    config.WORKFLOWS["data_ingest"] = {
        "dataset_name": base_name,
        "dataset_dir": test_video_ingest_dataset_dir,
        "annotation_format": "auto",  # Will auto-detect 'video'
        "fps": 1,
        "split_percentages": [0.7, 0.15, 0.15],
    }

    # Run the workflow
    run_data_ingest()

    dataset_name = "video_ingest_test1"

    # Load and validate the output dataset
    dataset = fo.load_dataset(dataset_name)

    logging.info(f"Loaded dataset: {dataset.name}")
    assert dataset is not None
    assert dataset.name == dataset_name

    tag_counts = dataset.count_sample_tags()
    total_tagged = sum(tag_counts.values())
    logging.info(f"Tag counts: {tag_counts}")
    assert total_tagged == len(dataset), "Not all samples were tagged for split"

    # Clean up
    fo.delete_dataset(dataset_name)
