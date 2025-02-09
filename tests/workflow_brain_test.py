import random

import fiftyone as fo
import pytest
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub

from main import workflow_brain_selection
from utils.dataset_loader import load_dataset_info
from workflows.brain import BRAIN_TAXONOMY


# Test might be too slow for GitHub Actions CI
@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_v51_brain_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=200, name=dataset_name
        )
    except:
        dataset = fo.load_dataset(dataset_name)
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"
    return dataset


def test_brain(dataset_v51):
    MODEL_NAME = "mobilenet-v2-imagenet-torch"
    dataset_info = load_dataset_info("fisheye8k")  # Use loader for actual dataset
    dataset_info["name"] = (
        "fisheye8k_v51_brain_test"  # Update with test name for local tests where both exist
    )
    workflow_brain_selection(dataset_v51, dataset_info, MODEL_NAME)

    # Check number of selected samples
    results_field = BRAIN_TAXONOMY["field"]
    n_samples_selected = 0
    for key in BRAIN_TAXONOMY:
        if "value" in key:
            value = BRAIN_TAXONOMY[key]
            view_result = dataset_v51.match(F(results_field) == value)
            n_samples = len(view_result)
            print(f"Found {n_samples} samples for {results_field}/{value}")
            n_samples_selected += n_samples

    # Assert if no samples were selected
    assert n_samples_selected != 0, "No samples were selected"
