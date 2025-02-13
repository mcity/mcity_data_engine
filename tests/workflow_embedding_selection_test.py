import fiftyone as fo
import pytest
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub

from main import workflow_embedding_selection
from utils.dataset_loader import load_dataset_info
from workflows.embedding_selection import BRAIN_TAXONOMY


@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_v51_brain_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=50, name=dataset_name
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

    config = {
        "mode": "train",
        "parameters": {
            "compute_representativeness": 0.99,
            "compute_unique_images_greedy": 0.01,
            "compute_unique_images_deterministic": 0.99,
            "compute_similar_images": 0.03,
            "neighbour_count": 3,
        },
    }

    wandb_activate = False

    workflow_embedding_selection(
        dataset_v51, dataset_info, MODEL_NAME, config, wandb_activate
    )

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
