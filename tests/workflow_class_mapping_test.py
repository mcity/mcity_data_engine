# workflow_class_mapping_test.py
import fiftyone as fo
import pytest
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub
from main import workflow_class_mapping
from utils.dataset_loader import load_dataset_info

max_samples=1

@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_v51_class_mapping_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=max_samples, name=dataset_name
        )
    except:
        dataset = fo.load_dataset(dataset_name)
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"
    return dataset

def test_class_mapping(dataset_v51, sample_index=0):
    """
    Test workflow on a single selected sample from the dataset, verifying that each model
    has added its specific tag.
    """

    first_sample = dataset_v51.first()

    # Select one sample by index
    sample = dataset_v51[sample_index]
    assert sample is not None, "Selected sample not found in dataset"

    # Optional: store original tags for debugging
    original_tags = sample.count_values("detections.detections.tags")

    # Execute workflow
    dataset_info = load_dataset_info("fisheye8k")  # Use loader for actual dataset
    dataset_info["name"] = "fisheye8k_v51_class_mapping_test"  # Update for local tests
    workflow_class_mapping(dataset_v51, dataset_info)

    # Retrieve updated sample
    updated_sample = dataset_v51[sample_index]

    # List of expected model identifiers from config
    expected_models = [
        "Salesforce/blip2-itm-vit-g",
        "openai/clip-vit-large-patch14",
        "google/siglip-so400m-patch14-384",
        "kakaobrain/align-base",
        "BAAI/AltCLIP",
        "CIDAS/clipseg-rd64-refined"
    ]
    # Construct expected tags assuming a "newclass-<model_id>" format
    expected_tags = [f"newclass-{model}" for model in expected_models]

    # Gather all tags from all detections in the updated sample
    found_tags = set()
    for detection in updated_sample.detections:
        if detection.tags:
            found_tags.update(detection.tags)

    # Validate that each expected tag is present
    for expected_tag in expected_tags:
        assert expected_tag in found_tags, f"Tag for model {expected_tag} not found in detections"
