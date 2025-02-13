# workflow_class_mapping_test.py
import fiftyone as fo
import pytest
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub

from main import workflow_class_mapping
from utils.dataset_loader import load_dataset_info

@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_v51_class_mapping_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=200, name=dataset_name
        )
    except:
        dataset = fo.load_dataset(dataset_name)
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"
    return dataset

def test_class_mapping(class_mapping_dataset):
    # Setup test configuration
    original_tags = class_mapping_dataset.count_values("detections.detections.tags")

    # Execute workflow
    dataset_info = load_dataset_info("fisheye8k")  # Use loader for actual dataset
    dataset_info["name"] = (
        "fisheye8k_v51_class_mapping_test"  # Update with test name for local tests where both exist
    )
    workflow_class_mapping(class_mapping_dataset, dataset_info)

    # Verify results
    updated_tags = class_mapping_dataset.count_values("detections.detections.tags")

    # Check that new tags were added
    new_tag_count = sum(count for tag, count in updated_tags.items()
                      if tag.startswith("new_class"))
    assert new_tag_count > 0, "No new class tags were added"

    # Check tag format
    for detection in class_mapping_dataset.iter_detections():
        for det in detection.detections:
            if det.tags:
                assert any(tag.startswith("new_class_") for tag in det.tags), \
                    "Invalid tag format"