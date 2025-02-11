import fiftyone as fo
import pytest
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub
from config.config import WORKFLOWS

@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_v51_class_mapping_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=50, name=dataset_name
        )
    except:
        dataset = fo.load_dataset(dataset_name)
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"
    return dataset

def test_class_mapping(dataset_v51):
    from class_mapping import ClassMapper

    # Get configuration from the global config
    config = WORKFLOWS["class_mapping"]

    # Initialize mapper with test dataset and default model
    mapper = ClassMapper(
        dataset=dataset_v51,
        model_name=config["models"][-1],  # Use the last model in the list
        config=config
    )

    # Run mapping process
    stats = mapper.run_mapping()

    # Assertions to verify the results
    assert stats["total_processed"] > 0, "No samples were processed"
    assert "class_counts" in stats, "Class counts not recorded"

    # Check if any tags were added
    view_mapped = dataset_v51.match(F("detections.tags").contains("new_class"))
    assert len(view_mapped) > 0, "No new classifications were made"

    return stats
