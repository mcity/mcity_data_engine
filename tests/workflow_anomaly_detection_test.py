import fiftyone as fo
import pytest
from fiftyone.utils.huggingface import load_from_hub


@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = (
        "Voxel51/fisheye8k"  # TODO Upload anomaly dataset to Hugging Face
    )
    dataset_name = "fisheye8k_v51_anomaly_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=200, name=dataset_name
        )
    except:
        dataset = fo.load_dataset(dataset_name)
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"
    return dataset


def test_anomaly_detection(dataset_v51):
    pass
    # TODO Run and test PADIM model, it does not require training
    # Check if any masks exist with values != 0
