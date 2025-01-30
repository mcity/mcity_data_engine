import fiftyone as fo
import pytest
from fiftyone.utils.huggingface import load_from_hub


@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_brain_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=2, name=dataset_name
        )
    except:
        dataset = fo.load_dataset(dataset_name)
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"
    return dataset

def test_mask_teacher(dataset_v51):
    pass