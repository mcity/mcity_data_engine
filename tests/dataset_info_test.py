from utils.dataset_loader import load_dataset_info


def test_load_dataset_info():
    dataset_name = "mcity_fisheye_2000"
    dataset_info = load_dataset_info(dataset_name)

    # Add assertions to verify the correctness of the loaded data
    assert dataset_info is not None, "Dataset info should not be None"
    assert isinstance(dataset_info, dict), "Dataset info should be a dictionary"
    assert "name" in dataset_info, "Dataset info should contain 'name' key"
    assert (
        dataset_info["name"] == dataset_name
    ), f"Dataset name should be {dataset_name}"
