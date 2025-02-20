import fiftyone as fo
import pytest
from fiftyone.utils.huggingface import load_from_hub

from config.config import WORKFLOWS
from main import workflow_auto_label_mask


@pytest.fixture
def dataset_v51():
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_mask_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=1, name=dataset_name
        )
    except Exception as e:
        print(f"Error loading from hub: {e}")
        dataset = fo.load_dataset(dataset_name)
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"
    return dataset


def test_auto_label_mask(dataset_v51):

    WORKFLOWS["auto_label_mask"]["semantic_segmentation"]["sam2"]["models"] = [
        "segment-anything-2.1-hiera-tiny-image-torch"
    ]
    WORKFLOWS["auto_label_mask"]["depth_estimation"]["dpt"]["models"] = {
        "Intel/dpt-swinv2-tiny-256"
    }

    dataset_info = {"name": "fisheye8k_mask_test"}

    print("\n[TEST] Starting workflow_auto_label_mask integration test")
    print(f"[TEST] Dataset Name: {dataset_v51.name}")
    print(f"[TEST] Number of samples: {len(dataset_v51)}")

    config = WORKFLOWS["auto_label_mask"]

    try:
        output = workflow_auto_label_mask(dataset_v51, dataset_info, config)
        assert output is not None, "workflow_auto_label_mask returned None"
        print("[TEST] workflow_auto_label_mask completed successfully!")
    except Exception as e:
        pytest.fail(f"workflow_auto_label_mask raised an exception: {e}")

    assert isinstance(output, fo.Dataset), "Expected output to be a FiftyOne dataset"
    assert len(output) > 0, "The dataset should not be empty after processing"

    for sample in output:

        expected_depth_field = "pred_de_Intel_dpt_swinv2_tiny_256"

        expected_sam_field = "pred_ss_segment_anything_2_1_hiera_tiny_image_torch_noprompt"

        assert hasattr(sample, expected_depth_field), (
            f"Missing depth field '{expected_depth_field}' on sample ID {sample.id}"
        )
        assert hasattr(sample, expected_sam_field), (
            f"Missing segmentation field '{expected_sam_field}' on sample ID {sample.id}"
        )
        break  # only loaded 1 sample

    print("[TEST] Verified that new fields are present in the dataset.")
