import fiftyone as fo
import pytest
from fiftyone.utils.huggingface import load_from_hub
import config.config

from main import workflow_auto_label_mask
from utils.logging import configure_logging

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

@pytest.fixture(autouse=True)
def setup_logging():
    configure_logging()

@pytest.fixture(autouse=True)
def deactivate_wandb_sync():
    config.config.WANDB_ACTIVE = False

@pytest.mark.parametrize("workflow_config", [
    {   # Test 1: SAM2 without a prompt
        "semantic_segmentation": {
            "sam2": {
                "prompt_field": None,
                "models": ["segment-anything-2.1-hiera-tiny-image-torch"]
            }
        },
        "depth_estimation": {}
    },
    {   # Test 2: SAM2 with a prompt
        "semantic_segmentation": {
            "sam2": {
                "prompt_field": "detections",
                "models": ["segment-anything-2.1-hiera-tiny-image-torch"]
            }
        },
        "depth_estimation": {}
    },
    {   # Test 3: Depth Estimation
        "semantic_segmentation": {},
        "depth_estimation": {
            "dpt": {
                "models": {"Intel/dpt-swinv2-tiny-256"}
            }
        }
    },
])

def test_auto_label_mask(dataset_v51, workflow_config):

    dataset_info = {"name": "fisheye8k_mask_test"}

    print("\n[TEST] Starting workflow_auto_label_mask integration test")
    print(f"[TEST] Dataset Name: {dataset_v51.name}")
    print(f"[TEST] Number of samples: {len(dataset_v51)}")


    # Run if valid config
    if workflow_config["semantic_segmentation"] or workflow_config["depth_estimation"]:
        workflow_auto_label_mask(dataset_v51, dataset_info, workflow_config)
        print("[TEST] workflow_auto_label_mask completed successfully!")
    else:
        print("[TEST] Skipping workflow, running assertion checks on unmodified dataset.")

    assert len(dataset_v51) > 0, "The dataset should not be empty after processing"

    for sample in dataset_v51:
        # Empty Config Test: NO new fields were added
        if not workflow_config["semantic_segmentation"] and not workflow_config["depth_estimation"]:
            expected_depth_field = "pred_de_Intel_dpt_swinv2_tiny_256"
            expected_sam_field = "pred_ss_segment_anything_2_1_hiera_tiny_image_torch_noprompt"

            assert not hasattr(sample, expected_depth_field), (
                f"[ERROR] Unexpected depth field '{expected_depth_field}' found on sample ID {sample.id}"
            )

            assert hasattr(sample, expected_sam_field), (
                f"[ERROR] Semantic mask '{expected_sam_field}' not found on sample ID {sample.id}"
            )

            bbox = getattr(sample, "bounding_box_field", None)
            semantic_mask = getattr(sample, expected_sam_field, None)

            assert bbox is not None, f"[ERROR] No bounding box found for sample ID {sample.id}"
            assert semantic_mask is not None, f"[ERROR] No semantic mask found for sample ID {sample.id}"

            assert (
                bbox.contains(semantic_mask)
            ), f"[ERROR] Semantic mask is outside the bounding box for sample ID {sample.id}"

            assert (
                bbox.class_name == semantic_mask.class_name
            ), f"[ERROR] Class name mismatch: bbox='{bbox.class_name}', mask='{semantic_mask.class_name}' on sample ID {sample.id}"

    print("[TEST] Verified that new fields are present in the dataset.")
