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

    if workflow_config["semantic_segmentation"]:
        prompt_field = workflow_config["semantic_segmentation"]["sam2"]["prompt_field"]
    else:
        prompt_field = None

    expected_depth_field = "pred_de_Intel_dpt_swinv2_tiny_256"
    expected_sam_field_noprompt = "pred_ss_segment_anything_2_1_hiera_tiny_image_torch_noprompt"
    expected_sam_field_prompt = f"pred_ss_segment_anything_2_1_hiera_tiny_image_torch_prompt_{prompt_field}"

    field_gt = "detections"
    field_masks = "pred_ss_segment_anything_2_1_hiera_tiny_image_torch_prompt_detections"

    classes_gt = set()
    classes_sam = set()
    n_found_fields = 0

    for sample in dataset_v51:
        print(f"Fields in sample: {sample.field_names}")
        if workflow_config["semantic_segmentation"]:
            bboxes_gt = sample[field_gt]
            sam_masks = sample[field_masks]

            try:
                if prompt_field is not None:
                    for bbox in bboxes_gt.detections:
                        classes_gt.add(bbox.label)
                    for mask in sam_masks.detections:
                        classes_sam.add(mask.label)
                    field = sample[expected_sam_field_prompt]
                else:
                    field = sample[expected_sam_field_noprompt]
                n_found_fields += 1

            except Exception as e:
                print(f"Error: {e}")
                pass
        if workflow_config["depth_estimation"]:
            try:
                field = sample[expected_depth_field]
                n_found_fields += 1
            except:
                pass

    assert classes_gt == classes_sam, f"Classes in Ground Truth {classes_gt} and SAM Masks {classes_sam}"
    print(f"[TEST] Found {n_found_fields} new fields in the dataset")
    assert n_found_fields > 0, "No new fields were added to the dataset"

    print("Class Distribution in Ground Truth Bounding Boxes:")
    print(f"\tGround Truth Classes: {classes_gt}")
    print(f"\tSAM Masks: {classes_sam}")

    print("[TEST] Verified that new fields are present in the dataset.")
