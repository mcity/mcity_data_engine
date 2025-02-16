import pytest
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
from main import workflow_mask_teacher
from config.config import WORKFLOWS  

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

# @pytest.mark.parametrize("seg_model", [
#     "segment-anything-2.1-hiera-tiny-image-torch",
# ])
# def test_semantic_segmentation_model(dataset_v51, seg_model):
#     WORKFLOWS["mask_teacher"]["semantic_segmentation"]["sam2"]["models"] = [seg_model]
    
#     print("Testing semantic segmentation model:", seg_model)
#     assert len(WORKFLOWS["mask_teacher"]["semantic_segmentation"]["sam2"]["models"]) == 1, \
#         "Expected only one semantic segmentation model"
    
#     dataset_info = {"name": "fisheye8k_mask_test"}

#     print("\n[TEST] Starting workflow_mask_teacher integration test for segmentation model")
#     print(f"[TEST] Dataset Name: {dataset_v51.name}")
#     print(f"[TEST] Number of samples in dataset: {len(dataset_v51)}")

#     try:
#         output = workflow_mask_teacher(dataset_v51, dataset_info)
#         if output is None:
#             pytest.fail("workflow_mask_teacher returned None")
#         print("[TEST] workflow_mask_teacher completed successfully!")
#     except Exception as e:
#         pytest.fail(f"workflow_mask_teacher raised an exception: {e}")

#     assert isinstance(output, fo.Dataset), "Expected output to be a FiftyOne dataset"
#     assert len(output) > 0, "The dataset should not be empty after processing"
#     print(f"[TEST] Output dataset contains {len(output)} samples.")
#     print("[TEST] Segmentation model test passed successfully!")

# @pytest.mark.parametrize("depth_model", [
#     "Intel/dpt-swinv2-tiny-256",
# ])
# def test_depth_estimation_model(dataset_v51, depth_model):
#     WORKFLOWS["mask_teacher"]["depth_estimation"]["dpt"]["models"] = {depth_model}
    
#     print("Testing depth estimation model:", depth_model)
#     assert len(WORKFLOWS["mask_teacher"]["depth_estimation"]["dpt"]["models"]) == 1, \
#         "Expected only one depth estimation model"
    
#     dataset_info = {"name": "fisheye8k_mask_test"}

#     print("\n[TEST] Starting workflow_mask_teacher integration test for depth estimation model")
#     print(f"[TEST] Dataset Name: {dataset_v51.name}")
#     print(f"[TEST] Number of samples in dataset: {len(dataset_v51)}")

#     try:
#         output = workflow_mask_teacher(dataset_v51, dataset_info)
#         if output is None:
#             pytest.fail("workflow_mask_teacher returned None")
#         print("[TEST] workflow_mask_teacher completed successfully!")
#     except Exception as e:
#         pytest.fail(f"workflow_mask_teacher raised an exception: {e}")

#     assert isinstance(output, fo.Dataset), "Expected output to be a FiftyOne dataset"
#     assert len(output) > 0, "The dataset should not be empty after processing"
#     print(f"[TEST] Output dataset contains {len(output)} samples.")
#     print("[TEST] Depth estimation model test passed successfully!")


def test_mask_teacher(dataset_v51):
    print("Inside test_mask_teacher")

    WORKFLOWS["mask_teacher"]["semantic_segmentation"]["sam2"]["models"] = [
        "segment-anything-2.1-hiera-tiny-image-torch"
    ]
    WORKFLOWS["mask_teacher"]["depth_estimation"]["dpt"]["models"] = {
        "Intel/dpt-swinv2-tiny-256"
    }

    assert len(WORKFLOWS["mask_teacher"]["semantic_segmentation"]["sam2"]["models"]) == 1, "Expected only one semantic segmentation model"
    assert len(WORKFLOWS["mask_teacher"]["depth_estimation"]["dpt"]["models"]) == 1, "Expected only one depth estimation model"

    dataset_info = {"name": "fisheye8k_mask_test"}

    print("\n[TEST] Starting workflow_mask_teacher integration test")
    print(f"[TEST] Dataset Name: {dataset_v51.name}")
    print(f"[TEST] Number of samples in dataset: {len(dataset_v51)}")

    try:
        output = workflow_mask_teacher(dataset_v51, dataset_info)
        if output is None:
            pytest.fail("workflow_mask_teacher returned None")
        print("[TEST] workflow_mask_teacher completed successfully!")
    except Exception as e:
        pytest.fail(f"workflow_mask_teacher raised an exception: {e}")

    assert isinstance(output, fo.Dataset), "Expected output to be a FiftyOne dataset"
    assert len(output) > 0, "The dataset should not be empty after processing"

    print(f"[TEST] Output dataset contains {len(output)} samples.")
    print("[TEST] All selected models passed successfully!")
