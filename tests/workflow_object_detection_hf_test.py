import logging
import os
import shutil

import fiftyone as fo
import pytest
from fiftyone.utils.huggingface import load_from_hub

import config.config
from main import workflow_auto_labeling_hf
from utils.data_loader import FiftyOneTorchDatasetCOCO, TorchToHFDatasetCOCO


@pytest.fixture(autouse=True)
def setup_logging():
    configure_logging()


@pytest.fixture(autouse=True)
def deactivate_hf_sync():
    config.config.HF_DO_UPLOAD = False


@pytest.fixture(autouse=True)
def deactivate_wandb_sync():
    config.config.WANDB_ACTIVE = False


import random

from config.config import ACCEPTED_SPLITS
from utils.dataset_loader import _post_process_dataset
from utils.logging import configure_logging


@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_v51_hf_od_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=50, name=dataset_name
        )
        dataset = _post_process_dataset(dataset)
        for sample in dataset.iter_samples(progress=True, autosave=True):
            split = random.choice(ACCEPTED_SPLITS)
            sample.tags = [split]
    except:
        dataset = fo.load_dataset(dataset_name)

    assert dataset is not None, "Failed to load or create the FiftyOne dataset"
    return dataset


@pytest.fixture
def dataset_hf(dataset_v51):

    pytorch_dataset = FiftyOneTorchDatasetCOCO(dataset_v51)
    pt_to_hf_converter = TorchToHFDatasetCOCO(pytorch_dataset)
    hf_dataset = pt_to_hf_converter.convert()

    return hf_dataset


@pytest.mark.parametrize("mode", ["train", "inference", "inference_hf"])
def test_hf_object_detection(dataset_v51, dataset_hf, mode):
    try:
        MODEL_NAME = "microsoft/conditional-detr-resnet-50"

        if mode == "inference_hf":
            selected_mode = ["inference"]
            # Delete the whole folder to force download from HF
            folder = f"./output/models/object_detection_hf/fisheye8k_v51_hf_od_test/{MODEL_NAME.replace("-","_")}"
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"Deleted folder: {folder}")
            else:
                print(f"Folder does not exist: {folder}")

        else:
            selected_mode = [mode]

        print(f"Test for mode {mode}")

        run_config = {
            "mode": selected_mode,
            "model_name": MODEL_NAME,
            "v51_dataset_name": dataset_v51.name,
            "epochs": 1,
            "early_stop_patience": 1,
            "early_stop_threshold": 0,
            "learning_rate": 5e-05,
            "weight_decay": 0.0001,
            "max_grad_norm": 0.01,
            "batch_size": 1,
            "image_size": [16, 16],
            "n_worker_dataloader": 1,
            "inference_settings": {
                "do_eval": True,
                "inference_on_evaluation": True,
                "model_hf": None,  # None (automatic selection) or Hugging Face ID
                "detection_threshold": 0.2,
            },
        }

        # Delete existing field
        try:
            dataset_v51.delete_sample_field(
                "pred_od_microsoft_conditional_detr_resnet_50_fisheye8k_v51_hf_od_test"
            )
        except Exception as e:
            print(f"Field was not deleted: {e}")

        workflow_auto_labeling_hf(
            dataset_v51, dataset_hf, run_config, wandb_activate=False
        )

        if mode != "train":
            # Check results
            # Get fields with predictions
            predictions_fields = []
            fields = dataset_v51.get_field_schema()
            for field in fields:
                if "pred_" in field:
                    print(f"Field with detections: {field}")
                    predictions_fields.append(field)

            assert len(predictions_fields) > 0, "No fields with predictions found"
            assert (
                len(predictions_fields) == 1
            ), f"More than one detection field found: {predictions_fields}"

            # Only test requested samples
            if run_config["inference_settings"]["inference_on_evaluation"] is True:
                dataset_view = dataset_v51.match_tags(["test", "val"])
            else:
                dataset_view = dataset_v51

            found_detections = 0
            for sample in dataset_view:
                for field in predictions_fields:
                    if sample[field]:
                        found_detections += len(sample[field].detections)

            assert found_detections > 0, "No detections found"

    except Exception as e:
        pytest.fail(f"Mode {mode} failed: {e}.")
