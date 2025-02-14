import logging

import fiftyone as fo
import pytest
from fiftyone.utils.huggingface import load_from_hub

import config.config
from main import (
    configure_logging,
    workflow_ensemble_exploration,
    workflow_zero_shot_object_detection,
)


@pytest.fixture(autouse=True)
def deactivate_wandb_sync():
    config.config.WANDB_ACTIVE = False


@pytest.fixture(autouse=True)
def setup_logging():
    configure_logging()


@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_zero_shot_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=2, name=dataset_name
        )
    except:
        dataset = fo.load_dataset(dataset_name)
    dataset.persistent = True
    return dataset


def test_zero_shot_inference(dataset_v51):

    config = {
        "n_post_processing_worker_per_inference_worker": 1,
        "n_worker_dataloader": 1,
        "prefetch_factor_dataloader": 1,
        "hf_models_zeroshot_objectdetection": {
            "omlab/omdet-turbo-swin-tiny-hf": {
                "batch_size": 2,  # Test 2
                "n_dataset_chunks": 1,
            },
            "IDEA-Research/grounding-dino-tiny": {
                "batch_size": 1,
                "n_dataset_chunks": 1,
            },
            "google/owlvit-large-patch14": {
                "batch_size": 1,
                "n_dataset_chunks": 1,
            },
            "google/owlv2-base-patch16-finetuned": {
                "batch_size": 1,
                "n_dataset_chunks": 2,  # Test 2
            },
            "google/owlv2-large-patch14-ensemble": {
                "batch_size": 1,
                "n_dataset_chunks": 1,
            },
        },
        "detection_threshold": 0.2,
        "object_classes": [
            "skater",
            "child",
            "bicycle",
            "bicyclist",
            "cyclist",
            "bike",
            "rider",
            "motorcycle",
            "motorcyclist",
            "pedestrian",
            "person",
            "walker",
            "jogger",
            "runner",
            "skateboarder",
            "scooter",
            "vehicle",
            "car",
            "bus",
            "truck",
            "taxi",
            "van",
            "pickup truck",
            "trailer",
            "emergency vehicle",
            "delivery driver",
        ],
    }

    dataset_info = {
        "name": "fisheye8k_zero_shot_test",
        "v51_type": "FiftyOneDataset",
        "splits": [],
    }

    fields_of_interest_include = "pred_zsod_"
    dataset_fields = dataset_v51.get_field_schema()
    # Cleanup fields prior to run
    for field in dataset_fields:
        if fields_of_interest_include in field:
            dataset_v51.delete_sample_field(field)
            logging.info(f"Removed dataset field {field} prior to test.")

    workflow_zero_shot_object_detection(dataset_v51, dataset_info, config)

    # Get fields of dataset after run
    dataset_fields = dataset_v51.get_field_schema()
    logging.info(f"Dataset fields after run: {dataset_fields}")
    fields_of_interest = [
        field for field in dataset_fields if fields_of_interest_include in field
    ]

    logging.info(f"Zero Shot fields found: {fields_of_interest}")
    assert (
        len(fields_of_interest) > 0
    ), f"No fields that include '{fields_of_interest_include}' found"

    n_detections_found = dict.fromkeys(fields_of_interest, 0)

    for sample in dataset_v51.iter_samples(progress=True, autosave=False):
        for field in fields_of_interest:
            detections = sample[field].detections
            logging.info(detections)
            n_detections_found[field] += len(detections)
            for detection in detections:
                # Test if detection label is in allowed classes
                assert detection.label in config["object_classes"], (
                    f"Detection label '{detection.label}' not in configured object classes: "
                    f"{config['object_classes']}"
                )

                # Test bbox coordinates (x, y, width, height)
                bbox = detection.bounding_box
                assert (
                    0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1
                ), f"Bounding box coordinates (x={bbox[0]}, y={bbox[1]}) must be between 0 and 1"
                assert (
                    0 <= bbox[2] <= 1 and 0 <= bbox[3] <= 1
                ), f"Bounding box dimensions (w={bbox[2]}, h={bbox[3]}) must be between 0 and 1"
                assert (
                    bbox[0] + bbox[2] <= 1
                ), f"Bounding box extends beyond right image border: x({bbox[0]}) + width({bbox[2]}) > 1"
                assert (
                    bbox[1] + bbox[3] <= 1
                ), f"Bounding box extends beyond bottom image border: y({bbox[1]}) + height({bbox[3]}) > 1"

                # Test bbox area is reasonable (not too small or large)
                bbox_area = bbox[2] * bbox[3]
                assert (
                    0 <= bbox_area <= 1
                ), f"Bounding box area ({bbox_area}) is outside of range [0, 1]"

    # Check if all models detected something at all
    for field, n_detections in n_detections_found.items():
        assert n_detections > 0, (
            f"No detections found for model {field}. "
            f"Expected at least 1 detection, got {n_detections}"
        )
        logging.info(f"Found {n_detections} detections for model {field}")


def test_ensemble_exploration(dataset_v51):

    dataset_info = {
        "name": "fisheye8k_zero_shot_test",
        "v51_type": "FiftyOneDataset",
        "splits": [],
    }

    run_config = {
        "field_includes": "pred_zsod_",  # V51 field used for detections, "pred_zsod_" default for zero-shot object detection models
        "agreement_threshold": 3,  # Threshold for n models necessary for agreement between models
        "iou_threshold": 0.5,  # Threshold for IoU between bboxes to consider them as overlapping
        "max_bbox_size": 0.01,  # Value between [0,1] for the max size of considered bboxes
        "positive_classes": [  # Classes to consider, must be subset of available classes in the detections. Example for Vulnerable Road Users.
            "skater",
            "child",
            "bicycle",
            "bicyclist",
            "cyclist",
            "bike",
            "rider",
            "motorcycle",
            "motorcyclist",
            "pedestrian",
            "person",
            "walker",
            "jogger",
            "runner",
            "skateboarder",
            "scooter",
            "delivery driver",
        ],
    }

    workflow_ensemble_exploration(
        dataset_v51, dataset_info, run_config, wandb_activate=False
    )

    fields_of_interest_include = "pred_zsod_"
    dataset_fields = dataset_v51.get_field_schema()
    fields_of_interest = [
        field for field in dataset_fields if fields_of_interest_include in field
    ]

    n_unique_field = "n_unique_exploration"
    for sample in dataset_v51.iter_samples(progress=True, autosave=False):
        # Check if field was populated with unique instances
        n_unique = sample[n_unique_field]
        assert n_unique > 0, f"No unique instances found in sample {sample}"
        n_samples_with_tags = 0
        # Check if detections were tagged
        for field in fields_of_interest:
            detections = sample[field].detections
            for detection in detections:
                tags = detection.tags
                if len(tags) > 0:
                    n_samples_with_tags += 1
        assert n_samples_with_tags > 0, "No samples with tags detected"
