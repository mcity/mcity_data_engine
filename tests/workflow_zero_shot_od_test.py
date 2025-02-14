import logging

import fiftyone as fo
import pytest
from fiftyone.utils.huggingface import load_from_hub

import config.config
from main import configure_logging, workflow_zero_shot_object_detection


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
        dataset.persistent = True
    except:
        dataset = fo.load_dataset(dataset_name)
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
