import logging

import fiftyone as fo
import pytest
from fiftyone.utils.huggingface import load_from_hub

from main import configure_logging, workflow_zero_shot_object_detection


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
            repo_id=dataset_name_hub, max_samples=1, name=dataset_name
        )
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
            "google/owlv2-base-patch16-finetuned": {
                "batch_size": 1,
                "n_dataset_chunks": 2,  # Test 2
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

    workflow_zero_shot_object_detection(dataset_v51, dataset_info, config)

    logging.info(dataset_v51)
