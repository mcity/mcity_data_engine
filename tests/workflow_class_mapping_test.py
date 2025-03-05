import fiftyone as fo
import pytest
import config.config
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub
from main import workflow_class_mapping
from utils.dataset_loader import load_dataset_info, _post_process_dataset
import logging
from utils.logging import configure_logging

@pytest.fixture(autouse=True)
def setup_logging():
    configure_logging()

@pytest.fixture(autouse=True)
def deactivate_wandb_sync():
    config.config.WANDB_ACTIVE = False

@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub with one target for test."""
    dataset_name_hub = "Abeyankar/class_mapping_test_dataset"

    dataset = load_from_hub(dataset_name_hub, overwrite=True)
    dataset = _post_process_dataset(dataset)

    return dataset

@pytest.fixture
def dataset_v51_2():
    """Fixture to load a FiftyOne dataset from the hub with one target for test."""
    dataset_name_hub2 = "Abeyankar/class_mapping_target_test_dataset3"

    dataset2 = load_from_hub(dataset_name_hub2, overwrite=True)
    dataset2 = _post_process_dataset(dataset2)

    return dataset2

@pytest.fixture
def dataset_v51_3():
    """Fixture to load a FiftyOne dataset from the hub with one target for test."""
    dataset_name_hub3 = "Abeyankar/class_mapping_source_test_dataset"

    dataset3 = load_from_hub(dataset_name_hub3, overwrite=True)
    dataset3 = _post_process_dataset(dataset3)

    return dataset3

def test_class_mapping(dataset_v51,dataset_v51_2, dataset_v51_3):
    """
    Test workflow on the sample with the specified filepath,
    verifying that each model has added its specific tag.
    """

    # Get the first sample from the dataset.
    sample = dataset_v51.first()
    assert sample is not None, "Target sample not found in dataset"

    logging.info("\nBefore workflow:")
    if hasattr(sample, 'ground_truth') and sample["ground_truth"].detections:
        logging.info(f"Total number of detections in Sample: {len(sample['ground_truth'].detections)}")

    dataset_info = load_dataset_info("fisheye8k")  # Use loader for actual dataset
    dataset_info["name"] = "fisheye8k_v51_cm_test"  # Update with test name for local tests where both exist

    # Define a simplified local version of WORKFLOWS
    config = {
        # get the source and target dataset names from datasets.yaml
        "dataset_source": "cm_test_source",
        "dataset_target": "cm_test_target",
        "hf_models_zeroshot_classification": [
            "Salesforce/blip2-itm-vit-g",
        ],
        "thresholds": {
            "confidence": 0.2
        },
        "candidate_labels": {
            #Target class(Generalized class) : Source classes(specific categories)
            "Car": ["car", "van", "pickup"],
            "Truck": ["truck", "pickup"],
            #One_to_one_mapping
            "Bike" : ["motorbike/cycler"]
            #Can add other class mappings in here
        }
    }

    models = config["hf_models_zeroshot_classification"]

    workflow_class_mapping(dataset_v51, dataset_info, config, wandb_activate=False, test_dataset_source=dataset_v51_3, test_dataset_target=dataset_v51_2)

    logging.info("\nAfter workflow:")
    if hasattr(sample, "ground_truth") and sample["ground_truth"].detections:
        logging.info(f"Total number of detections in Sample: {len(sample.ground_truth.detections)}")

    # Gather all tags from all detections in the updated sample
    found_tags = set()
    if hasattr(sample, "ground_truth") and sample["ground_truth"].detections:
        for detection in sample["ground_truth"].detections:
            if hasattr(detection, 'tags') and detection.tags:
                found_tags.update(detection.tags)

    logging.info("\nAll found tags: %s", found_tags)

    # Identify which models did not add tags
    missing_models = []
    for model in models:
        if not any(f"new_class_{model}" in tag for tag in found_tags):
            missing_models.append(model)

    if missing_models:
        logging.warning("\nMissing tags for models: %s", missing_models)

    # Validate that each expected model name appears in at least one tag.
    for model in models:
        assert any(f"new_class_{model}" in tag for tag in found_tags), (
            f"Tag for model {model} not found in detections"
        )
