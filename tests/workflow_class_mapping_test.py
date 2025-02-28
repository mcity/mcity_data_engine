import fiftyone as fo
import pytest
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub
from main import workflow_class_mapping
from utils.dataset_loader import load_dataset_info, _post_process_dataset
import logging
from utils.logging import configure_logging

@pytest.fixture(autouse=True)
def setup_logging():
    configure_logging()

@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub with one target for test."""
    dataset_name_hub = "Abeyankar/class_mapping_test_dataset"

    dataset = load_from_hub(dataset_name_hub, overwrite=True)
    dataset = _post_process_dataset(dataset)

    return dataset

def test_class_mapping(dataset_v51):
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
        "dataset_source": "fisheye8k",
        "dataset_target": "mcity_fisheye_2000",
        "hf_models_zeroshot_classification": [
            "Salesforce/blip2-itm-vit-g",
            "openai/clip-vit-large-patch14",
            "google/siglip-so400m-patch14-384",
            "kakaobrain/align-base",
            "BAAI/AltCLIP",
            "CIDAS/clipseg-rd64-refined"
        ]
    }

    models = config["hf_models_zeroshot_classification"]

    workflow_class_mapping(dataset_v51, dataset_info, config)

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
