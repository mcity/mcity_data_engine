import random

import fiftyone as fo
import pytest
from fiftyone.utils.huggingface import load_from_hub

from main import workflow_brain
from utils.dataset_loader import load_dataset_info


@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "dbogdollumich/mcity_fisheye_v51"
    dataset_name = "mcity_fisheye_v51_brain_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=3, name=dataset_name
        )
    except:
        dataset = fo.load_dataset(dataset_name)
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"
    return dataset

def test_brain(dataset_v51):
    MODEL_NAME = "dinov2-vitl14-torch"
    dataset_info = load_dataset_info("mcity_fisheye_2000")
    v51_keys = workflow_brain(dataset_v51, dataset_info, MODEL_NAME)

    embedding_key = v51_keys["embedding"]
    similiarity_key = v51_keys["similarity"]
    uniqueness_key = v51_keys["uniqueness"]

    embeddings = dataset_v51.values(embedding_key)
    similarity = dataset_v51.values(similiarity_key)
    uniqueness = dataset_v51.values(uniqueness_key)

    print(embeddings)
    print(similarity)
    print(uniqueness)

    # Assert the results are not empty and have the correct length
    assert len(embeddings) == len(dataset_v51), "Embeddings length does not match dataset length"
    assert len(similarity) == len(dataset_v51), "Similarity length does not match dataset length"
    assert len(uniqueness) == len(dataset_v51), "Uniqueness length does not match dataset length"
