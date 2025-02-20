import logging
import os
import shutil

import fiftyone as fo
import pytest
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub

import config.config
from main import workflow_anomaly_detection
from utils.anomaly_detection_data_preparation import AnomalyDetectionDataPreparation
from utils.dataset_loader import _post_process_dataset
from utils.logging import configure_logging


@pytest.fixture(autouse=True)
def deactivate_hf_sync():
    """Hugging face deactivation."""
    config.config.HF_DO_UPLOAD = False


@pytest.fixture(autouse=True)
def setup_logging():
    """Logging setup."""
    configure_logging()


@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = "Voxel51/fisheye8k"
    dataset_name = "fisheye8k_v51_anomaly_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=30, name=dataset_name
        )
        dataset = _post_process_dataset(dataset)
        print(f"Loaded dataset {dataset_name} from hub: {dataset_name_hub}")
    except:
        dataset = fo.load_dataset(dataset_name)
        print(f"Dataset {dataset_name} was already loaded")
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"

    return dataset


def test_anomaly_detection_train(dataset_v51):
    """
    Test the anomaly detection training process.

    This function prepares the dataset for anomaly detection, configures the
    training parameters, and runs the anomaly detection workflow. It also
    verifies that the anomaly detection process identifies at least one
    anomalous sample.

    Args:
        dataset_v51: The dataset to be used for anomaly detection training.

    Raises:
        AssertionError: If no samples are selected through anomaly detection.
    """
    prep_config = {
        "location": "cam3",
        "rare_classes": ["Bus"],
    }

    data_preparer = AnomalyDetectionDataPreparation(
        dataset_v51, "fisheye8k_v51_anomaly_test", config=prep_config
    )
    run_config = {
        "model_name": "Padim",
        "image_size": [32, 32],
        "batch_size": 1,
        "epochs": 1,
        "early_stop_patience": 1,
        "data_root": data_preparer.export_root,
        "mode": ["train"],
    }

    eval_metrics = ["AUPR", "AUROC"]
    dataset_info = {"name": "fisheye8k_v51_anomaly_test"}

    # Delete content from field if other runs filled it already
    results_field = "pred_anomaly_score_Padim"
    try:
        data_preparer.dataset_ano_dec.delete_sample_field(results_field)
        print(f"Removed field {results_field} from dataset.")
    except:
        pass

    workflow_anomaly_detection(
        dataset_v51,
        data_preparer.dataset_ano_dec,
        dataset_info,
        eval_metrics,
        run_config,
        wandb_activate=False,
    )

    # Select all samples that are considered anomalous
    print(
        f"Sample fields in dataset: {data_preparer.dataset_ano_dec.get_field_schema()}"
    )
    view_anomalies = data_preparer.dataset_ano_dec.match(F(results_field) >= 0)
    n_samples_selected = len(view_anomalies)
    print(
        f"{n_samples_selected} samples anomalies found that were assessed by anomaly detection."
    )
    assert n_samples_selected != 0, "No samples were selected through anomaly detection"


@pytest.mark.parametrize("load_local", [True, False])
def test_anomaly_detection_inference(dataset_v51, load_local):
    """
    Test the anomaly detection inference process.

    This function tests the anomaly detection inference using the Padim model on the fisheye8k dataset.
    It optionally deletes local model weights to ensure they are downloaded from Hugging Face if `load_local` is False.
    The function configures the model, runs the anomaly detection workflow, and asserts that anomalies are detected.

    Args:
        dataset_v51: The dataset object to be used for testing.
        load_local (bool): Flag indicating whether to load local model weights or download them.

    Raises:
        AssertionError: If no samples are selected through anomaly detection.
    """

    if load_local == False:
        # Delete local weights if they exist so they get downloaded from Hugging Face
        local_folder = "./output/models/anomalib/Padim/fisheye8k"
        if os.path.exists(local_folder):
            try:
                shutil.rmtree(local_folder)
                logging.warning(f"Deleted local weights folder: {local_folder}")
            except Exception as e:
                logging.error(f"Error deleting local weights folder: {e}")

    dataset_ano_dec = None
    data_root = None

    run_config = {
        "model_name": "Padim",
        "image_size": [32, 32],
        "batch_size": 1,
        "epochs": 1,
        "early_stop_patience": 1,
        "data_root": data_root,
        "mode": ["inference"],
    }

    eval_metrics = ["AUPR", "AUROC"]
    dataset_info = {"name": "fisheye8k"}

    # Delete content from field if other runs filled it already
    results_field = "pred_anomaly_score_Padim"
    try:
        dataset_v51.delete_sample_field(results_field)
        logging.warning(f"Removed field {results_field} from dataset.")
    except:
        pass

    workflow_anomaly_detection(
        dataset_v51,
        dataset_ano_dec,
        dataset_info,
        eval_metrics,
        run_config,
        wandb_activate=False,
    )

    # Select all samples that are considered anomalous
    logging.info(f"Sample fields in dataset: {dataset_v51.get_field_schema()}")
    view_anomalies = dataset_v51.match(F(results_field) >= 0)
    n_samples_selected = len(view_anomalies)
    logging.info(
        f"{n_samples_selected} samples anomalies found that were assessed by anomaly detection."
    )

    assert n_samples_selected != 0, "No samples were selected through anomaly detection"
