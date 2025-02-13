import os

import fiftyone as fo
import pytest
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub

from main import workflow_anomaly_detection
from utils.anomaly_detection_data_preparation import AnomalyDetectionDataPreparation
from utils.dataset_loader import _post_process_dataset


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
    prep_config = {
        "location": "cam3",
        "rare_classes": ["Bus"],
    }

    data_preparer = AnomalyDetectionDataPreparation(
        dataset_v51, "fisheye8k", config=prep_config
    )
    run_config = {
        "model_name": "Padim",
        "image_size": [32, 32],
        "batch_size": 1,
        "epochs": 1,
        "early_stop_patience": 1,
        "data_root": data_preparer.export_root,
        "mode": "train",
    }

    eval_metrics = ["AUPR", "AUROC"]
    dataset_info = {"name": "fisheye8k"}

    # Delete content from field if other runs filled it already
    results_field = "pred_anomaly_Padim"
    try:
        data_preparer.dataset_ano_dec.delete_sample_field(results_field)
        print(f"Removed field {results_field} from dataset.")
    except:
        pass

    workflow_anomaly_detection(
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
    view_anomalies = data_preparer.dataset_ano_dec.filter_labels(
        results_field, F("label") == "anomaly"
    )
    n_samples_selected = len(view_anomalies)
    print(
        f"{n_samples_selected} samples anomalies found that were assessed by anomaly detection."
    )
    assert n_samples_selected != 0, "No samples were selected through anomaly detection"


def test_anomaly_detection_inference(dataset_v51):
    prep_config = {
        "location": "cam3",
        "rare_classes": ["Bus"],
    }

    data_preparer = AnomalyDetectionDataPreparation(
        dataset_v51, "fisheye8k", config=prep_config
    )
    run_config = {
        "model_name": "Padim",
        "image_size": [32, 32],
        "batch_size": 1,
        "epochs": 1,
        "early_stop_patience": 1,
        "data_root": data_preparer.export_root,
        "mode": "inference",
    }

    eval_metrics = ["AUPR", "AUROC"]
    dataset_info = {"name": "fisheye8k"}

    # Delete content from field if other runs filled it already
    results_field = "pred_anomaly_Padim"
    try:
        data_preparer.dataset_ano_dec.delete_sample_field(results_field)
        print(f"Removed field {results_field} from dataset.")
    except:
        pass

    workflow_anomaly_detection(
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
    view_anomalies = data_preparer.dataset_ano_dec.filter_labels(
        results_field, F("label") == "anomaly"
    )
    n_samples_selected = len(view_anomalies)
    print(
        f"{n_samples_selected} samples anomalies found that were assessed by anomaly detection."
    )
    assert n_samples_selected != 0, "No samples were selected through anomaly detection"
