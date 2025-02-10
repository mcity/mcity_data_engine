import fiftyone as fo
import pytest
from fiftyone import ViewField as F
from fiftyone.utils.huggingface import load_from_hub

from config.config import ACCEPTED_SPLITS
from main import workflow_anomaly_detection
from utils.anomaly_detection_data_preparation import AnomalyDetectionDataPreparation
from utils.dataset_loader import _post_process_dataset


@pytest.fixture
def dataset_v51():
    """Fixture to load a FiftyOne dataset from the hub."""
    dataset_name_hub = (
        "Voxel51/fisheye8k"  # TODO Upload anomaly dataset to Hugging Face
    )
    dataset_name = "fisheye8k_v51_anomaly_test"
    try:
        dataset = load_from_hub(
            repo_id=dataset_name_hub, max_samples=50, name=dataset_name
        )
    except:
        dataset = fo.load_dataset(dataset_name)
    assert dataset is not None, "Failed to load or create the FiftyOne dataset"

    dataset = _post_process_dataset(dataset)
    return dataset


def test_anomaly_detection(dataset_v51):

    prep_config = {"location": "cam3", "rare_classes": ["Bus"]}

    data_preparer = AnomalyDetectionDataPreparation(
        dataset_v51, "fisheye8k", config=prep_config
    )
    run_config = {
        "model_name": "Padim",
        "image_size": [265, 265],
        "batch_size": 1,
        "epochs": 1,
        "early_stop_patience": 1,
        "data_root": data_preparer.export_root,
    }

    eval_metrics = ["AUPR", "AUROC"]

    dataset_info = {"name": "fisheye8k"}

    workflow_anomaly_detection(
        data_preparer.dataset_ano_dec,
        dataset_info,
        eval_metrics,
        run_config,
    )

    # Select all samples that are considered anomalous
    view_anomalies = data_preparer.dataset_ano_dec.filter_labels(
        "pred_anomaly_Padim", F("label") == "anomaly"
    )
    n_samples_selected = len(view_anomalies)
    print(f"{n_samples_selected} samples with anomalies found")
    assert n_samples_selected != 0, "No samples were selected through anomaly detection"
