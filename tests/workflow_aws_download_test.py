import pytest

import config.config
from main import workflow_aws_download
from utils.logging import configure_logging


@pytest.fixture(autouse=True)
def deactivate_hf_sync():
    config.config.HF_DO_UPLOAD = False


@pytest.fixture(autouse=True)
def setup_logging():
    configure_logging()


@pytest.mark.parametrize("selected_dataset_overwrite", [True, False])
def test_aws_download(selected_dataset_overwrite):

    config = {
        "bucket": "mcity-data-engine",
        "prefix": "",
        "download_path": "output/datasets/annarbor_rolling",
        "test_run": True,
        "selected_dataset_overwrite": selected_dataset_overwrite,
    }

    dataset, dataset_name, files_to_be_downloaded = workflow_aws_download(
        config, wandb_activate=False
    )

    if len(dataset) == 0 and config["test_run"] == True:
        assert files_to_be_downloaded != 0, "No files found in AWS bucket"
    else:
        assert len(dataset) != 0, "Dataset has no samples"
