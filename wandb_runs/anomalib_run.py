from ano_dec import Anodec

import wandb
import logging

from utils.data_loader import *


def main(config):
    # Get dataset with Voxel51

    logging.warning(wandb.run)

    dataset_name = config["v51_dataset_name"]
    dataset_info = load_dataset_info(dataset_name)
    if dataset_info:
        loader_function = dataset_info.get("loader_fct")
        dataset = globals()[loader_function](dataset_info)
    else:
        logging.error(
            str(dataset_name)
            + " is not a valid dataset name. Check _datasets_ in datasets.yaml."
        )

    # Train with Anomalib
    ano_dec = Anodec(dataset, dataset_info, config)
    ano_dec.train_and_export_model()


if __name__ == "__main__":
    main()
