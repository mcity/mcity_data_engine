import yaml
import time

from colorlog import ColoredFormatter
import logging

from data_loader.data_loader import *
from brain import compute_embeddings, compute_similarity, compute_unique_images

# Configure logging
formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        handler,  # Log to console with colors
        logging.FileHandler("dataengine.log"),  # Log to a file without colors
    ],
)


def main():
    time_start = time.time()
    # Load the main configuration
    with open("config.yaml") as f:
        main_config = yaml.safe_load(f)

    # Load the selected dataset
    logging.info(f"Available V51 datasets: {fo.list_datasets()}")
    dataset_info = load_dataset_info(main_config["selected_dataset"])
    if dataset_info:
        loader_function = dataset_info.get("loader_fct")
        dataset = globals()[loader_function](dataset_info)
    else:
        logging.error("No valid dataset name provided in config.yaml")

    # Compute model embeddings
    embedding_model_names = main_config.get("v51_embedding_models")
    embeddings_vis, embeddings_models = compute_embeddings(
        dataset, dataset_info, embedding_model_names
    )
    similarities = compute_similarity(dataset, embeddings_models)
    similarities = compute_unique_images(similarities, embeddings_vis)

    # Launch V51 session
    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
