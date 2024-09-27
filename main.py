import time

from colorlog import ColoredFormatter
import logging

from config import SELECTED_DATASET, V51_EMBEDDING_MODELS, COMPUTE_BRAIN

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

    # Load the selected dataset
    dataset_info = load_dataset_info(SELECTED_DATASET)

    if dataset_info:
        loader_function = dataset_info.get("loader_fct")
        dataset = globals()[loader_function](dataset_info)
    else:
        logging.error("No valid dataset name provided in config.yaml")

    logging.warning(dataset.get_field("embedding_clip_vit_base32_torch"))
    logging.warning(dataset.get_field("points_clip_vit_base32_torch_umap"))

    if COMPUTE_BRAIN:

        # Compute model embeddings
        embeddings_vis, embeddings_models = compute_embeddings(
            dataset, dataset_info, V51_EMBEDDING_MODELS
        )
        similarities = compute_similarity(dataset, embeddings_models)
        compute_unique_images(dataset, similarities, embeddings_vis)

        dataset.save()

        # Layout
        samples_panel = fo.Panel(type="Samples", pinned=True)

        embeddings_panel = fo.Panel(
            type="Embeddings",
            state=dict(brainResult=next(iter(embeddings_vis)), colorByField="tags"),
        )

        spaces = fo.Space(
            children=[
                fo.Space(
                    children=[
                        fo.Space(children=[samples_panel]),
                    ],
                    orientation="horizontal",
                ),
                fo.Space(children=[embeddings_panel]),
            ],
            orientation="horizontal",
        )

    # Launch V51 session
    # fo.pprint(dataset.stats(include_media=True))
    # logging.info(dataset)
    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")
    session = fo.launch_app(dataset)  # spaces=spaces
    session.wait()


if __name__ == "__main__":
    main()
