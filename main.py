import time

from colorlog import ColoredFormatter
import logging

from config import SELECTED_DATASET, V51_EMBEDDING_MODELS

from data_loader.data_loader import *
from brain import Brain


def configure_logging():

    # Configure logging
    if not os.path.exists("logs"):
        os.makedirs("logs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/{timestamp}_dataengine.log"

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
            logging.FileHandler(log_filename),
        ],
    )


def main():
    time_start = time.time()
    configure_logging()

    # Load the selected dataset
    dataset_info = load_dataset_info(SELECTED_DATASET)

    if dataset_info:
        loader_function = dataset_info.get("loader_fct")
        dataset = globals()[loader_function](dataset_info)
    else:
        logging.error("No valid dataset name provided in config file")

    v51_brain = Brain(dataset, dataset_info)

    # Compute model embeddings
    v51_brain.compute_embeddings(V51_EMBEDDING_MODELS)
    v51_brain.compute_similarity()
    v51_brain.compute_unique_images()

    dataset.save()

    # Layout
    samples_panel = fo.Panel(type="Samples", pinned=True)

    embeddings_panel = fo.Panel(
        type="Embeddings",
        state=dict(
            brainResult=next(iter(v51_brain.embeddings_vis)), colorByField="tags"
        ),
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
    logging.info(dataset)
    fo.pprint(dataset.stats(include_media=True))
    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")
    session = fo.launch_app(dataset, spaces=spaces)
    session.wait()


if __name__ == "__main__":
    main()
