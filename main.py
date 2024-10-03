import time

from colorlog import ColoredFormatter
import logging

from config import SELECTED_WORKFLOW, SELECTED_DATASET, V51_EMBEDDING_MODELS, WORKFLOWS

from data_loader.data_loader import *
from brain import Brain
from ano_dec import Anodec


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


def panel_embeddings(v51_brain, color_field="unique"):
    samples_panel = fo.Panel(type="Samples", pinned=True)

    embeddings_panel = fo.Panel(
        type="Embeddings",
        state=dict(
            brainResult=next(iter(v51_brain.embeddings_vis)), colorByField=color_field
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

    return spaces


# TODO Generate documentation https://docs.python.org/3/library/pydoc.html


def main():
    time_start = time.time()
    configure_logging()

    # Load the selected dataset
    dataset_info = load_dataset_info(SELECTED_DATASET)

    if dataset_info:
        loader_function = dataset_info.get("loader_fct")
        dataset = globals()[loader_function](dataset_info)
    else:
        logging.error(
            str(SELECTED_DATASET)
            + " is not a valid dataset name. Check _datasets_ in datasets.yaml."
        )

    if SELECTED_WORKFLOW == "brain_selection":
        logging.info("Running WORKFLO " + SELECTED_WORKFLOW)
        v51_brain = Brain(dataset, dataset_info)
        # Compute model embeddings and index for similarity
        v51_brain.compute_embeddings(V51_EMBEDDING_MODELS)
        v51_brain.compute_similarity()

        # Find representative and unique samples as center points for further selections
        v51_brain.compute_representativeness()
        v51_brain.compute_unique_images_greedy()
        v51_brain.compute_unique_images_deterministic()

        # Select samples similar to the center points to enlarge the dataset
        v51_brain.compute_similar_images()
        dataset.save()

    elif SELECTED_WORKFLOW == "learn_normality":
        logging.info("Running WORKFLO " + SELECTED_WORKFLOW)
        ano_dec = Anodec(dataset, dataset_info)

    else:
        logging.error(
            str(SELECTED_WORKFLOW)
            + " is not a valid workflow. Check _WORKFLOWS_ in main.py."
        )
    #

    # Create layout for the web interface

    # Launch V51 session
    logging.info(dataset)
    fo.pprint(dataset.stats(include_media=True))
    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")
    session = fo.launch_app(dataset, spaces=panel_embeddings(v51_brain))
    session.wait(-1)


if __name__ == "__main__":
    main()
