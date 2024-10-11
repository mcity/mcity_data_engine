import time
import signal

from utils.logging import configure_logging
import logging

from config.config import (
    SELECTED_WORKFLOW,
    SELECTED_DATASET,
    V51_EMBEDDING_MODELS,
    ANOMALIB_IMAGE_MODELS,
)

from tqdm import tqdm

from utils.data_loader import *
from brain import Brain
from ano_dec import Anodec

import wandb
import sys


def panel_embeddings(v51_brain, color_field="unique"):
    """
    Creates a layout of panels for visualizing sample embeddings.

    Args:
        v51_brain: An object containing the embeddings visualization data.
        color_field (str, optional): The field used to color the embeddings. Defaults to "unique".

    Returns:
        fo.Space: A layout space containing the samples panel and embeddings panel.
    """
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


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    # Perform any cleanup or final actions here
    try:
        wandb.finish()
    except:
        pass
    sys.exit(0)


def main():
    # FIXME sudo chown -R dbogdoll:dbogdoll /home/dbogdoll/mcity_data_engine/output when docker changes owner to root

    time_start = time.time()
    signal.signal(signal.SIGINT, signal_handler)  # Signal handler for CTRL+C
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

    spaces = None
    logging.info("Running workflow " + SELECTED_WORKFLOW.upper())
    if SELECTED_WORKFLOW == "brain_selection":
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
        spaces = panel_embeddings(v51_brain)

    elif SELECTED_WORKFLOW == "learn_normality":
        for MODEL_NAME in (pbar := tqdm(ANOMALIB_IMAGE_MODELS, desc="Anomalib")):
            pbar.set_description("Training/Loading Anomalib model " + MODEL_NAME)

            ano_dec = Anodec(dataset, dataset_info, model_name=MODEL_NAME)
            ano_dec.train_and_export_model()
            # ano_dec.run_inference()
            # ano_dec.eval_v51()
            del ano_dec

    else:
        logging.error(
            str(SELECTED_WORKFLOW)
            + " is not a valid workflow. Check _WORKFLOWS_ in config.py."
        )
    #

    # Create layout for the web interface

    # Launch V51 session
    dataset.save()
    logging.info(dataset)
    fo.pprint(dataset.stats(include_media=True))
    wandb_run.finish()
    time_stop = time.time()
    logging.info(f"Elapsed time: {time_stop - time_start:.2f} seconds")
    if not os.getenv("RUNNING_IN_DOCKER"):  # ENV variable set in Dockerfile only
        session = fo.launch_app(dataset, spaces=spaces)
        session.wait(-1)


if __name__ == "__main__":
    main()
