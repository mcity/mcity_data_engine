import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

import re
import os
from pathlib import Path
import pickle

import logging

from config import NUM_WORKERS


def compute_embeddings(dataset, dataset_info, embedding_model_names, seed=0):
    """
    Computes embeddings for a given dataset using UMAP, t-SNE, and PCA methods.
    https://docs.voxel51.com/user_guide/brain.html#brain-embeddings-visualization

    Args:
        dataset: The dataset for which embeddings are to be computed.

    Returns:
        dict: A dictionary containing the computed embeddings for 'umap', 'tsne', and 'pca'.
    """

    logging.info("Computing embeddings.")

    dataset_name = dataset_info["name"]
    embeddings_root = "./datasets/embeddings/" + dataset_name + "/"
    Path(embeddings_root).mkdir(parents=True, exist_ok=True)

    # Use V51 pre-defined dim. reduction methods
    dim_reduction_methods = list(fob.brain_config.visualization_methods.keys())
    dim_reduction_methods.remove("manual")
    embeddings_models = {}
    embeddings_vis = {}

    # TODO: Support loading models outside of the model zoo
    v51_model_zoo = foz.list_zoo_models()
    for model_name in embedding_model_names:
        model = foz.load_zoo_model(model_name)
        embedding_key = "embedding_" + re.sub(r"[\W-]+", "_", model_name)

        if model.has_embeddings:
            embedding_file_name = (
                embeddings_root + re.sub(r"[\W-]+", "_", model_name) + ".pkl"
            )
            if os.path.exists(embedding_file_name):
                with open(embedding_file_name, "rb") as f:
                    embeddings_models[model_name] = pickle.load(f)
            else:
                embeddings_models[model_name] = dataset.compute_embeddings(
                    model=model,
                    embeddings_field=embedding_key,
                )
                with open(embedding_file_name, "wb") as f:
                    pickle.dump(embeddings_models[model_name], f)

            # Store embedding also in V51 dataset field
            # https://docs.voxel51.com/tutorials/clustering.html
            dataset.set_values(embedding_key, embeddings_models[model_name])

            for method in dim_reduction_methods:
                key = re.sub(r"[\W-]+", "_", model_name + "_" + method)
                points_key = "points_" + key
                vis_file_name = embeddings_root + key + ".pkl"

                if os.path.exists(vis_file_name):
                    with open(vis_file_name, "rb") as f:
                        points = pickle.load(f)
                    embeddings_vis[key] = fob.compute_visualization(
                        dataset,
                        method=method,
                        points=points,
                        embeddings=embedding_key,
                        seed=seed,
                        brain_key=key,
                        num_workers=NUM_WORKERS,
                    )
                else:
                    embeddings_vis[key] = fob.compute_visualization(
                        dataset,
                        method=method,
                        embeddings=embedding_key,
                        seed=seed,
                        brain_key=key,
                        num_workers=NUM_WORKERS,
                    )

                    # Save VisualizationResults object
                    with open(vis_file_name, "wb") as f:
                        pickle.dump(embeddings_vis[key].current_points, f)  # points

                # Save points also to V51 field
                current_points = embeddings_vis[key].current_points
                dataset.set_values(points_key, current_points)
                dataset.save()

        else:
            logging.warning("Model " + model_name + " does not provide embeddings.")

    return embeddings_vis, embeddings_models


def compute_similarity(dataset, embeddings_models):
    # https://docs.voxel51.com/user_guide/brain.html#similarity

    logging.info("Computing similarities.")

    similarities = {}
    for embedding_name in embeddings_models:
        embedding = embeddings_models[embedding_name]
        key = re.sub(r"[\W-]+", "_", embedding_name) + "_simil"

        similarities[key] = fob.compute_similarity(
            dataset, embeddings=embedding, brain_key=key, num_workers=NUM_WORKERS
        )

        # TODO: Store and load

    return similarities


def compute_unique_images(
    dataset, similarities, embeddings_vis, num_of_unique=500, DEBUG_VIS=False
):
    # https://docs.voxel51.com/user_guide/brain.html#cifar-10-example

    logging.info("Computing unique images.")

    for sim_key in similarities:
        embeddings_vis_subset = {
            key: embedding
            for key, embedding in embeddings_vis.items()
            if sim_key[:-6] in key  # Remove _simil suffix
        }
        for embedding_vis in embeddings_vis_subset.values():
            similarities[sim_key].find_unique(num_of_unique)
            if DEBUG_VIS:
                plot = similarities[sim_key].visualize_unique(
                    visualization=embedding_vis
                )
                plot.show(height=800, yaxis_scaleanchor="x")

            # Add V51 tag to unique labels
            for unique_id in similarities[sim_key].unique_ids:
                sample = dataset[unique_id]
                if "unique" not in sample.tags:
                    sample.tags.append("unique")
                    sample.save()

            # TODO: Store and load
