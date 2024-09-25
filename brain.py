import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

import re
import os
from pathlib import Path
import json
import numpy as np
import pickle


def compute_embeddings(dataset, dataset_info, embedding_model_names, seed=0):
    """
    Computes embeddings for a given dataset using UMAP, t-SNE, and PCA methods.

    Args:
        dataset: The dataset for which embeddings are to be computed.

    Returns:
        dict: A dictionary containing the computed embeddings for 'umap', 'tsne', and 'pca'.
    """

    dataset_name = dataset_info["name"]
    embeddings_root = "./datasets/embeddings/" + dataset_name + "/"
    Path(embeddings_root).mkdir(parents=True, exist_ok=True)

    # Use V51 pre-defined dim. reduction methods
    dim_reduction_methods = list(fob.brain_config.visualization_methods.keys())
    dim_reduction_methods.remove("manual")
    embeddings = {}

    for model_name in embedding_model_names:
        model = foz.load_zoo_model(model_name)

        if model.has_embeddings:
            embedding_file_name = (
                embeddings_root + re.sub(r"[\W-]+", "_", model_name) + ".pkl"
            )
            if os.path.exists(embedding_file_name):
                with open(embedding_file_name, "rb") as f:
                    model_embeddings = pickle.load(f)
            else:
                model_embeddings = dataset.compute_embeddings(model=model)
                with open(embedding_file_name, "wb") as f:
                    pickle.dump(model_embeddings, f)

            for method in dim_reduction_methods:
                key = re.sub(r"[\W-]+", "_", model_name + "_" + method)
                vis_file_name = embeddings_root + key + ".pkl"

                if os.path.exists(vis_file_name):
                    with open(vis_file_name, "rb") as f:
                        points = pickle.load(f)
                    embeddings[key] = fob.compute_visualization(
                        dataset,
                        method=method,
                        embeddings=model_embeddings,
                        points=points,
                        seed=seed,
                        brain_key=key,
                    )
                else:
                    embeddings[key] = fob.compute_visualization(
                        dataset,
                        method=method,
                        embeddings=model_embeddings,
                        seed=seed,
                        brain_key=key,
                    )

                    # Save VisualizationResults object
                    with open(vis_file_name, "wb") as f:
                        pickle.dump(embeddings[key].points, f)
                    # embeddings[key].write_json(vis_file_name)
    else:
        model_embeddings = None

    return embeddings, model_embeddings
