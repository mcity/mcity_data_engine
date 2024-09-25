import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

import re
import os
from pathlib import Path
import json


def compute_embeddings(dataset, dataset_info, embedding_model_names, seed=0):
    """
    Computes embeddings for a given dataset using UMAP, t-SNE, and PCA methods.

    Args:
        dataset: The dataset for which embeddings are to be computed.

    Returns:
        dict: A dictionary containing the computed embeddings for 'umap', 'tsne', and 'pca'.
    """

    # Use V51 pre-defined dim. reduction methods
    embeddings = dict.fromkeys(fob.brain_config.visualization_methods.keys())
    embeddings.pop("manual", None)
    embeddings.pop("umap", None)  # TODO Remove
    embeddings.pop("tsne", None)  # TODO Remove

    dataset_name = dataset_info["name"]
    embeddings_root = "./datasets/embeddings/" + dataset_name + "/"
    Path(embeddings_root).mkdir(parents=True, exist_ok=True)

    for embedding_name in embeddings:
        file_path = embeddings_root + embedding_name + ".json"
        if os.path.exists(file_path):
            with open(file_path, "r") as json_file:
                loaded_embedding = json.load(json_file)

                # TODO: Implement loading
        else:
            embedding = fob.compute_visualization(
                dataset,
                method=embedding_name,
                brain_key=embedding_name,
            )
            # TODO embedding.write_json(file_path)
        embeddings[embedding_name] = embedding

    # TODO Implement saving + loading
    if embedding_model_names:
        model_embeddings = dict.fromkeys(embedding_model_names)
        for model_name in model_embeddings:
            model = foz.load_zoo_model(model_name)
            if model.has_embeddings:
                embedding = dataset.compute_embeddings(model)
                embedding_key = re.sub("[\W_]+", "", model_name)
                embeddings[model_name] = fob.compute_visualization(
                    dataset,
                    embeddings=embedding,
                    seed=seed,
                    brain_key=embedding_key,
                )
            else:
                print("Model ", model_name, " does not provide embeddings.")
    else:
        model_embeddings = None

    return embeddings, model_embeddings
