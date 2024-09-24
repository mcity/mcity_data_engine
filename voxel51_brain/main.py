import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz

import os
import re
from datetime import datetime


def process_midadvrb_metadata(filename):

    # Extract name
    filename = os.path.basename(filename)
    # Define regex patterns for different formats
    results = {"filename": filename, "location": None, "name": None, "timestamp": None}

    available_locations = [
        "beal",
        "bishop",
        "georgetown",
        "gridsmart_ne",
        "gridsmart_nw",
        "gridsmart_se",
        "gridsmart_sw",
        "Huron_Plymouth-Geddes",
        "Main_stadium",
    ]

    for location in available_locations:
        if location in filename:
            results["location"] = location
            break

    # Split string into first and second part based on first 4 digit number
    match = re.search(r"\d{4}", filename)
    if match:
        year_index = match.start()
        part1 = filename[:year_index]
        part2 = filename[year_index:]

    # Cleanup first part
    results["name"] = re.sub(r"[-_]+$", "", part1)

    # Extract timestamp from second part
    match = re.search(r"\d{8}T\d{6}|\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", part2)
    if match:
        timestamp_str = match.group(0)
        if "T" in timestamp_str:
            results["timestamp"] = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
        else:
            results["timestamp"] = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
    return results


def load_dataset(dataset_name, dataset_dir, dataset_type, splits):
    """
    Loads a dataset into a FiftyOne Dataset object.

    Args:
        name (str): The name of the dataset.
        dataset_dir (str): The directory where the dataset is located.
        dataset_type (type): The type of the dataset (e.g., `fo.types.ImageDirectory`).
        splits (list): A list of dataset splits to load (e.g., ['train', 'test']).

    Returns:
        fo.Dataset: The loaded FiftyOne Dataset object.
    """

    try:
        fo.delete_dataset(dataset_name)
    except:
        print("No prior dataset active")

    dataset = fo.Dataset(dataset_name)
    for split in splits:
        dataset.add_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            split=split,
            tags=split,
        )

    return dataset


def brain_compute_embeddings(dataset):
    """
    Computes embeddings for a given dataset using UMAP, t-SNE, and PCA methods.

    Args:
        dataset: The dataset for which embeddings are to be computed.

    Returns:
        dict: A dictionary containing the computed embeddings for 'umap', 'tsne', and 'pca'.
    """
    embeddings = {"umap": None, "tsne": None, "pca": None}

    for embedding in embeddings:
        result = fob.compute_visualization(
            dataset,
            method=embedding,  #
            brain_key=embedding,
        )
        embeddings[embedding] = result
    return embeddings


def brain_compute_model_embeddings(dataset, seed=51):
    """
    Computes model embeddings for a given dataset using a specified seed.

    Args:
        dataset: The dataset for which embeddings are to be computed.
        seed (int, optional): The seed value for random operations. Defaults to 51.

    Returns:
        dict: A dictionary where keys are model names and values are the computed embeddings.
    """

    print(
        "Available models: ", foz.list_zoo_models()
    )  # https://docs.voxel51.com/user_guide/model_zoo/models.html

    embedding_models = (
        "mobilenet-v2-imagenet-torch",
        "dinov2-vitl14-torch",
        "inception-v3-imagenet-torch",
    )
    embeddings = dict.fromkeys(embedding_models)

    for model_name in embeddings:
        model = foz.load_zoo_model(model_name)
        if model.has_embeddings:
            model_embeddings = dataset.compute_embeddings(model)
            embedding_key = re.sub("[\W_]+", "", model_name)
            embeddings[model_name] = fob.compute_visualization(
                dataset, embeddings=model_embeddings, seed=seed, brain_key=embedding_key
            )
        else:
            print("Model ", model_name, " does not provide embeddings.")
    return embeddings


def main():
    dataset = load_dataset(
        dataset_name="midadvrb_2000",
        dataset_dir="/home/dbogdoll/datasets/midadvrb_2000",
        dataset_type=fo.types.YOLOv5Dataset,
        splits=["train", "val"],
    )
    dataset.compute_metadata()

    # Add dataset specific metedata based on filename
    view = dataset.view()
    for sample in view:  # https://docs.voxel51.com/api/fiftyone.core.sample.html
        metadata = process_midadvrb_metadata(sample["filepath"])
        sample["location"] = metadata["location"]
        sample["name"] = metadata["name"]
        sample["timestamp"] = metadata["timestamp"]
        sample.save()

    embeddings = brain_compute_embeddings(dataset)
    model_embeddings = brain_compute_model_embeddings(dataset)

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
