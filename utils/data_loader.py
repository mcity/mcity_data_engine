import yaml
import re
import os
from datetime import datetime

from nuscenes.nuscenes import NuScenes
import fiftyone as fo

import logging

from config.config import NUM_WORKERS, PERSISTENT, SELECTED_SPLITS


def load_dataset_info(dataset_name, config_path="config/datasets.yaml"):
    """
    Load dataset information from a YAML configuration file.

    Args:
        dataset_name (str): The name of the dataset to retrieve information for.
        config_path (str, optional): The path to the YAML configuration file. Defaults to "datasets/datasets.yaml".

    Returns:
        dict or None: A dictionary containing the dataset information if found, otherwise None.
    """
    with open(config_path) as f:
        datasets_config = yaml.safe_load(f)

    datasets = datasets_config["datasets"]
    dataset_info = next((ds for ds in datasets if ds["name"] == dataset_name), None)

    if dataset_info:
        return dataset_info
    else:
        return None


def load_mcity_fisheye_2000(dataset_info):
    """
    Loads the Mcity Fisheye 2000 dataset based on the provided dataset information.

    Args:
        dataset_info (dict): A dictionary containing the following keys:
            - "name" (str): The name of the dataset.
            - "local_path" (str): The local path to the dataset directory.
            - "v51_type" (str): The type of the dataset, corresponding to a type in `fo.types`.
            - "v51_splits" (list): A list of dataset splits to be loaded.

    Returns:
        fo.Dataset: The loaded dataset object.

    Raises:
        KeyError: If any of the required keys are missing in `dataset_info`.
        AttributeError: If `v51_type` does not correspond to a valid type in `fo.types`.
    """
    dataset_name = dataset_info["name"]
    dataset_dir = dataset_info["local_path"]
    dataset_type = getattr(fo.types, dataset_info["v51_type"])
    if SELECTED_SPLITS:
        dataset_splits = SELECTED_SPLITS
    else:
        dataset_splits = dataset_info["v51_splits"]  # Use all available splits

    if PERSISTENT == False:
        try:
            fo.delete_dataset(dataset_info["name"])
        except:
            pass

    logging.info(f"Available V51 datasets: {fo.list_datasets()}")

    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        logging.info("Existing dataset " + dataset_name + " was loaded.")
    else:
        dataset = fo.Dataset(dataset_name)
        for split in dataset_splits:
            dataset.add_dir(
                dataset_dir=dataset_dir,
                dataset_type=dataset_type,
                split=split,
                tags=split,
            )
        dataset.compute_metadata(num_workers=NUM_WORKERS)

        # Add dataset specific metedata based on filename
        view = dataset.view()
        for sample in view:  # https://docs.voxel51.com/api/fiftyone.core.sample.html
            metadata = process_mcity_fisheye_2000_filename(sample["filepath"])
            sample["location"] = metadata["location"]
            sample["name"] = metadata["name"]
            sample["timestamp"] = metadata["timestamp"]
            sample.save()

        dataset.persistent = PERSISTENT  # https://docs.voxel51.com/user_guide/using_datasets.html#dataset-persistence
    return dataset


def process_mcity_fisheye_2000_filename(filename):
    """
    Processes a given filename to extract metadata including location, name, and timestamp.

    Args:
        filename (str): The full path or name of the file to be processed.

    Returns:
        dict: A dictionary containing the following keys:
            - 'filename' (str): The base name of the file.
            - 'location' (str or None): The location extracted from the filename, if available.
            - 'name' (str or None): The cleaned name extracted from the filename.
            - 'timestamp' (datetime or None): The timestamp extracted from the filename, if available.

    The function performs the following steps:
        1. Extracts the base name of the file.
        2. Searches for a known location within the filename.
        3. Splits the filename into two parts based on the first occurrence of a 4-digit year.
        4. Cleans up the first part to derive the name.
        5. Extracts and parses the timestamp from the second part of the filename.
    """

    filename = os.path.basename(filename)
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

    # Split string into first and second part based on first 4 digit year number
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


def load_mars_multiagent(dataset_info):
    hugging_face_id = "ai4ce/MARS/Multiagent_53scene"


def load_mars_multitraversal(dataset_info):
    location = 10
    data_root = "./datasets/MARS/Multitraversal_2023_10_04-2024_03_08"
    nusc = NuScenes(version="v1.0", dataroot=f"data_root/{location}", verbose=True)
