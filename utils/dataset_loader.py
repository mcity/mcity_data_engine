import yaml
import re
import os
from datetime import datetime

from nuscenes.nuscenes import NuScenes
import fiftyone as fo

import logging

from config.config import NUM_WORKERS, PERSISTENT


def load_dataset_info(dataset_name, config_path="./config/datasets.yaml"):
    """
    Load dataset information from a YAML configuration file.

    Args:
        dataset_name (str): The name of the dataset to retrieve information for.
        config_path (str, optional): The path to the YAML configuration file. Defaults to "datasets/datasets.yaml".

    Returns:
        dict or None: A dictionary containing the dataset information if found, otherwise None.
    """
    logging.info(f"Loaded V51 datasets: {fo.list_datasets()}")
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
    dataset_splits = dataset_info["v51_splits"]  # Use all available splits

    if PERSISTENT == False:
        try:
            fo.delete_dataset(dataset_info["name"])
        except:
            pass

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


def load_mcity_fisheye_3_months(dataset_info):
    """
    Loads the Mcity Fisheye 3 months dataset based on the provided dataset information.

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
    dataset_splits = dataset_info["v51_splits"]  # Use all available splits

    if PERSISTENT == False:
        try:
            fo.delete_dataset(dataset_info["name"])
        except:
            pass

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
        dataset.persistent = PERSISTENT  # https://docs.voxel51.com/user_guide/using_datasets.html#dataset-persistence
    return dataset


def load_fisheye_8k(dataset_info):
    """
    Loads a fisheye 8k dataset based on the provided dataset information.

    Args:
        dataset_info (dict): A dictionary containing the following keys:
            - "name" (str): The name of the dataset.
            - "local_path" (str): The local directory path where the dataset is stored.
            - "v51_type" (str): The type of the dataset as defined in `fo.types`.
            - "v51_splits" (list): A list of dataset splits to be used.

    Returns:
        fo.Dataset: The loaded FiftyOne dataset.

    Notes:
        - If the dataset is not persistent and already exists, it will be deleted.
        - If the dataset already exists, it will be loaded; otherwise, it will be created and metadata will be computed.
        - The dataset will be tagged with the split names and additional metadata will be added to each sample.
    """
    dataset_name = dataset_info["name"]
    dataset_dir = dataset_info["local_path"]
    dataset_type = getattr(fo.types, dataset_info["v51_type"])
    dataset_splits = dataset_info["v51_splits"]  # Use all available splits

    if PERSISTENT == False:
        try:
            fo.delete_dataset(dataset_info["name"])
        except:
            pass

    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        logging.info("Existing dataset " + dataset_name + " was loaded.")
    else:
        dataset = fo.Dataset(dataset_name)
        for split in dataset_splits:
            dataset.add_dir(
                data_path=os.path.join(dataset_dir, split, "images"),
                labels_path=os.path.join(dataset_dir, split, f"{split}.json"),
                dataset_type=dataset_type,
                tags=split,
            )

        dataset.compute_metadata(num_workers=NUM_WORKERS)

        view = dataset.view()
        for sample in view:  # https://docs.voxel51.com/api/fiftyone.core.sample.html
            metadata = process_fisheye_8k_filename(sample["filepath"])
            sample["location"] = metadata["location"]
            sample["time_of_day"] = metadata["time_of_day"]
            sample.save()

        dataset.persistent = PERSISTENT  # https://docs.voxel51.com/user_guide/using_datasets.html#dataset-persistence
    return dataset


def process_fisheye_8k_filename(filename):
    """
    Process a fisheye 8K filename to extract metadata.

    This function extracts the location and time of day from a given filename.
    The filename is expected to follow the format: cameraX_T_YYY.png, where:
    - X is the camera number.
    - T is the time of day indicator (A for afternoon, E for evening, N for night, M for morning).
    - YYY is an arbitrary sequence of digits.

    Args:
        filename (str): The filename to process.

    Returns:
        dict: A dictionary containing the following keys:
            - 'filename' (str): The basename of the input filename.
            - 'location' (str): The location derived from the camera number (e.g., 'camera4' becomes 'cam4').
            - 'time_of_day' (str or None): The time of day derived from the time of day indicator, or None if the indicator is not recognized.
    """

    time_of_day_map = {"A": "afternoon", "E": "evening", "N": "night", "M": "morning"}

    filename = os.path.basename(filename)
    parts = filename.split("_")

    location = parts[0].replace("camera", "cam")
    time_of_day = time_of_day_map.get(parts[1], None)

    results = {"filename": filename, "location": location, "time_of_day": time_of_day}

    return results


def load_mars_multiagent(dataset_info):
    hugging_face_id = "ai4ce/MARS/Multiagent_53scene"


def load_mars_multitraversal(dataset_info):
    location = 10
    data_root = "./datasets/MARS/Multitraversal_2023_10_04-2024_03_08"
    nusc = NuScenes(version="v1.0", dataroot=f"data_root/{location}", verbose=True)
