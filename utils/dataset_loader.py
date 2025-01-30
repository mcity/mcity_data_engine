import datetime
import logging
import os
import re

import fiftyone as fo
import yaml
from fiftyone.utils.huggingface import load_from_hub
from nuscenes.nuscenes import NuScenes

from config.config import NUM_WORKERS, PERSISTENT


def _align_splits(dataset):
    # Get dataset tags related to splits
    SUPPORTED_SPLITS = ["train", "training", "val", "validation", "test", "testing"]
    tags = dataset.distinct("tags")
    splits = [tag for tag in tags if tag in SUPPORTED_SPLITS]

    # Rename splits if necessary
    rename_mapping = {
        "training": "train",
        "validation": "val",
        "testing": "test"
    }

    for old_tag, new_tag in rename_mapping.items():
        if old_tag in splits:
            dataset.rename_tag(old_tag, new_tag)
            splits = [tag if tag != old_tag else new_tag for tag in splits]

    # If only val and no test, rename val to test
    if "val" in splits and "test" not in splits:
        dataset.rename_tag("val", "test")
        splits = [tag if tag != "val" else "test" for tag in splits]

    # If train exists, val/test should also exist. If val/test exists, train should also exist
    if ("train" in splits and "test" not in splits and "val" not in splits) or \
       (("test" in splits or "val" in splits) and "train" not in splits):
        logging.warning(f"Inconsistent splits {splits}: 'train' should exist if 'val' or 'test' exists, and vice versa.")

    # Logging of available splits
    tags = dataset.distinct("tags")
    ACCEPTED_SPLITS = ["train", "val", "test"]
    splits = [tag for tag in tags if tag in ACCEPTED_SPLITS]
    logging.info(f"Available splits: {splits}")

    return splits

def _align_ground_truth(dataset, gt_field = "ground_truth"):

    dataset_fields = dataset.get_field_schema()
    if gt_field not in dataset_fields:
        FIFTYONE_DEFAULT_FIELDS = ["id", "filepath", "tags", "metadata", "created_at", "last_modified_at"]
        non_default_fields = {k: v for k, v in dataset_fields.items() if k not in FIFTYONE_DEFAULT_FIELDS}
        label_fields = {k: v for k, v in non_default_fields.items() if isinstance(v, fo.EmbeddedDocumentField) and issubclass(v.document_type, fo.core.labels.Label)}
        if len(label_fields) == 1:
            gt_label_old = next(iter(label_fields))
            dataset.rename_sample_field(gt_label_old, gt_field)
            logging.warning(f"Label field '{gt_label_old}' renamed to '{gt_field}' for training.")
        elif len(label_fields) > 1:
            logging.warning(f"The dataset has {len(label_fields)} fields with detections: {label_fields}. Rename one to {gt_field} with the command 'dataset.rename_sample_field(<field>, {gt_field})' to use it for training.")


def _post_process_dataset(dataset):
    # Set persistance
    # https://docs.voxel51.com/user_guide/using_datasets.html#dataset-persistence
    dataset.persistent = PERSISTENT

    # Compute metadata
    dataset.compute_metadata(num_workers=NUM_WORKERS)

    # Align split names
    splits = _align_splits(dataset)

    # Align ground truth field
    _align_ground_truth(dataset)

    return dataset

def load_dataset_info(dataset_name, config_path="./config/datasets.yaml"):
    """
    Load dataset information from a YAML configuration file.

    Args:
        dataset_name (str): The name of the dataset to retrieve information for.
        config_path (str, optional): The path to the YAML configuration file. Defaults to "datasets/datasets.yaml".

    Returns:
        dict or None: A dictionary containing the dataset information if found, otherwise None.
    """
    logging.info(f"Currently active V51 datasets: {fo.list_datasets()}")
    with open(config_path) as f:
        datasets_config = yaml.safe_load(f)

    datasets = datasets_config["datasets"]
    dataset_info = next((ds for ds in datasets if ds["name"] == dataset_name), None)

    if dataset_info:
        return dataset_info
    else:
        return None

def load_annarbor_rolling(dataset_info):
    dataset_name = dataset_info["name"]
    dataset_dir = dataset_info["local_path"]
    dataset_type = getattr(fo.types, dataset_info["v51_type"])

    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        logging.info("Existing dataset " + dataset_name + " was loaded.")
    else:
        dataset = fo.Dataset(dataset_name)
        dataset.add_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
        )

    return _post_process_dataset(dataset)

def load_mcity_fisheye_2000(dataset_info):
    """
    Loads the Mcity Fisheye 2000 dataset based on the provided dataset information.

    Args:
        dataset_info (dict): A dictionary containing the following keys:
            - "name" (str): The name of the dataset.
            - "local_path" (str): The local path to the dataset directory.
            - "v51_type" (str): The type of the dataset, corresponding to a type in `fo.types`.

    Returns:
        fo.Dataset: The loaded dataset object.

    Raises:
        KeyError: If any of the required keys are missing in `dataset_info`.
        AttributeError: If `v51_type` does not correspond to a valid type in `fo.types`.
    """
    dataset_name = dataset_info["name"]
    dataset_dir = dataset_info["local_path"]
    dataset_type = getattr(fo.types, dataset_info["v51_type"])
    dataset_splits = dataset_info["v51_splits"]

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

        # Add dataset specific metadata based on filename
        for sample in dataset.iter_samples(progress=True, autosave=True):
            metadata = _process_mcity_fisheye_filename(sample["filepath"])
            sample["location"] = metadata["location"]
            sample["name"] = metadata["name"]
            sample["timestamp"] = metadata["timestamp"]

    return _post_process_dataset(dataset)


def _process_mcity_fisheye_filename(filename):
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

    # TODO Check if some locations are duplicated (e.g. beal vs gs_Plymouth_Beal)
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
        "gs_Geddes_Huron",
        "gs_Huron_Plymouth",
        "gs_Plymouth_Beal",
        "gs_Plymouth_Georgetown",
        "gs_Plymouth_Bishop",
        "gs_Plymouth_EPA",
    ]

    for location in available_locations:
        if location in filename:
            results["location"] = location
            break

    if results["location"] is None:
        logging.error(f"Filename {filename} could not be assigned to a known location")

    # Split string into first and second part based on first 4 digit year number
    match = re.search(r"\d{4}", filename)
    if match:
        year_index = match.start()
        part1 = filename[:year_index]
        part2 = filename[year_index:]

    # Cleanup first part
    results["name"] = re.sub(r"[-_]+$", "", part1)

    # Extract timestamp from second part
    match = re.search(r"\d{8}T\d{6}|\d{4}-\d{2}-\d{2}[_ ]\d{2}-\d{2}-\d{2}", part2)
    if match:
        extracted_timestamp = match.group(0)

        if re.match(r"\d{8}T\d{6}", extracted_timestamp):
            results["timestamp"] = datetime.datetime.strptime(
                extracted_timestamp, "%Y%m%dT%H%M%S"
            )
        elif re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", extracted_timestamp):
            results["timestamp"] = datetime.datetime.strptime(
                extracted_timestamp, "%Y-%m-%d_%H-%M-%S"
            )
        elif re.match(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", extracted_timestamp):
            results["timestamp"] = datetime.datetime.strptime(
                extracted_timestamp, "%Y-%m-%d %H-%M-%S"
            )
        else:
            logging.error(f"Unknown timestamp format: {match}")
    else:
        logging.error(f"No valid timestamp found in string: {part2}")

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

        # Add dataset specific metedata based on filename
        for sample in dataset.iter_samples(progress=True, autosave=True):
            metadata = _process_mcity_fisheye_filename(sample["filepath"])
            sample["location"] = metadata["location"]
            sample["name"] = metadata["name"]
            sample["timestamp"] = metadata["timestamp"]

    return _post_process_dataset(dataset)


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
    hf_dataset_name = dataset_info["hf_dataset_name"]

    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        logging.info("Existing dataset " + dataset_name + " was loaded.")
    else:
        dataset = load_from_hub(hf_dataset_name)

    return _post_process_dataset(dataset)

def load_mars_multiagent(dataset_info):
    hugging_face_id = "ai4ce/MARS/Multiagent_53scene"

    dataset = None # TODO Implement loading

    return _post_process_dataset(dataset)


def load_mars_multitraversal(dataset_info):
    location = 10
    data_root = "./datasets/MARS/Multitraversal_2023_10_04-2024_03_08"
    nusc = NuScenes(version="v1.0", dataroot=f"data_root/{location}", verbose=True)

    dataset = None # TODO Implement loading

    return _post_process_dataset(dataset)
