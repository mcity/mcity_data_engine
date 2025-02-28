import datetime
import logging
import os
import re
from typing import List, Union

import fiftyone as fo
import yaml
from fiftyone.utils.huggingface import load_from_hub
from nuscenes.nuscenes import NuScenes

from config.config import ACCEPTED_SPLITS, GLOBAL_SEED, NUM_WORKERS, PERSISTENT
from utils.sample_field_operations import rename_sample_field


def load_dataset(selected_dataset: str) -> fo.Dataset:
    """Loads a dataset by name, optionally reducing it to a requested number of samples while maintaining original split distributions."""
    dataset_info = load_dataset_info(selected_dataset["name"])

    if dataset_info:
        loader_function = dataset_info.get("loader_fct")
        dataset = globals()[loader_function](dataset_info)
        n_samples_original = len(dataset)
        n_samples_requested = selected_dataset["n_samples"]

        if (
            n_samples_requested is not None
            and n_samples_requested <= n_samples_original
        ):
            logging.info("Dataset reduction in process.")
            # Make sure that the reduced datasets has samples from every available split
            split_views = []

            # Get split distribution
            tags_count_dataset_dict = dataset.count_sample_tags()
            for tag in tags_count_dataset_dict:
                if tag in ACCEPTED_SPLITS:
                    count = tags_count_dataset_dict[tag]
                    percentage = count / n_samples_original
                    n_split_samples = int(n_samples_requested * percentage)
                    logging.info(f"Split {tag}: {n_split_samples} samples")

                    split_view = dataset.match_tags(tag).limit(n_split_samples)
                    split_views.append(split_view)

            # Concatenate views properly
            if split_views:
                combined_view = split_views[0]
                for view in split_views[1:]:
                    combined_view = combined_view.concat(view)

                # Fill dataset if smaller than requested
                if len(combined_view) < n_samples_requested:
                    n_samples_needed = n_samples_requested - len(combined_view)
                    view_random = dataset.take(n_samples_needed, seed=GLOBAL_SEED)
                    combined_view = combined_view.concat(view_random)

                logging.warning(
                    f"Dataset size was reduced from {len(dataset)} to {len(combined_view)} samples."
                )
                return combined_view, dataset_info

    else:
        logging.error(
            str(selected_dataset["name"])
            + " is not a valid dataset name. Check supported datasets in datasets.yaml."
        )

    return dataset, dataset_info


def get_split(v51_sample: Union[fo.core.sample.Sample, List[str]]) -> str:
    """Gets dataset split (train, val, test) from a sample's tags or list of tags."""
    if isinstance(v51_sample, fo.core.sample.Sample):
        sample_tags = v51_sample.tags
    elif isinstance(v51_sample, list):
        sample_tags = v51_sample
    else:
        logging.error(
            f"Type {isinstance(v51_sample)} is not supported for split retrieval."
        )

    found_splits = [split for split in ACCEPTED_SPLITS if split in sample_tags]

    if len(found_splits) == 0:
        logging.warning(f"No split found in sample tags: {sample_tags}")
        return None
    elif len(found_splits) > 1:
        logging.warning(f"Multiple splits found in sample tags: '{found_splits}'")
        return None
    else:
        split = found_splits[0]
        return split


def _separate_split(dataset, current_split, new_split, split_ratio=2):
    """Separates a portion of samples from the current split in the dataset and assigns them to a new split."""
    # Select samples for split change
    view_current_split = dataset.match_tags(current_split)
    n_samples_current_split = len(view_current_split)
    view_new_split = view_current_split.take(
        int(n_samples_current_split / split_ratio), seed=GLOBAL_SEED
    )
    view_new_split.tag_samples(new_split)
    view_new_split.untag_samples(current_split)

    # Get number of samples in each split
    view_current_split_changed = dataset.match_tags(current_split)
    n_samples_current_split_changed = len(view_current_split_changed)
    view_new_split = dataset.match_tags(new_split)
    n_samples_new_split = len(view_new_split)

    return n_samples_current_split, n_samples_current_split_changed, n_samples_new_split


def _align_splits(dataset):
    """Standardize dataset splits by renaming and creating missing splits (train/val/test) as needed."""
    SUPPORTED_SPLITS = ["train", "training", "val", "validation", "test", "testing"]
    tags = dataset.distinct("tags")
    splits = [tag for tag in tags if tag in SUPPORTED_SPLITS]

    # Rename splits if necessary
    rename_mapping = {"training": "train", "validation": "val", "testing": "test"}

    for old_tag, new_tag in rename_mapping.items():
        if old_tag in splits:
            dataset.rename_tag(old_tag, new_tag)
            splits = [tag if tag != old_tag else new_tag for tag in splits]

    # If only val or only test, create val and test splits
    if "val" in splits and "test" not in splits:
        (
            n_samples_current_split,
            n_samples_current_split_changed,
            n_samples_new_split,
        ) = _separate_split(dataset, current_split="val", new_split="test")
        logging.warning(
            f"Dataset had no 'test' split. Split {n_samples_current_split} 'val' into {n_samples_current_split_changed} 'val' and {n_samples_new_split} 'test'."
        )

    elif "test" in splits and "val" not in splits:
        (
            n_samples_current_split,
            n_samples_current_split_changed,
            n_samples_new_split,
        ) = _separate_split(dataset, current_split="test", new_split="val")
        logging.warning(
            f"Dataset had no 'val' split. Split {n_samples_current_split} 'test' into {n_samples_current_split_changed} 'val' and {n_samples_new_split} 'test'."
        )
    if "train" in splits and "test" not in splits and "val" not in splits:
        logging.warning(
            "Found 'train' split, but 'test' and 'val' splits are missing. Training might fail."
        )

    # Logging of available splits
    tags = dataset.distinct("tags")
    splits = [tag for tag in tags if tag in ACCEPTED_SPLITS]
    logging.info(f"Available splits: {splits}")

    return splits


def _align_ground_truth(dataset, gt_field="ground_truth"):
    """Ensures dataset has ground truth field named correctly, renaming single label field if found."""

    dataset_fields = dataset.get_field_schema()
    if gt_field not in dataset_fields:
        FIFTYONE_DEFAULT_FIELDS = [
            "id",
            "filepath",
            "tags",
            "metadata",
            "created_at",
            "last_modified_at",
        ]
        non_default_fields = {
            k: v for k, v in dataset_fields.items() if k not in FIFTYONE_DEFAULT_FIELDS
        }
        label_fields = {
            k: v
            for k, v in non_default_fields.items()
            if isinstance(v, fo.EmbeddedDocumentField)
            and issubclass(v.document_type, fo.core.labels.Label)
        }
        if len(label_fields) == 1:
            gt_label_old = next(iter(label_fields))
            rename_sample_field(dataset, gt_label_old, gt_field)
            logging.warning(
                f"Label field '{gt_label_old}' renamed to '{gt_field}' for training."
            )
        elif len(label_fields) > 1:
            logging.warning(
                f"The dataset has {len(label_fields)} fields with detections: {label_fields}. Rename one to {gt_field} with the command 'dataset.rename_sample_field(<your_field>, {gt_field})' to use it for training."
            )


def _post_process_dataset(dataset):
    """Post-processes the dataset by setting persistence, computing metadata, aligning splits, and aligning ground truth."""
    logging.info(f"Running dataset post-processing.")
    # Set persistance
    # https://docs.voxel51.com/user_guide/using_datasets.html#dataset-persistence
    dataset.persistent = PERSISTENT

    # Compute metadata
    dataset.compute_metadata(num_workers=NUM_WORKERS, overwrite=False, progress=True)

    # Align split names
    splits = _align_splits(dataset)

    # Align ground truth field
    _align_ground_truth(dataset)

    return dataset


def load_dataset_info(dataset_name, config_path="./config/datasets.yaml"):
    """Load dataset information from a YAML configuration file."""
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
    """Loads the Ann Arbor rolling dataset from local storage into FiftyOne, creating a new dataset if it doesn't exist."""
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
        _post_process_dataset(dataset)

    return dataset


def load_mcity_fisheye_2000(dataset_info):
    """Loads the MCityFisheye2000 dataset from local path or Hugging Face, creating or loading a FiftyOne dataset."""
    dataset_name = dataset_info["name"]
    dataset_dir = dataset_info["local_path"]
    hf_dataset_name = dataset_info.get("hf_dataset_name", None)
    dataset_type = getattr(fo.types, dataset_info["v51_type"])
    dataset_splits = dataset_info["v51_splits"]

    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        logging.info("Existing dataset " + dataset_name + " was loaded.")
    elif hf_dataset_name is not None:
        # Read API key for HF access
        hf_token = None
        try:
            with open(".secret", "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.split("=")[1].strip()
        except FileNotFoundError:
            logging.error(
                "'.secret' file not found. Please create it to load private datasets."
            )
            hf_token = None

        if hf_token is None:
            logging.error(
                "Provide your Hugging Face 'HF_TOKEN' in the .secret file to load private datasets."
            )
        dataset = load_from_hub(hf_dataset_name, name=dataset_name, token=hf_token)
        _post_process_dataset(dataset)
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

        _post_process_dataset(dataset)

    return dataset


def load_mcity_fisheye_2100_vru(dataset_info):
    """Loads the Mcity Fisheye 2100 VRU dataset from HuggingFace Hub or locally if it exists."""
    dataset_name = dataset_info["name"]
    hf_dataset_name = dataset_info["hf_dataset_name"]

    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        logging.info("Existing dataset " + dataset_name + " was loaded.")
    else:
        # Read API key for HF access
        hf_token = None
        try:
            with open(".secret", "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.split("=")[1].strip()
        except FileNotFoundError:
            logging.error(
                "'.secret' file not found. Please create it to load private datasets."
            )
            hf_token = None

        if hf_token is None:
            logging.error(
                "Provide your Hugging Face 'HF_TOKEN' in the .secret file to load private datasets."
            )
        dataset = load_from_hub(hf_dataset_name, name=dataset_name, token=hf_token)
        _post_process_dataset(dataset)

    return dataset


def _process_mcity_fisheye_filename(filename):
    """Processes a Mcity fisheye camera filename to extract location, name, and timestamp information."""

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
    """Loads or creates a FiftyOne dataset for the Mcity fisheye 3-month dataset using the provided dataset info."""

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

        _post_process_dataset(dataset)

    return dataset


def load_fisheye_8k(dataset_info):
    """Loads a fisheye 8k dataset from FiftyOne, creating it from HuggingFace if it doesn't exist locally."""

    dataset_name = dataset_info["name"]
    hf_dataset_name = dataset_info["hf_dataset_name"]

    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        logging.info("Existing dataset " + dataset_name + " was loaded.")
    else:
        dataset = load_from_hub(hf_dataset_name, name=dataset_name)
        _post_process_dataset(dataset)

    return dataset


def load_mars_multiagent(dataset_info):
    """Load the MARS multi-agent dataset from Hugging Face."""
    hugging_face_id = "ai4ce/MARS/Multiagent_53scene"

    dataset = None  # TODO Implement loading
    _post_process_dataset(dataset)

    return dataset


def load_mars_multitraversal(dataset_info):
    """Loads and post-processes multi-traversal MARS dataset from specified location."""
    location = 10
    data_root = "./datasets/MARS/Multitraversal_2023_10_04-2024_03_08"
    nusc = NuScenes(version="v1.0", dataroot=f"data_root/{location}", verbose=True)

    dataset = None  # TODO Implement loading
    _post_process_dataset(dataset)

    return dataset
