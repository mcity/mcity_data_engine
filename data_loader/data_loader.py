import yaml

from nuscenes.nuscenes import NuScenes
import fiftyone as fo


def load_dataset_info(dataset_name):
    """
    Load the information for a specific dataset from the datasets configuration file.

    Args:
        dataset_name (str): The name of the dataset to load information for.

    Returns:
        dict or None: A dictionary containing the dataset information if found,
                      otherwise None.
    """
    with open("datasets/datasets.yaml") as f:
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
    dataset_splits = dataset_info["v51_splits"]
    try:
        fo.delete_dataset(dataset_name)
    except:
        print("No prior dataset active")

    dataset = fo.Dataset(dataset_name)
    for split in dataset_splits:
        dataset.add_dir(
            dataset_dir="./datasets/" + dataset_dir,
            dataset_type=dataset_type,
            split=split,
            tags=split,
        )

    return dataset


def load_mars_multiagent(dataset_info):
    hugging_face_id = "ai4ce/MARS/Multiagent_53scene"


def load_mars_multitraversal(dataset_info):
    location = 10
    data_root = "./datasets/MARS/Multitraversal_2023_10_04-2024_03_08"
    nusc = NuScenes(version="v1.0", dataroot=f"data_root/{location}", verbose=True)
