import yaml

from nuscenes.nuscenes import NuScenes


def load_dataset_info(dataset_name):
    # Load the datasets configuration
    with open("data_loader/datasets.yaml") as f:
        datasets_config = yaml.safe_load(f)

    # Find the selected dataset
    datasets = datasets_config["datasets"]

    dataset_info = next((ds for ds in datasets if ds["name"] == dataset_name), None)

    if dataset_info:
        return dataset_info
    else:
        return None


def load_mars_multiagent():
    hugging_face_id = "ai4ce/MARS/Multiagent_53scene"


def load_mars_multitraversal(dataset_info):
    location = 10
    data_root = "./datasets/MARS/Multitraversal_2023_10_04-2024_03_08"
    nusc = NuScenes(version="v1.0", dataroot=f"data_root/{location}", verbose=True)

    # nuscenes dev-kit is for python3.7, we are on 3.12. It asks for shapely<2.0
