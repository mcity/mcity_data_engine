import yaml

from data_loader.data_loader import *


def main():
    # Load the main configuration
    with open("config.yaml") as f:
        main_config = yaml.safe_load(f)

    # Load the selected dataset
    dataset_info = load_dataset_info(main_config["selected_dataset"])
    if dataset_info:
        loader_function = dataset_info.get("loader")
        dataset = globals()[loader_function](dataset_info)
    else:
        print("No valid dataset name provided in config.yaml")


if __name__ == "__main__":
    main()
