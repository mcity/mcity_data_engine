import yaml
import logging

from data_loader.data_loader import *
from brain import compute_embeddings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

    # Compute model embeddings
    embedding_model_names = main_config.get("v51_embedding_models")
    embeddings, model_embeddings = compute_embeddings(dataset, dataset_info, None)

    # Launch V51 session
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
