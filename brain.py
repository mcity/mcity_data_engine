import fiftyone as fo
from fiftyone import ViewField
import fiftyone.brain as fob
import fiftyone.zoo as foz

from tqdm import tqdm

import re
import os
from pathlib import Path
import pickle

import logging

from config import NUM_WORKERS


class Brain:

    def __init__(self, dataset, dataset_info, embeddings_path="./datasets/embeddings/"):
        self.dataset = dataset
        self.brains = dataset.list_brain_runs()
        self.dataset_name = dataset_info["name"]
        self.seed = 0

        # Storing variables
        self.embeddings_vis = {}
        self.embeddings_models = {}
        self.similarities = {}
        self.unique_ids = {}

        # Generate folder to store all embedding-related results
        self.embeddings_root = embeddings_path + self.dataset_name + "/"
        Path(self.embeddings_root).mkdir(parents=True, exist_ok=True)

    def compute_embeddings(self, embedding_model_names):
        # Use V51 pre-defined dim. reduction methods
        dim_reduction_methods = list(fob.brain_config.visualization_methods.keys())
        dim_reduction_methods.remove("manual")

        v51_model_zoo = foz.list_zoo_models()
        for model_name in tqdm(
            embedding_model_names, desc="Generating data embeddings"
        ):
            if model_name not in v51_model_zoo:
                logging.warning(
                    "Model " + v51_model_zoo + " is not part of the V51 model zoo."
                )
            else:
                model = foz.load_zoo_model(model_name)
                model_name_key = re.sub(r"[\W-]+", "_", model_name)
                embedding_key = "embedding_" + model_name_key
                embedding_file_name = self.embeddings_root + model_name_key + ".pkl"

                if model.has_embeddings:
                    # Load or compute embeddings for the model
                    if self.dataset.get_field(embedding_key) is not None:
                        self.embeddings_models[model_name] = self.dataset.values(
                            embedding_key
                        )
                    elif os.path.exists(embedding_file_name):
                        with open(embedding_file_name, "rb") as f:
                            self.embeddings_models[model_name] = pickle.load(f)
                        self.dataset.set_values(
                            embedding_key, self.embeddings_models[model_name]
                        )
                    else:
                        self.embeddings_models[model_name] = (
                            self.dataset.compute_embeddings(
                                model=model,
                                embeddings_field=embedding_key,
                            )
                        )
                        self.dataset.set_values(
                            embedding_key, self.embeddings_models[model_name]
                        )
                        with open(embedding_file_name, "wb") as f:
                            pickle.dump(self.embeddings_models[model_name], f)

                    for method in dim_reduction_methods:
                        key = model_name_key + "_" + re.sub(r"[\W-]+", "_", method)
                        points_key = "points_" + key
                        vis_file_name = self.embeddings_root + key + ".pkl"

                        if key in self.brains:
                            brain_info = self.dataset.get_brain_info(key)
                            self.embeddings_vis[key] = self.dataset.load_brain_results(
                                key
                            )

                        elif os.path.exists(vis_file_name):
                            with open(vis_file_name, "rb") as f:
                                points = pickle.load(f)

                            self.embeddings_vis[key] = fob.compute_visualization(
                                self.dataset,
                                method=method,
                                points=points,
                                embeddings=embedding_key,
                                seed=self.seed,
                                brain_key=key,
                                num_workers=NUM_WORKERS,
                            )
                            self.dataset.set_values(
                                points_key, self.embeddings_vis[key].current_points
                            )

                        else:
                            self.embeddings_vis[key] = fob.compute_visualization(
                                self.dataset,
                                method=method,
                                embeddings=embedding_key,
                                seed=self.seed,
                                brain_key=key,
                                num_workers=NUM_WORKERS,
                            )
                            self.dataset.set_values(
                                points_key, self.embeddings_vis[key].current_points
                            )

                            with open(vis_file_name, "wb") as f:
                                pickle.dump(self.embeddings_vis[key].current_points, f)
                else:
                    logging.warning(
                        "Model " + model_name + " does not provide embeddings."
                    )

    def compute_similarity(self):
        # https://docs.voxel51.com/user_guide/brain.html#similarity

        for embedding_name in tqdm(
            self.embeddings_models, desc="Computing similarities"
        ):
            embedding = self.embeddings_models[embedding_name]
            key = re.sub(r"[\W-]+", "_", embedding_name) + "_simil"

            if key in self.brains:
                self.similarities[key] = self.dataset.load_brain_results(key)

            else:
                self.similarities[key] = fob.compute_similarity(
                    self.dataset,
                    embeddings=embedding,
                    brain_key=key,
                    num_workers=NUM_WORKERS,
                )

    def compute_unique_images(self, num_of_unique=500, DEBUG_VIS=False):
        # https://docs.voxel51.com/user_guide/brain.html#cifar-10-example

        tag_unique = "unique"

        for sim_key in tqdm(self.similarities, desc="Computing unique images"):
            embeddings_vis_subset = {
                key: embedding
                for key, embedding in self.embeddings_vis.items()
                if sim_key[:-6] in key  # Remove _simil suffix
            }

            # Check if any sample has the label label_unique:
            dataset_labels = self.dataset.count_sample_tags()
            if tag_unique in dataset_labels:
                pass
                # self.unique_ids[sim_key] = self.similarities[sim_key].unique_ids

            else:
                self.similarities[sim_key].find_unique(num_of_unique)
                self.unique_ids[sim_key] = self.similarities[sim_key].unique_ids

                # Add V51 tag to unique labels
                for unique_id in self.unique_ids[sim_key]:
                    sample = self.dataset[unique_id]
                    if tag_unique not in sample.tags:
                        sample.tags.append(tag_unique)
                        sample.save()

            if DEBUG_VIS:
                for embedding_vis_key in embeddings_vis_subset:
                    embedding_vis = embeddings_vis_subset[embedding_vis_key]
                    plot = self.similarities[sim_key].visualize_unique(
                        visualization=embedding_vis
                    )
                    plot.show(height=800, yaxis_scaleanchor="x")
