import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F

import numpy as np

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

        # Generate folder to store all embedding-related results
        self.embeddings_root = embeddings_path + self.dataset_name + "/"
        Path(self.embeddings_root).mkdir(parents=True, exist_ok=True)

        self.brain_taxonomy = {
            "field": "brain_selection",
            "value_compute_representativeness": "representativeness_center",
            "value_find_unique": "greedy_center",
            "value_compute_uniqueness": "deterministic_center",
            "value_find_unique_neighbour": "greedy_neighbour",
            "value_compute_uniqueness_neighbour": "deterministic_neighbour",
            "value_compute_representativeness_neighbour": "representativeness_neighbour",
        }

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
                        self.dataset.compute_embeddings(
                            model=model, embeddings_field=embedding_key
                        )
                        self.embeddings_models[model_name] = self.dataset.values(
                            embedding_key
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
        # Indexes dataset by similarity
        # Calculates cosine distance for the embeddings without dimensionality reduction

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

    def compute_representativeness(self, threshold=0.99):
        # https://docs.voxel51.com/brain.html#image-representativeness

        field = self.brain_taxonomy["field"]
        value = self.brain_taxonomy["value_compute_representativeness"]
        methods_cluster_center = ["cluster-center", "cluster-center-downweight"]

        for embedding_name in tqdm(
            self.embeddings_models, desc="Computing representative frames"
        ):
            embedding = self.embeddings_models[embedding_name]
            for method in methods_cluster_center:
                key = re.sub(
                    r"[\W-]+",
                    "_",
                    embedding_name + "_" + method + "_representativeness",
                )

                if key in self.brains:
                    self.similarities[key] = self.dataset.load_brain_results(key)

                fob.compute_representativeness(
                    self.dataset,
                    representativeness_field=key,
                    method=method,
                    embeddings=embedding,
                )

                # quant_threshold = self.dataset.quantiles(key, threshold)
                # view = self.dataset.match(F(key) >= quant_threshold)
                view = self.dataset.match(F(key) >= threshold)
                for sample in view:
                    sample[field] = value
                    sample.save()

    def compute_unique_images_greedy(self, perct_unique=0.01):
        # https://docs.voxel51.com/user_guide/brain.html#cifar-10-example
        # find_unique(n) uses a greedy algorithm that, for a given n,
        # finds a subset of your dataset whose embeddings are as far as possible
        # from each other (= maximizes k=1 neighbor distance)

        sample_count = len(self.dataset.view())
        num_of_unique = perct_unique * sample_count
        field = self.brain_taxonomy["field"]
        value = self.brain_taxonomy["value_find_unique"]

        # Check if any sample has the label label_unique:
        dataset_labels = self.dataset.count_sample_tags()
        center_view = self.dataset.match(F(field) == value)

        if field in dataset_labels and len(center_view) > 0:
            pass

        else:
            for sim_key in tqdm(self.similarities, desc="Computing uniqueness greedy"):
                self.similarities[sim_key].find_unique(num_of_unique)
                for unique_id in self.similarities[sim_key].unique_ids:
                    sample = self.dataset[unique_id]
                    sample[field] = value
                    sample.save()

    def compute_unique_images_deterministic(self, threshold=0.99):
        # https://docs.voxel51.com/api/fiftyone.brain.html#fiftyone.brain.compute_uniqueness
        # compute_uniqueness() computes a deterministic uniqueness score
        # for each sample in [0, 1] (= weighted k-neighbors distances for each sample)

        field = self.brain_taxonomy["field"]
        value = self.brain_taxonomy["value_compute_uniqueness"]

        for embedding_name in tqdm(
            self.embeddings_models, desc="Computing uniqueness deterministic"
        ):
            embedding = self.embeddings_models[embedding_name]
            key = re.sub(r"[\W-]+", "_", embedding_name) + "_uniqueness"

            fob.compute_uniqueness(
                self.dataset, embeddings=embedding, uniqueness_field=key
            )

            # quant_threshold = self.dataset.quantiles(key, threshold)
            # view = self.dataset.match(F(key) >= quant_threshold)
            view = self.dataset.match(F(key) >= threshold)
            for sample in view:
                sample[field] = value
                sample.save()

    def find_samples_by_text(self, prompt, model_name):
        # https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.sort_by_similarity
        if model_name == "clip-vit-base32-torch":
            view = self.dataset.sort_by_similarity(prompt, k=5)

    def compute_duplicate_images(self, fraction=0.99):
        # Find duplicates of least similar images
        # https://docs.voxel51.com/brain.html#finding-near-duplicate-images
        pass

    def compute_similar_images(self, dist_threshold=0.03, neighbour_count=3):
        field = self.brain_taxonomy["field"]
        field_neighbour_distance = "distance"

        value_find_unique = self.brain_taxonomy["value_find_unique"]
        value_compute_uniqueness = self.brain_taxonomy["value_compute_uniqueness"]
        value_compute_representativeness = self.brain_taxonomy[
            "value_compute_representativeness"
        ]

        value_find_unique_neighbour = self.brain_taxonomy["value_find_unique_neighbour"]
        value_compute_uniqueness_neighbour = self.brain_taxonomy[
            "value_compute_uniqueness_neighbour"
        ]
        value_compute_representativeness_neighbour = self.brain_taxonomy[
            "value_compute_representativeness_neighbour"
        ]

        # Check if samples have already assigned fields
        dataset_labels = self.dataset.count_sample_tags()
        neighbour_view_greedy = self.dataset.match(
            F(field) == value_find_unique_neighbour
        )
        neighbour_view_deterministic = self.dataset.match(
            F(field) == value_compute_uniqueness_neighbour
        )
        neighbour_view_representativeness = self.dataset.match(
            F(field) == value_compute_representativeness_neighbour
        )

        if field in dataset_labels and (
            len(neighbour_view_greedy) > 0
            and len(neighbour_view_deterministic) > 0
            and len(neighbour_view_representativeness) > 0
        ):
            pass

        else:
            unique_view_greedy = self.dataset.match(F(field) == value_find_unique)
            unique_view_deterministic = self.dataset.match(
                F(field) == value_compute_uniqueness
            )
            unique_view_representativeness = self.dataset.match(
                F(field) == value_compute_representativeness
            )

            views_values = [
                (unique_view_greedy, value_find_unique_neighbour),
                (unique_view_deterministic, value_compute_uniqueness_neighbour),
                (
                    unique_view_representativeness,
                    value_compute_representativeness_neighbour,
                ),
            ]

            for sim_key in tqdm(self.similarities, desc="Finding similar images"):
                for unique_view, value in views_values:
                    for sample in unique_view:
                        view = self.dataset.sort_by_similarity(
                            sample.id,
                            k=neighbour_count,
                            brain_key=sim_key,
                            dist_field=field_neighbour_distance,
                        )
                        for sample_neighbour in view:
                            distance = sample_neighbour[field_neighbour_distance]
                            if (
                                distance < dist_threshold
                                and sample_neighbour[field] == None
                            ):
                                sample_neighbour[field] = value
                                sample_neighbour.save()
