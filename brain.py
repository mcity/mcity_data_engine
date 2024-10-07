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

from config.config import NUM_WORKERS, GLOBAL_SEED

"""
Implementing Voxel51 brain methods.
"""


class Brain:
    def __init__(self, dataset, dataset_info, embeddings_path="./output/embeddings/"):
        self.dataset = dataset
        self.brains = dataset.list_brain_runs()
        self.dataset_name = dataset_info["name"]

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
        """
        Computes and stores embeddings for the given list of embedding model names. Uses V51 pre-defined dim. reduction methods.

        This method performs the following steps:
        1. Retrieves the list of pre-defined dimensionality reduction methods.
        2. Iterates over the provided embedding model names.
        3. Checks if each model is part of the V51 model zoo.
        4. Loads or computes embeddings for each model.
        5. Saves the computed embeddings to disk.
        6. Computes and stores visualizations for the embeddings using various dimensionality reduction methods.

        Parameters:
        embedding_model_names (list): A list of model names for which embeddings need to be computed.

        Raises:
        Warning: If a model is not part of the V51 model zoo or does not provide embeddings.
        """

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
                                seed=GLOBAL_SEED,
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
                                seed=GLOBAL_SEED,
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
        """
        Computes the similarity of embeddings for the dataset.

        This method indexes the dataset by similarity and calculates the cosine
        distance for the embeddings without dimensionality reduction. It iterates
        over the available embedding models and computes or loads the similarity
        results for each model.

        The similarity results are stored in the `self.similarities` dictionary
        with keys derived from the embedding model names.

        References:
            - Voxel51 Brain Similarity Documentation:
              https://docs.voxel51.com/user_guide/brain.html#similarity

        Attributes:
            self.embeddings_models (dict): A dictionary of embedding models.
            self.brains (dict): A dictionary to store brain results.
            self.similarities (dict): A dictionary to store similarity results.
            self.dataset (Dataset): The dataset object to compute similarities on.

        """

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
        """
        Computes the representativeness of frames in the dataset using specified
        embedding models and methods.

        Args:
            threshold (float, optional): The threshold value for selecting
                representative frames. Defaults to 0.99.

        This method iterates over the embedding models and computes the
        representativeness using different methods. The results are stored in
        the dataset, and frames that meet the threshold criteria are updated
        with a specific field and value from the brain taxonomy.

        The method uses the following representativeness computation methods:
            - "cluster-center"
            - "cluster-center-downweight"

        The computed representativeness is stored in the dataset under keys
        generated from the embedding model names and methods. If the key
        already exists in the brains attribute, the existing similarities
        are loaded from the dataset.

        The method also updates the dataset samples that meet the threshold
        criteria by setting a specific field to a value defined in the brain
        taxonomy and saving the changes.

        References:
            - https://docs.voxel51.com/brain.html#image-representativeness
        """

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
        """
        Computes a subset of unique images from the dataset using a greedy algorithm.

        This method identifies a subset of images whose embeddings are as far apart as possible,
        maximizing the k=1 neighbor distance. The percentage of unique images to be found is
        specified by the `perct_unique` parameter.

        Args:
            perct_unique (float): The percentage of unique images to find in the dataset.
                      Default is 0.01 (1%).

        Notes:
            - The method checks if any sample in the dataset already has the label indicating
              uniqueness. If such samples exist, the method does nothing.
            - If no such samples exist, the method iterates over the similarities and computes
              unique images using the `find_unique` method of each similarity object.
            - The unique images are then tagged with a specific field and value and saved back
              to the dataset.

        References:
            - Voxel51 Brain documentation: https://docs.voxel51.com/user_guide/brain.html#cifar-10-example
        """

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
        """
        Computes a deterministic uniqueness score for each sample in the dataset
        and updates the dataset with the computed uniqueness values. Weighted k-neighbors distances for each sample.

        This method iterates over the embeddings models, computes the uniqueness
        score for each embedding, and updates the dataset with the computed scores.
        Samples with uniqueness scores greater than or equal to the specified
        threshold are marked with a specific field and value.

        Args:
            threshold (float, optional): The threshold value for determining
            unique samples. Samples with uniqueness scores greater than or
            equal to this threshold will be marked. Default is 0.99.

        References:
            - https://docs.voxel51.com/api/fiftyone.brain.html#fiftyone.brain.compute_uniqueness
        """

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
        """
        Computes and assigns similar images based on a distance threshold and neighbour count.

        This method checks if samples have already assigned fields and if not, it finds unique views
        and assigns neighbours based on similarity. It iterates through the similarities and updates
        the dataset with the computed values.

        Parameters:
        dist_threshold (float): The distance threshold to consider for similarity. Default is 0.03.
        neighbour_count (int): The number of neighbours to consider for each sample. Default is 3.

        Returns:
        None
        """
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
