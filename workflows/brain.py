import logging
import os
import pickle
import re
import time
from pathlib import Path

import fiftyone.brain as fob
import fiftyone.zoo as foz
import numpy as np
from fiftyone import ViewField as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config import GLOBAL_SEED, NUM_WORKERS

"""
Implementing Voxel51 brain methods.
https://docs.voxel51.com/brain.html
"""


class Brain:
    def __init__(
        self, dataset, dataset_info, model_name, embeddings_path="./output/embeddings/"
    ):
        # WandB counter
        self.steps = 0
        self.dataset = dataset
        self.brains = dataset.list_brain_runs()
        self.dataset_name = dataset_info["name"]
        self.v51_model_zoo = foz.list_zoo_models()
        self.writer = SummaryWriter(log_dir="logs/tensorboard/brain")

        # Model
        if model_name not in self.v51_model_zoo:
            logging.warning(
                "Model " + model_name + " is not part of the V51 model zoo."
            )
        self.model = foz.load_zoo_model(model_name)

        # Keys
        self.model_name = model_name
        self.model_name_key = re.sub(r"[\W-]+", "_", model_name)
        self.embedding_key = "embedding_" + self.model_name_key
        self.similiarity_key = "simil_" + self.model_name_key
        self.uniqueness_key = "uniqueness_" + self.model_name_key

        # Storing variables
        self.embeddings_vis = {}  # Multiple methods per model
        self.representativeness = {}  # Multiple methods per model
        self.embeddings_model = None
        self.similarities = None

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

    def __del__(self):
        self.steps -= 1  # +1 after every function, need to decrement for final step
        self.writer.close()

    def compute_embeddings(self):
        """
        Computes and stores embeddings for the given model name. Uses V51 pre-defined dim. reduction methods.

        This method performs the following steps:
        1. Retrieves the list of pre-defined dimensionality reduction methods.
        4. Loads or computes embeddings for the model.
        5. Saves the computed embeddings to disk.
        6. Computes and stores visualizations for the embeddings using various dimensionality reduction methods.

        Parameters:
        model_name: Model name for which embeddings need to be computed.

        Raises:
        Warning: If a model is not part of the V51 model zoo or does not provide embeddings.
        """
        start_time = time.time()

        dim_reduction_methods = list(fob.brain_config.visualization_methods.keys())
        dim_reduction_methods.remove("manual")

        embedding_file_name = self.embeddings_root + self.model_name_key + ".pkl"

        if self.model.has_embeddings:
            # Load or compute embeddings for the model
            if self.dataset.get_field(self.embedding_key) is not None:
                logging.info("Loading embeddings from V51.")
                self.embeddings_model = self.dataset.values(self.embedding_key)
            elif os.path.exists(embedding_file_name):
                logging.info("Loading embeddings from disk.")
                with open(embedding_file_name, "rb") as f:
                    self.embeddings_model = pickle.load(f)
                self.dataset.set_values(self.embedding_key, self.embeddings_model)
            else:
                logging.info("Computing embeddings.")
                self.dataset.compute_embeddings(
                    model=self.model, embeddings_field=self.embedding_key
                )
                self.embeddings_model = self.dataset.values(self.embedding_key)

                # TODO Is this necessary? (since compute_embeddings is not stored into a variable)
                self.dataset.set_values(self.embedding_key, self.embeddings_model)
                with open(embedding_file_name, "wb") as f:
                    pickle.dump(self.embeddings_model, f)

            for method in tqdm(dim_reduction_methods, "Dimensionality reductions"):
                method_key = self.model_name_key + "_" + re.sub(r"[\W-]+", "_", method)
                points_key = "points_" + method_key
                vis_file_name = self.embeddings_root + method_key + ".pkl"

                if method_key in self.brains:
                    logging.info("Loading vis from V51.")
                    brain_info = self.dataset.get_brain_info(method_key)
                    self.embeddings_vis[method_key] = self.dataset.load_brain_results(
                        method_key
                    )

                elif os.path.exists(vis_file_name):
                    logging.info("Loading vis from disk.")
                    with open(vis_file_name, "rb") as f:
                        points = pickle.load(f)

                    self.embeddings_vis[method_key] = fob.compute_visualization(
                        self.dataset,
                        method=method,
                        points=points,
                        embeddings=self.embedding_key,
                        seed=GLOBAL_SEED,
                        brain_key=method_key,
                        num_workers=NUM_WORKERS,
                    )
                    self.dataset.set_values(
                        points_key, self.embeddings_vis[method_key].current_points
                    )

                else:
                    logging.info("Computing vis.")
                    self.embeddings_vis[method_key] = fob.compute_visualization(
                        self.dataset,
                        method=method,
                        embeddings=self.embedding_key,
                        seed=GLOBAL_SEED,
                        brain_key=method_key,
                        num_workers=NUM_WORKERS,
                    )
                    self.dataset.set_values(
                        points_key, self.embeddings_vis[method_key].current_points
                    )

                    with open(vis_file_name, "wb") as f:
                        pickle.dump(self.embeddings_vis[method_key].current_points, f)
        else:
            logging.warning(
                "Model " + self.model_name + " does not provide embeddings."
            )
        end_time = time.time()
        duration = end_time - start_time
        self.writer.add_scalar("brain/duration_in_seconds", duration, self.steps)
        self.steps += 1

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
            self.embeddings_model (dict): A dictionary of embedding models.
            self.brains (dict): A dictionary to store brain results.
            self.similarities (dict): A dictionary to store similarity results.
            self.dataset (Dataset): The dataset object to compute similarities on.

        """

        start_time = time.time()
        if self.similiarity_key in self.brains:
            logging.info("Loading similarities from V51.")
            self.similarities = self.dataset.load_brain_results(self.similiarity_key)

        else:
            logging.info("Computing similarities.")
            self.similarities = fob.compute_similarity(
                self.dataset,
                embeddings=self.embeddings_model,
                brain_key=self.similiarity_key,
                num_workers=NUM_WORKERS,
            )
        end_time = time.time()
        duration = end_time - start_time
        self.writer.add_scalar("brain/duration_in_seconds", duration, self.steps)
        self.steps += 1

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

        start_time = time.time()
        field = self.brain_taxonomy["field"]
        value = self.brain_taxonomy["value_compute_representativeness"]
        methods_cluster_center = ["cluster-center", "cluster-center-downweight"]

        for method in tqdm(methods_cluster_center, desc="Representativeness"):
            method_key = re.sub(
                r"[\W-]+",
                "_",
                "representativeness_" + self.model_name + "_" + method,
            )

            if method_key in self.brains:
                self.representativeness[method_key] = self.dataset.load_brain_results(
                    method_key
                )

            logging.info("Computing representativeness.")
            fob.compute_representativeness(
                self.dataset,
                representativeness_field=method_key,
                method=method,
                embeddings=self.embeddings_model,
                num_workers=NUM_WORKERS,
                progress=True,
            )

            # quant_threshold = self.dataset.quantiles(key, threshold)
            # view = self.dataset.match(F(key) >= quant_threshold)
            view = self.dataset.match(F(method_key) >= threshold)
            # TODO Speed up with values() and set_values()
            for sample in view.iter_samples(progress=True, autosave=True):
                sample[field] = value
        end_time = time.time()
        duration = end_time - start_time
        self.writer.add_scalar("brain/duration_in_seconds", duration, self.steps)
        self.steps += 1

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

        start_time = time.time()
        sample_count = len(self.dataset.view())
        num_of_unique = perct_unique * sample_count
        field = self.brain_taxonomy["field"]
        value = self.brain_taxonomy["value_find_unique"]

        # Check if any sample has the label label_unique:
        dataset_labels = self.dataset.count_sample_tags()
        center_view = self.dataset.match(F(field) == value)

        if field in dataset_labels and len(center_view) > 0:
            logging.info("No unique images.")
            pass

        else:
            self.similarities.find_unique(num_of_unique)
            for unique_id in tqdm(
                self.similarities.unique_ids, desc="Tagging unique images"
            ):
                sample = self.dataset[unique_id]
                sample[field] = value
                sample.save()
        end_time = time.time()
        duration = end_time - start_time
        self.writer.add_scalar("brain/duration_in_seconds", duration, self.steps)
        self.steps += 1

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

        start_time = time.time()
        field = self.brain_taxonomy["field"]
        value = self.brain_taxonomy["value_compute_uniqueness"]

        fob.compute_uniqueness(
            self.dataset,
            embeddings=self.embeddings_model,
            uniqueness_field=self.uniqueness_key,
            num_workers=NUM_WORKERS,
        )

        # quant_threshold = self.dataset.quantiles(key, threshold)
        # view = self.dataset.match(F(key) >= quant_threshold)
        view = self.dataset.match(F(self.uniqueness_key) >= threshold)
        # TODO Improve with values() and set_values()
        for sample in view.iter_samples(progress=True, autosave=True):
            sample[field] = value
        end_time = time.time()
        duration = end_time - start_time
        self.writer.add_scalar("brain/duration_in_seconds", duration, self.steps)
        self.steps += 1

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
        start_time = time.time()
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

            for unique_view, value in tqdm(views_values, desc="Tagging similar images"):
                for sample in unique_view:
                    view = self.dataset.sort_by_similarity(
                        sample.id,
                        k=neighbour_count,
                        brain_key=self.similiarity_key,
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
        end_time = time.time()
        duration = end_time - start_time
        self.writer.add_scalar("brain/duration_in_seconds", duration, self.steps)
        self.steps += 1
