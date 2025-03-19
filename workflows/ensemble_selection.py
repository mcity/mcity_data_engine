import logging
import time

import numpy as np
from fiftyone import ViewField as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    # box format: [minx, miny, width, height] normalized between [0,1]
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2
    x1_max, y1_max = x1_min + w1, y1_min + h1
    x2_max, y2_max = x2_min + w2, y2_min + h2

    # Calculate the intersection box
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        inter_area = 0

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculcate_bbox_size(box):
    """Calculate the size of a bounding box."""
    return box[2] * box[3]


class EnsembleSelection:
    """Performs ensemble selection to identify overlapping detections from multiple models based on IoU and agreement thresholds."""

    def __init__(self, dataset, config):
        """Initializes ensemble selection with dataset and configuration for detection agreement analysis."""
        self.dataset = dataset
        self.config = config
        self.positive_classes = config["positive_classes"]
        self.agreement_threshold = config["agreement_threshold"]
        self.iou_threshold = config["iou_threshold"]
        self.max_bbox_size = config["max_bbox_size"]
        pred_prefix = config["field_includes"]
        self.v51_agreement_tag = "detections_overlap"
        self.n_unique_field = "n_unique_ensemble_selection"
        self.n_unique_id_field = "n_unique_id_ensemble_selection"

        logging.info(
            f"Collecting detections of fields with prefix '{pred_prefix}'. Successful detections will be tagged with '{self.v51_agreement_tag}'."
        )

        # Get V51 fields that store detection results
        self.v51_detection_fields = []
        dataset_schema = self.dataset.get_field_schema()
        for field in tqdm(
            dataset_schema, desc="Collecting dataset fields with detections"
        ):
            if pred_prefix in field:
                self.v51_detection_fields.append(field)

                # Make sure that some classes in 'positive_classes' are part of the classes used for detections
                detection_classes = self.dataset.distinct("%s.detections.label" % field)
                pos_set = set(self.positive_classes)
                det_set = set(detection_classes)
                common_classes = pos_set & det_set
                if common_classes:
                    logging.info(
                        f"The classes {common_classes} are shared between the list of positive classes and detections in '{field}'."
                    )
                else:
                    logging.warning(
                        f"No classes in the list of positive classes {pos_set} are part of the classes used for detections in '{field}'."
                    )

        if len(self.v51_detection_fields) < self.agreement_threshold:
            logging.error(
                f"Number of detection models used ({len(self.v51_detection_fields)}) is less than the agreement threshold ({self.agreement_threshold}). No agreements will be possible. Detections are expected in the field {pred_prefix}. Detections can be generated with the workflow `auto_labeling_zero_shot`"
            )

        # Get filtered V51 view for faster processing
        conditions = [
            (F(f"{field}") != None)  # Field exists
            & (F(f"{field}.detections") != [])  # Field has detections
            & F(f"{field}.detections.label").contains(
                self.positive_classes
            )  # Detections include "cat" or "dog"
            for field in self.v51_detection_fields
        ]

        self.view = self.dataset.match(F.any(conditions))

        for field in tqdm(
            self.v51_detection_fields,
            desc="Generating filtered Voxel51 view for fast processing",
        ):
            self.view = self.view.filter_labels(
                f"{field}.detections",
                F("label").is_in(self.positive_classes),
                only_matches=False,
            )
        self.n_samples = len(self.view)

    def ensemble_selection(self):
        """Selects and tags overlapping detections from multiple models using IoU-based ensemble method, tracking unique VRU detections."""

        writer = SummaryWriter(log_dir="logs/tensorboard/ensemble_selection")

        # Get detections from V51 with efficient "values" method
        samples_detections = []  # List of lists of list [model][sample][detections]
        for field in tqdm(
            self.v51_detection_fields, desc="Collecting model detections"
        ):
            field_detections = self.view.values(
                f"{field}.detections"
            )  # list of lists of detections per sample
            samples_detections.append(field_detections)

        # Cleaning up tags from previous runs
        for i in tqdm(range(self.n_samples), desc="Cleaning up tags"):
            for j in range(len(self.v51_detection_fields)):
                detections = samples_detections[j][i]
                if detections:
                    for k in range(len(detections)):
                        samples_detections[j][i][k].tags = [
                            x
                            for x in samples_detections[j][i][k].tags
                            if x != self.v51_agreement_tag
                        ]

        # Counting variables
        n_bboxes_agreed = 0
        n_samples_agreed = 0
        n_unique_vru = 0

        # Iterate over all samples and check overlapping detections
        for step, sample_index in enumerate(
            tqdm(range(self.n_samples), desc="Finding detection agreements in samples")
        ):
            start_time = time.time()
            unique_vru_detections_set = set()
            all_bboxes = []  # List of all bounding box detections per sample
            bbox_model_indices = []  # Track which model each bounding box belongs to
            bbox_detection_indices = (
                []
            )  # Track the index of each detection in the model's list

            for model_index, model_detections in enumerate(samples_detections):
                for detection_index, det in enumerate(model_detections[sample_index]):
                    all_bboxes.append(det.bounding_box)
                    bbox_model_indices.append(model_index)
                    bbox_detection_indices.append(detection_index)

            n_bboxes = len(all_bboxes)
            if n_bboxes == 0:
                logging.warning(f"No detections found in sample {sample_index}.")
                continue

            # Compute IoU between all bounding boxes and store additional information
            involved_models_matrix = np.full((n_bboxes, n_bboxes), -1)
            bbox_ids_matrix = np.full((n_bboxes, n_bboxes), -1)
            for a in range(n_bboxes):
                for b in range(a + 1, n_bboxes):
                    iou = calculate_iou(all_bboxes[a], all_bboxes[b])
                    # Only compare detections of small bounding boxes
                    if (
                        self.max_bbox_size
                        and calculcate_bbox_size(all_bboxes[a]) <= self.max_bbox_size
                        and calculcate_bbox_size(all_bboxes[b]) <= self.max_bbox_size
                    ):
                        # Only compare detections of high IoU scores
                        if iou > self.iou_threshold:
                            involved_models_matrix[a, b] = bbox_model_indices[
                                b
                            ]  # Store model index that was compared to
                            involved_models_matrix[b, a] = bbox_model_indices[a]
                            involved_models_matrix[a, a] = bbox_model_indices[
                                a
                            ]  # Trick to also store the model itself, as the diagonal is not used (b = a+1, symmetry)
                            involved_models_matrix[b, b] = bbox_model_indices[b]

                            bbox_ids_matrix[a, b] = (
                                b  # Store detection indices to get involved bounding_boxes
                            )
                            bbox_ids_matrix[b, a] = a
                            bbox_ids_matrix[a, a] = a
                            bbox_ids_matrix[b, b] = b

            # Get number of involved models by finding unique values in rows
            # "-1" ist not an involved model
            involved_models = [np.unique(row) for row in involved_models_matrix]
            for index in range(len(involved_models)):
                involved_models[index] = involved_models[index][
                    involved_models[index] != -1
                ]

            # Get list of involved bounding box indices
            involved_bboxes = [np.unique(row) for row in bbox_ids_matrix]
            for index in range(len(bbox_ids_matrix)):
                involved_bboxes[index] = involved_bboxes[index][
                    involved_bboxes[index] != -1
                ]

            # Checking that all arrays have the same lengths
            if not (
                len(all_bboxes)
                == len(bbox_model_indices)
                == len(bbox_detection_indices)
                == len(involved_models)
            ):
                logging.error(
                    "Array lengths mismatch: all_bboxes(%d), bbox_model_indices(%d), bbox_detection_indices(%d), involved_models(%d)",
                    len(all_bboxes),
                    len(bbox_model_indices),
                    len(bbox_detection_indices),
                    len(involved_models),
                )

            # Check bbox detections for agreements
            for index in range(n_bboxes):
                model_indices = involved_models[index]
                bbox_indices = involved_bboxes[index]
                all_connected_boxes = bbox_indices

                if len(model_indices) >= self.agreement_threshold:
                    # Get all involved bounding boxe indices
                    for bbox_index in bbox_indices:
                        connected_bboxes = involved_bboxes[bbox_index]
                        all_connected_boxes = np.unique(
                            np.concatenate((all_connected_boxes, connected_bboxes))
                        )
                    unique_detection_id = np.min(all_connected_boxes)

                    # If bounding box has not been processed yet
                    if unique_detection_id not in unique_vru_detections_set:
                        unique_vru_detections_set.add(unique_detection_id)
                        # Set V51 tag to all connected boxes
                        for bbox_index in all_connected_boxes:
                            model_index = bbox_model_indices[bbox_index]
                            det_index = bbox_detection_indices[bbox_index]
                            samples_detections[model_index][sample_index][
                                det_index
                            ].tags.append(self.v51_agreement_tag)
                            samples_detections[model_index][sample_index][det_index][
                                self.n_unique_id_field
                            ] = unique_detection_id
                            n_bboxes_agreed += 1

            if len(unique_vru_detections_set) > 0:
                n_samples_agreed += 1
                n_unique_vru += len(unique_vru_detections_set)

            # Log inference performance
            end_time = time.time()
            sample_duration = end_time - start_time
            frames_per_second = 1 / sample_duration
            writer.add_scalar("inference/frames_per_second", frames_per_second, step)

        # Save results to dataset
        for field, field_detections in tqdm(
            zip(self.v51_detection_fields, samples_detections),
            desc="Saving results to dataset",
            total=len(self.v51_detection_fields),
        ):
            self.view.set_values(field + ".detections", field_detections)

        logging.info("Calculate number of unique detections per sample")
        try:
            self.dataset.delete_sample_field(self.n_unique_field)
        except:
            pass
        view_tagged = self.view.select_labels(tags=self.v51_agreement_tag)
        for sample in view_tagged.iter_samples(progress=True, autosave=True):
            n_unique_set = set()
            for field in self.v51_detection_fields:
                detections = sample[field].detections
                n_unique_set.update(d[self.n_unique_id_field] for d in detections)
            sample[self.n_unique_field] = len(n_unique_set)

        logging.info(
            f"Found {n_unique_vru} unique detections in {n_samples_agreed} samples. Based on {n_bboxes_agreed} total detections with {self.agreement_threshold} or more overlapping detections."
        )
