import logging
import os

import fiftyone as fo
import plotly.graph_objects as go
from fiftyone import ViewField as F
from PIL import Image, ImageDraw

from config.config import WORKFLOWS
from utils.selector import select_by_class


class AnomalyDetectionDataPreparation:
    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_ano_dec = None
        self.dataset_name = dataset_name
        self.export_root = "output/datasets/"
        self.config = WORKFLOWS["anomaly_detection"]["data_preparation"].get(
            self.dataset_name, None
        )
        if self.config is None:
            logging.error(
                f"Data preparation config for dataset {self.dataset_name} missing"
            )

        SUPPORTED_DATASETS = {"fisheye8k"}

        if self.dataset_name in SUPPORTED_DATASETS:
            # Call method that is named like dataset
            method = getattr(self, self.dataset_name)
            method()
        else:
            logging.error(
                f"Dataset {self.dataset_name} is currently not supported for Anomaly Detection. Please prepare a workflow to prepare to define normality and a rare class."
            )
            return None

    def fisheye8k(self, gt_field="ground_truth"):

        location_filter = self.config.get("location", "cam1")
        rare_classes = self.config.get("rare_classes", ["Truck"])
        # Filter to only include data from one camera to make the data distribution clearer
        view_location = self.dataset.match(F("location") == location_filter)

        # Build training and validation datasets
        view_train = select_by_class(view_location, classes_out=rare_classes)
        view_val = select_by_class(view_location, classes_in=rare_classes)

        # Data export
        dataset_name_ano_dec = f"{self.dataset_name}_anomaly_detection"
        export_dir = os.path.join(self.export_root, dataset_name_ano_dec)

        classes = self.dataset.distinct("ground_truth.detections.label")
        dataset_splits = ["train", "val"]
        dataset_type = fo.types.YOLOv5Dataset

        view_train.export(
            export_dir=export_dir,
            dataset_type=dataset_type,
            label_field=gt_field,
            split=dataset_splits[0],
            classes=classes,
        )

        view_val.export(
            export_dir=export_dir,
            dataset_type=dataset_type,
            label_field=gt_field,
            split=dataset_splits[1],
            classes=classes,
        )

        # Load the exported dataset
        if dataset_name_ano_dec in fo.list_datasets():
            dataset_ano_dec = fo.load_dataset(dataset_name_ano_dec)
            logging.info(f"Existing dataset {dataset_name_ano_dec} was loaded.")
        else:
            dataset_ano_dec = fo.Dataset(dataset_name_ano_dec)
            for split in dataset_splits:
                dataset_ano_dec.add_dir(
                    dataset_dir=export_dir,
                    dataset_type=dataset_type,
                    split=split,
                    tags=split,
                )
            dataset_ano_dec.compute_metadata()

        self.dataset_ano_dec = dataset_ano_dec

        # Select samples that include a rare class
        anomalous_view = dataset_ano_dec.match_tags("val", "test")
        logging.info(f"Processing {len(anomalous_view)} val samples")

        # Prepare data for Anomalib
        dataset_name_ano_dec_masks = f"{dataset_name_ano_dec}_masks"
        export_dir_masks = os.path.join(self.export_root, dataset_name_ano_dec_masks)
        os.makedirs(export_dir_masks, exist_ok=True)

        for sample in anomalous_view.iter_samples(progress=True):
            img_width = sample.metadata.width
            img_height = sample.metadata.height
            mask = Image.new("L", (img_width, img_height), 0)  # Create a black image
            draw = ImageDraw.Draw(mask)
            for bbox in sample.ground_truth.detections:
                if bbox.label in rare_classes:
                    # Convert V51 format to image format

                    x_min_rel, y_min_rel, width_rel, height_rel = bbox.bounding_box
                    x_min = int(x_min_rel * img_width)
                    y_min = int(y_min_rel * img_height)
                    x_max = int((x_min_rel + width_rel) * img_width)
                    y_max = int((y_min_rel + height_rel) * img_height)

                    # draw.rectangle([x0, y0, x1, y1], fill=255)  # [x0, y0, x1, y1]
                    draw.rectangle(
                        [x_min, y_min, x_max, y_max], fill=255
                    )  # [x0, y0, x1, y1]

            # Save the mask
            filename = os.path.basename(sample.filepath).replace(".jpg", ".png")
            mask.save(os.path.join(export_dir_masks, f"{filename}"))
