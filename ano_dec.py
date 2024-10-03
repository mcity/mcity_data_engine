import fiftyone as fo
from fiftyone import ViewField as F

from anomalib import TaskType
from anomalib.data.image.folder import Folder
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms.v2 import Resize

import logging

# https://docs.voxel51.com/tutorials/anomaly_detection.html
# https://github.com/openvinotoolkit/anomalib
# https://anomalib.readthedocs.io/en/v1.1.1/


class Anodec:
    def __init__(self, dataset, dataset_info, embeddings_path="./datasets/embeddings/"):
        self.dataset = dataset
        self.brains = dataset.list_brain_runs()
        self.dataset_name = dataset_info["name"]
        self.TASK = TaskType.SEGMENTATION
        self.IMAGE_SIZE = (256, 256)  ## preprocess image size for uniformity
        self.filepath_masks = dataset_info["anomalib_masks_path"]

    def create_datamodule(self, transform=None):
        ## Build transform
        if transform is None:
            transform = Resize(self.IMAGE_SIZE, antialias=True)

        normal_data = self.dataset.match_tags("train")
        abnormal_data = self.dataset.match_tags("val")

        filepath_train = normal_data.take(1).first().filepath
        filepath_val = abnormal_data.take(1).first().filepath

        normal_dir = os.path.dirname(filepath_train)
        abnormal_dir = os.path.dirname(filepath_val)
        mask_dir = os.path.dirname(self.filepath_masks)

        # Symlink the images and masks to the directory Anomalib expects.
        for sample in abnormal_data.iter_samples():
            # Add mask groundtruth
            base_filename = sample.filename
            mask_filename = os.path.basename(base_filename).replace(".jpg", ".png")
            mask_path = os.path.join(mask_dir, mask_filename)
            sample["anomaly_mask"] = fo.Segmentation(mask_path=mask_path)
            sample.save()

            dir_name = os.path.dirname(sample.filepath).split("/")[-1]
            new_filename = f"{dir_name}_{base_filename}"
            if not os.path.exists(os.path.join(abnormal_dir, new_filename)):
                os.symlink(sample.filepath, os.path.join(abnormal_dir, new_filename))

            if not os.path.exists(os.path.join(mask_dir, new_filename)):
                os.symlink(
                    sample.anomaly_mask.mask_path, os.path.join(mask_dir, new_filename)
                )

        datamodule = Folder(
            name="pedestrians",
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            mask_dir=mask_dir,
            task=self.TASK,
            transform=transform,
        )
        datamodule.setup()
        return datamodule
