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

        # TODO Get train and val dir paths from V51 splits

    def create_datamodule(self, object_type, transform=None):
        ## Build transform
        if transform is None:
            transform = Resize(self.IMAGE_SIZE, antialias=True)

        normal_data = self.dataset.match(F("category.label") == object_type).match(
            F("split") == "train"
        )
        abnormal_data = (
            self.dataset.match(F("category.label") == object_type)
            .match(F("split") == "test")
            .match(F("defect.label") != "good")
        )

        normal_dir = Path(ROOT_DIR) / object_type / "normal"  # FIXME
        abnormal_dir = ROOT_DIR / object_type / "abnormal"  # FIXME

        # create directories if they do not exist
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(abnormal_dir, exist_ok=True)

        if not os.path.exists(str(normal_dir)):
            normal_data.export(
                export_dir=str(normal_dir),
                dataset_type=fo.types.ImageDirectory,
                export_media="symlink",
            )

        for sample in abnormal_data.iter_samples():
            base_filename = sample.filename
            dir_name = os.path.dirname(sample.filepath).split("/")[-1]
            new_filename = f"{dir_name}_{base_filename}"
            if not os.path.exists(str(abnormal_dir / new_filename)):
                os.symlink(sample.filepath, str(abnormal_dir / new_filename))

        datamodule = Folder(
            name=object_type,
            root=ROOT_DIR,  # FIXME
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            task=self.TASK,
            transform=transform,
        )
        datamodule.setup()
        return datamodule
