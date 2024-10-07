import fiftyone as fo
from fiftyone import ViewField as F

from anomalib import TaskType
from anomalib.data.image.folder import Folder
from anomalib.deploy import ExportType, OpenVINOInferencer
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, Draem
from anomalib.loggers import AnomalibWandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms.v2 import Resize
import torch
import logging

from config.config import NUM_WORKERS, GLOBAL_SEED

# https://docs.voxel51.com/tutorials/anomaly_detection.html
# https://medium.com/@enrico.randellini/anomalib-a-library-for-image-anomaly-detection-and-localization-fb363639104f
# https://github.com/openvinotoolkit/anomalib
# https://anomalib.readthedocs.io/en/v1.1.1/


class Anodec:
    def __init__(
        self,
        dataset,
        dataset_info,
        models_path="./output/models/anomalib/",
        model=Draem(),
    ):
        torch.set_float32_matmul_precision(
            "medium"
        )  # Utilize Tensor core, came in warning
        self.dataset = dataset
        self.normal_data = dataset.match_tags("train")
        self.abnormal_data = dataset.match_tags("val")
        self.brains = dataset.list_brain_runs()
        self.dataset_name = dataset_info["name"]
        self.TASK = TaskType.SEGMENTATION
        self.IMAGE_SIZE = (256, 256)  ## preprocess image size for uniformity
        self.filepath_masks = dataset_info["anomalib_masks_path"]
        self.model = model
        self.model_key = type(model).__name__
        self.models_path = models_path
        self.inferencer = None

    def create_datamodule(self, transform=None):
        ## Build transform
        # We create subsets of our data containing only the “good” training images and “anomalous” images for validation.
        # We symlink the images and masks to the directory Anomalib expects.
        # We instantiate and setup a datamodule from Anomalib’s Folder, which is the general-purpose class for custom datasets.
        #  It is also possible to create a torch DataLoader from scratch and pass it to the engine’s fit() method
        if transform is None:
            transform = Resize(self.IMAGE_SIZE, antialias=True)

        filepath_train = self.normal_data.take(1).first().filepath
        filepath_val = self.abnormal_data.take(1).first().filepath

        normal_dir = os.path.dirname(filepath_train)
        abnormal_dir = os.path.dirname(filepath_val)
        mask_dir = os.path.dirname(self.filepath_masks)

        # Symlink the images and masks to the directory Anomalib expects.
        for sample in self.abnormal_data.iter_samples():
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

        if self.model_key == "Draem":
            batch_size = 16
        else:
            batch_size = 32

        datamodule = Folder(
            name=self.dataset_name,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            mask_dir=mask_dir,
            task=self.TASK,
            transform=transform,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=NUM_WORKERS,
            seed=GLOBAL_SEED,
        )
        datamodule.setup()
        return datamodule

    def train_and_export_model(
        self, transform=None, max_epochs=100, early_stop_patience=5
    ):
        # Now we can put it all together. The train_and_export_model() function
        # below trains an anomaly detection model using Anomalib’s Engine class,
        # exports the model to OpenVINO, and returns the model “inferencer” object.
        # The inferencer object is used to make predictions on new images.

        openvino_root = os.path.join(
            self.models_path, self.model_key, self.dataset_name
        )
        openvino_model_path = os.path.join(
            openvino_root,
            "latest/weights/openvino/model.bin",
        )
        metadata_path = os.path.join(
            openvino_root,
            "latest/weights/openvino/metadata.json",
        )

        if not (os.path.exists(openvino_model_path) or os.path.exists(metadata_path)):
            os.makedirs(self.models_path, exist_ok=True)
            datamodule = self.create_datamodule(transform=transform)
            wandb_logger = AnomalibWandbLogger(
                name="anomalib_" + self.dataset_name + "_" + self.model_key
            )

            # Callbacks
            callbacks = [
                ModelCheckpoint(
                    mode="max",
                    monitor="pixel_AUROC",
                    save_last=True,
                    verbose=True,
                    auto_insert_metric_name=True,
                    every_n_epochs=1,
                ),
                EarlyStopping(
                    monitor="pixel_AUROC", mode="max", patience=early_stop_patience
                ),
            ]
            kwargs = {
                "default_root_dir": os.path.join(openvino_root, "lightning")
            }  # Custom log directory path
            engine = Engine(
                task=self.TASK,
                default_root_dir=self.models_path,
                logger=wandb_logger,
                max_epochs=max_epochs,
                callbacks=callbacks,
                pixel_metrics=[  # https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/metrics/index.html
                    "AUPIMO",
                    "AUPR",
                    "AUPRO",
                    "AUROC",
                    "AnomalyScoreDistribution",
                    "BinaryPrecisionRecallCurve",
                    "F1AdaptiveThreshold",
                    "F1Max",
                    "F1Score",
                    "ManualThreshold",
                    "MinMax",
                    "PIMO",
                    "PRO",
                ],
                accelerator="auto",
                **kwargs,
            )
            engine.fit(model=self.model, datamodule=datamodule)

            test_results = engine.test(
                model=self.model,
                datamodule=datamodule,
                ckpt_path=engine.trainer.checkpoint_callback.best_model_path,
            )

            logging.info(test_results)

            engine.export(
                model=self.model,
                export_type=ExportType.OPENVINO,
            )

        inferencer = OpenVINOInferencer(
            path=openvino_model_path,
            metadata=metadata_path,
            device="CPU",  # TODO: Just use pytorch model, OpenVINO does not support Nvidia GPUs https://anomalib.readthedocs.io/en/v0.3.7/tutorials/inference.html
        )
        self.inferencer = inferencer

    def run_inference(self, threshold=0.5):
        # Take a FiftyOne sample collection (e.g. our test set) as input, along with the inferencer object,
        # and a key for storing the results in the samples. It will run the model on each sample in the collection
        # and store the results. The threshold argument acts as a cutoff for the anomaly score.
        # If the score is above the threshold, the sample is considered anomalous
        for sample in self.abnormal_data.iter_samples(autosave=True, progress=True):
            output = self.inferencer.predict(image=Image.open(sample.filepath))

            conf = output.pred_score
            anomaly = "normal" if conf < threshold else "anomaly"

            sample[f"pred_anomaly_score_{self.model_key}"] = conf
            sample[f"pred_anomaly_{self.model_key}"] = fo.Classification(label=anomaly)
            sample[f"pred_anomaly_map_{self.model_key}"] = fo.Heatmap(
                map=output.anomaly_map
            )
            sample[f"pred_defect_mask_{self.model_key}"] = fo.Segmentation(
                mask=output.pred_mask
            )

    def eval(self):
        eval_seg = self.abnormal_data.evaluate_segmentations(
            f"pred_defect_mask_{self.model_key}",
            gt_field="anomaly_mask",
            eval_key=f"eval_seg_{self.model_key}",
        )
        eval_seg.print_report(classes=[0, 255])
