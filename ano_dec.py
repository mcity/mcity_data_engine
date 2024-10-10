import fiftyone as fo
from fiftyone import ViewField as F

import anomalib.models
from anomalib import TaskType
from anomalib.data.image.folder import Folder
from anomalib.data.utils import read_image
from anomalib.deploy import ExportType, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import (
    Cfa,
    Cflow,
    Csflow,
    Dfkde,
    Dfm,
    Draem,
    Dsr,
    EfficientAd,
    Fastflow,
    Ganomaly,
    Padim,
    Patchcore,
    ReverseDistillation,
    Rkde,
    Stfpm,
    Uflow,
    WinClip,
)
from anomalib.loggers import AnomalibTensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import wandb

import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms.v2 import Resize
import torch
import logging

from config.config import NUM_WORKERS, GLOBAL_SEED, ANOMALIB_EVAL_METRICS

# https://docs.voxel51.com/tutorials/anomaly_detection.html
# https://medium.com/@enrico.randellini/anomalib-a-library-for-image-anomaly-detection-and-localization-fb363639104f
# https://github.com/openvinotoolkit/anomalib
# https://anomalib.readthedocs.io/en/v1.1.1/


class Anodec:
    def __init__(
        self,
        dataset,
        dataset_info,
        anomalib_output_root="./output/models/anomalib/",
        model_name="Padim",
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
        try:
            # Setting wandb config allows for overwriting from the WandB interface for new runs
            wandb.config["anomalib_model"] = model_name
            self.model = getattr(anomalib.models, wandb.config["anomalib_model"])()
            self.model_key = wandb.config["anomalib_model"]
        except:
            logging.error(
                "Chosen anomalib model "
                + model_name
                + " is no valid model. Please select from https://anomalib.readthedocs.io/en/v1.1.1/markdown/guides/reference/models/image/index.html."
            )
        self.anomalib_output_root = anomalib_output_root
        self.model_path = os.path.join(
            anomalib_output_root,
            self.model_key,
            self.dataset_name,
            "weights/torch/model.pt",
        )

        filepath_masks = dataset_info["anomalib_masks_path"]
        filepath_train = self.normal_data.take(1).first().filepath
        filepath_val = self.abnormal_data.take(1).first().filepath

        self.normal_dir = os.path.dirname(filepath_train)
        self.abnormal_dir = os.path.dirname(filepath_val)
        self.mask_dir = os.path.dirname(filepath_masks)

        # Anomalib objects
        self.inferencer = None
        self.engine = None
        self.datamodule = None

    def __del__(self):
        logging.warning("UNLINKING SYMLINKS + CLOSING WANDB RUN")
        try:
            wandb.finish()
        except:
            pass
        try:
            self.unlink_symlinks()
        except:
            pass

    def create_datamodule(self, transform=None):
        ## Build transform
        # We create subsets of our data containing only the “good” training images and “anomalous” images for validation.
        # We symlink the images and masks to the directory Anomalib expects.
        # We instantiate and setup a datamodule from Anomalib’s Folder, which is the general-purpose class for custom datasets.
        #  It is also possible to create a torch DataLoader from scratch and pass it to the engine’s fit() method
        if transform is None:
            transform = Resize(self.IMAGE_SIZE, antialias=True)

        # Symlink the images and masks to the directory Anomalib expects.
        for sample in self.abnormal_data.iter_samples():
            # Add mask groundtruth
            base_filename = sample.filename
            mask_filename = os.path.basename(base_filename).replace(".jpg", ".png")
            mask_path = os.path.join(self.mask_dir, mask_filename)
            sample["anomaly_mask"] = fo.Segmentation(mask_path=mask_path)
            sample.save()

            dir_name = os.path.dirname(sample.filepath).split("/")[-1]
            new_filename = f"{dir_name}_{base_filename}"
            if not os.path.exists(os.path.join(self.abnormal_dir, new_filename)):
                os.symlink(
                    sample.filepath, os.path.join(self.abnormal_dir, new_filename)
                )

            if not os.path.exists(os.path.join(self.mask_dir, new_filename)):
                os.symlink(
                    sample.anomaly_mask.mask_path,
                    os.path.join(self.mask_dir, new_filename),
                )

        if self.model_key == "Draem":
            wandb.config["batch_size"] = 16
        else:
            wandb.config["batch_size"] = 32

        self.datamodule = Folder(
            name=self.dataset_name,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            mask_dir=self.mask_dir,
            task=self.TASK,
            transform=transform,
            train_batch_size=wandb.config["batch_size"],
            eval_batch_size=wandb.config["batch_size"],
            num_workers=NUM_WORKERS,
            seed=GLOBAL_SEED,
        )
        self.datamodule.setup()

    def unlink_symlinks(self):
        for sample in self.abnormal_data.iter_samples():
            base_filename = sample.filename
            dir_name = os.path.dirname(sample.filepath).split("/")[-1]
            new_filename = f"{dir_name}_{base_filename}"

            try:
                os.unlink(os.path.join(self.abnormal_dir, new_filename))
            except:
                pass

            try:
                os.unlink(os.path.join(self.mask_dir, new_filename))
            except:
                pass

    def train_and_export_model(
        self, transform=None, max_epochs=100, early_stop_patience=5
    ):
        # Now we can put it all together. The train_and_export_model() function
        # below trains an anomaly detection model using Anomalib’s Engine class,
        # exports the model to OpenVINO, and returns the model “inferencer” object.
        # The inferencer object is used to make predictions on new images.

        if not os.path.exists(self.model_path):
            os.makedirs(self.anomalib_output_root, exist_ok=True)
            self.unlink_symlinks()
            self.create_datamodule(transform=transform)
            wandb_logger = AnomalibTensorBoardLogger(
                save_dir="./output/tb_logs",
                name="anomalib_" + self.dataset_name + "_" + self.model_key,
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
            self.engine = Engine(
                task=self.TASK,
                default_root_dir=self.anomalib_output_root,
                logger=wandb_logger,
                max_epochs=max_epochs,
                callbacks=callbacks,
                image_metrics=ANOMALIB_EVAL_METRICS,
                pixel_metrics=ANOMALIB_EVAL_METRICS,
                accelerator="auto",
            )
            self.engine.fit(model=self.model, datamodule=self.datamodule)

            # Test model
            test_results = self.engine.test(
                model=self.model,
                datamodule=self.datamodule,
                ckpt_path=self.engine.trainer.checkpoint_callback.best_model_path,
            )
            logging.info(test_results)

            # Export and generate inferencer
            export_root = self.model_path.replace("weights/torch/model.pt", "")
            self.engine.export(
                model=self.model,
                export_root=export_root,
                export_type=ExportType.TORCH,
                ckpt_path=self.engine.trainer.checkpoint_callback.best_model_path,
            )

        inferencer = TorchInferencer(
            path=os.path.join(self.model_path),
            device="cuda",
        )
        self.inferencer = inferencer

    def run_inference(self, threshold=0.5):
        # Take a FiftyOne sample collection (e.g. our test set) as input, along with the inferencer object,
        # and a key for storing the results in the samples. It will run the model on each sample in the collection
        # and store the results. The threshold argument acts as a cutoff for the anomaly score.
        # If the score is above the threshold, the sample is considered anomalous
        for sample in self.abnormal_data.iter_samples(autosave=True, progress=True):
            # output = self.engine.predict(
            #    datamodule=self.datamodule,
            #    model=self.model,
            #    ckpt_path=self.engine.trainer.checkpoint_callback.best_model_path,
            # )
            # image = Image.open(sample.filepath)
            image = read_image(sample.filepath, as_tensor=True)
            output = self.inferencer.predict(image)
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

    def eval_v51(self):
        eval_seg = self.abnormal_data.evaluate_segmentations(
            f"pred_defect_mask_{self.model_key}",
            gt_field="anomaly_mask",
            eval_key=f"eval_seg_{self.model_key}",
        )
        eval_seg.print_report(classes=[0, 255])
