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
import wandb
from utils.wandb_helper import get_wandb_conf
from anomalib.loggers import AnomalibTensorBoardLogger, AnomalibWandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms.v2 import Resize
import torch
import logging

from config.config import NUM_WORKERS, GLOBAL_SEED, ANOMALIB_EVAL_METRICS, WANDB_CONFIG

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
        config=None,
        wandb_project="Data Engine",
        wandb_group="Anomalib",
    ):
        self.config = config
        torch.set_float32_matmul_precision(
            "medium"
        )  # Utilize Tensor core, came in warning
        self.dataset = dataset
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        self.normal_data = dataset.match_tags("train")
        self.abnormal_data = dataset.match_tags("val")
        self.brains = dataset.list_brain_runs()
        self.dataset_name = dataset_info["name"]
        self.TASK = TaskType.SEGMENTATION
        self.IMAGE_SIZE = (256, 256)  ## preprocess image size for uniformity
        self.model_name = get_wandb_conf(self.config, "v51_dataset_name")
        self.anomalib_output_root = anomalib_output_root
        self.model_path = os.path.join(
            anomalib_output_root,
            self.model_name,
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
        self.anomalib_logger = None

    def __del__(self):
        try:
            self.unlink_symlinks()
            self.anomalib_logger.finalize("success")
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

        self.datamodule = Folder(
            name=self.dataset_name,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            mask_dir=self.mask_dir,
            task=self.TASK,
            transform=transform,
            train_batch_size=self.config["batch_size"],
            eval_batch_size=self.config["batch_size"],
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

    def train_and_export_model(self, transform=None):
        # Now we can put it all together. The train_and_export_model() function
        # below trains an anomaly detection model using Anomalib’s Engine class,
        # exports the model to OpenVINO, and returns the model “inferencer” object.
        # The inferencer object is used to make predictions on new images.

        MAX_EPOCHS = self.config["epochs"]
        PATIENCE = self.config["early_stop_patience"]

        # FIXME if not os.path.exists(self.model_path):
        wandb_run = wandb.init(
            allow_val_change=True,
            sync_tensorboard=True,
            config=self.config,
            entity=WANDB_CONFIG["entity"],
            project=self.wandb_project,
            group=self.wandb_group,
            job_type="train",
            tags=[self.dataset_name, self.model_name],
        )
        self.model = getattr(anomalib.models, self.model_name)()

        os.makedirs(self.anomalib_output_root, exist_ok=True)
        os.makedirs("./logs/tensorboard", exist_ok=True)
        self.unlink_symlinks()
        self.create_datamodule(transform=transform)
        self.anomalib_logger = AnomalibTensorBoardLogger(
            save_dir="./logs/tensorboard",
            name="anomalib_" + self.dataset_name + "_" + self.model_name,
            # project="mcity-data-engine",
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
            EarlyStopping(monitor="pixel_AUROC", mode="max", patience=PATIENCE),
        ]
        self.engine = Engine(
            task=self.TASK,
            default_root_dir=self.anomalib_output_root,
            logger=self.anomalib_logger,
            max_epochs=MAX_EPOCHS,
            callbacks=callbacks,
            image_metrics=ANOMALIB_EVAL_METRICS,
            pixel_metrics=ANOMALIB_EVAL_METRICS,
            accelerator="auto",
        )
        self.engine.fit(model=self.model, datamodule=self.datamodule)

        # Export and generate inferencer
        export_root = self.model_path.replace("weights/torch/model.pt", "")
        self.engine.export(
            model=self.model,
            export_root=export_root,
            export_type=ExportType.TORCH,
            ckpt_path=self.engine.trainer.checkpoint_callback.best_model_path,
        )
        wandb_run.finish

    def validate_model(self):
        wandb_run = wandb.init(
            allow_val_change=True,
            sync_tensorboard=True,
            config=self.config,
            entity=WANDB_CONFIG["entity"],
            project=self.wandb_project,
            group=self.wandb_group,
            job_type="eval",
            tags=[self.dataset_name, self.model_name],
        )
        # Test model
        if self.engine:
            test_results = self.engine.test(
                model=self.model,
                datamodule=self.datamodule,
                ckpt_path=self.engine.trainer.checkpoint_callback.best_model_path,
            )
            logging.info(test_results)
        else:
            pass  # TODO Load model from path like in inference mode
        wandb_run.finish

    def run_inference(self, threshold=0.5):
        # Take a FiftyOne sample collection (e.g. our test set) as input, along with the inferencer object,
        # and a key for storing the results in the samples. It will run the model on each sample in the collection
        # and store the results. The threshold argument acts as a cutoff for the anomaly score.
        # If the score is above the threshold, the sample is considered anomalous
        if self.engine:
            outputs = self.engine.predict(
                model=self.model,
                datamodule=self.datamodule,
                ckpt_path=self.engine.trainer.checkpoint_callback.best_model_path,
            )
            for sample, output in zip(
                self.abnormal_data.iter_samples(autosave=True, progress=True), outputs
            ):
                if sample.filepath != output["image_path"][0]:
                    logging.error("Sample and output are not aligned!")
                    continue
                conf = output["pred_scores"].item()
                anomaly = "normal" if conf < threshold else "anomaly"

                sample[f"pred_anomaly_score_{self.model_name}"] = conf
                sample[f"pred_anomaly_{self.model_name}"] = fo.Classification(
                    label=anomaly
                )
                sample[f"pred_anomaly_map_{self.model_name}"] = fo.Heatmap(
                    map=output["anomaly_maps"].numpy()
                )
                sample[f"pred_defect_mask_{self.model_name}"] = fo.Segmentation(
                    mask=output["pred_masks"].numpy()
                )

        else:
            inferencer = TorchInferencer(
                path=os.path.join(self.model_path), device="cuda"
            )
            self.inferencer = inferencer

            for sample in self.abnormal_data.iter_samples(autosave=True, progress=True):

                image = read_image(sample.filepath, as_tensor=True)
                output = self.inferencer.predict(image)
                conf = output.pred_score
                anomaly = "normal" if conf < threshold else "anomaly"

                sample[f"pred_anomaly_score_{self.model_name}"] = conf
                sample[f"pred_anomaly_{self.model_name}"] = fo.Classification(
                    label=anomaly
                )
                sample[f"pred_anomaly_map_{self.model_name}"] = fo.Heatmap(
                    map=output.anomaly_map
                )
                sample[f"pred_defect_mask_{self.model_name}"] = fo.Segmentation(
                    mask=output.pred_mask
                )

    def eval_v51(self):
        eval_seg = self.abnormal_data.evaluate_segmentations(
            f"pred_defect_mask_{self.model_name}",
            gt_field="anomaly_mask",
            eval_key=f"eval_seg_{self.model_name}",
        )
        eval_seg.print_report(classes=[0, 255])
