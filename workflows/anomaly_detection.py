import logging
import os

import anomalib.models
import fiftyone as fo
import torch
from anomalib import TaskType
from anomalib.data.image.folder import Folder
from anomalib.data.utils import read_image
from anomalib.deploy import ExportType, TorchInferencer
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger
from fiftyone import ViewField as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.transforms.v2 import Compose, Resize

from config.config import GLOBAL_SEED, NUM_WORKERS

# https://docs.voxel51.com/tutorials/anomaly_detection.html
# https://medium.com/@enrico.randellini/anomalib-a-library-for-image-anomaly-detection-and-localization-fb363639104f
# https://github.com/openvinotoolkit/anomalib
# https://anomalib.readthedocs.io/en/v1.1.1/


class Anodec:
    def __init__(
        self,
        dataset,
        eval_metrics,
        dataset_info,
        config,
        tensorboard_output,
        anomalib_output_root="./output/models/anomalib/",
    ):
        torch.set_float32_matmul_precision(
            "medium"
        )  # Utilize Tensor core, came in warning
        self.config = config
        self.dataset = dataset
        self.eval_metrics = eval_metrics
        self.normal_data = dataset.match_tags("train")
        logging.info(f"Using {len(self.normal_data)} normal images for training.")
        self.abnormal_data = dataset.match_tags(["val", "test"])
        logging.info(
            f"Using {len(self.abnormal_data)} images with anomalies for evaluating."
        )
        self.brains = dataset.list_brain_runs()
        self.dataset_name = dataset_info["name"]
        self.TASK = TaskType.SEGMENTATION
        self.model_name = self.config["model_name"]
        self.image_size = self.config["image_size"]
        self.batch_size = self.config["batch_size"]
        self.tensorboard_output = tensorboard_output
        self.anomalib_output_root = anomalib_output_root
        self.model_path = os.path.join(
            anomalib_output_root,
            self.model_name,
            self.dataset_name,
            "weights/torch/model.pt",
        )

        data_root = self.config["data_root"]
        dataset_folder_ano_dec_masks = f"{self.dataset_name}_anomaly_detection_masks/"
        filepath_masks = os.path.join(data_root, dataset_folder_ano_dec_masks)
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
        """
        Create and setup a data module for anomaly detection.

        This method performs the following steps:
        1. Builds a transform if none is provided.
        2. Creates subsets of data containing only the “good” training images and “anomalous” images for validation.
        3. Symlinks the images and masks to the directory Anomalib expects.
        4. Instantiates and sets up a datamodule from Anomalib’s Folder class, which is the general-purpose class for custom datasets.

        Args:
            transform (callable, optional): A transformation to apply to the images. Defaults to resizing to IMAGE_SIZE with antialiasing.

        Returns:
            None
        """
        if transform is None:
            transform = Compose([Resize(self.image_size, antialias=True)])

        # Symlink the images and masks to the directory Anomalib expects.
        for sample in self.abnormal_data.iter_samples(progress=True, autosave=True):
            # Add mask groundtruth
            base_filename = sample.filename
            mask_filename = os.path.basename(base_filename).replace(".jpg", ".png")
            mask_path = os.path.join(self.mask_dir, mask_filename)
            logging.debug(f"Assigned mask {mask_path} to sample {base_filename}")

            if not os.path.exists(mask_path):
                logging.error(f"Mask file not found: {mask_path}")

            sample["anomaly_mask"] = fo.Segmentation(mask_path=mask_path)

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
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
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
        """
        Trains an anomaly detection model using Anomalib’s Engine class, exports the model,
        and returns the model “inferencer” object. The inferencer object is used to make predictions on new images.

        Args:
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
                                            E.g, ``transforms.RandomCrop`` for images.

        Returns:
            None
        """

        MAX_EPOCHS = self.config["epochs"]
        PATIENCE = self.config["early_stop_patience"]

        self.create_datamodule(transform=transform)
        if not os.path.exists(self.model_path):
            self.model = getattr(anomalib.models, self.model_name)()

            os.makedirs(self.anomalib_output_root, exist_ok=True)
            os.makedirs(self.tensorboard_output, exist_ok=True)
            self.unlink_symlinks()
            self.anomalib_logger = AnomalibTensorBoardLogger(
                save_dir=self.tensorboard_output,
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
                # image_metrics=self.eval_metrics, #Classification for whole image
                pixel_metrics=self.eval_metrics,
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
        else:
            logging.info(f"Model {self.model_path} already trained.")

    def validate_model(self):
        """
        Validates the model using the engine's test method.

        This method tests the model if the engine is available. It logs the test results
        using the `logging` module. If the engine is not available, it currently does nothing
        but has a TODO comment indicating that the model should be loaded from a path similar
        to inference mode.

        Returns:
            None
        """
        # Test model
        if self.engine:
            test_results = self.engine.test(
                model=self.model,
                datamodule=self.datamodule,
                ckpt_path=self.engine.trainer.checkpoint_callback.best_model_path,
            )
            logging.info(f"Model test results: {test_results}")
        else:
            logging.error(f"Engine '{self.engine}' not available.")

    def run_inference(self, threshold=0.5):
        """
        Runs inference on a collection of samples to detect anomalies.

        Parameters:
        threshold (float): The cutoff value for the anomaly score. Samples with a score above this threshold
                           are considered anomalous. Default is 0.5.

        Description:
        This method takes a FiftyOne sample collection (e.g., a test set) as input, along with an inferencer object,
        and a key for storing the results in the samples. It runs the model on each sample in the collection and stores
        the results. The threshold argument acts as a cutoff for the anomaly score. If the score is above the threshold,
        the sample is considered anomalous.

        If the engine is available, it uses the engine to predict the outputs. For each sample, it checks if the sample
        and output are aligned. If they are not aligned, it logs an error and continues to the next sample. It then
        calculates the confidence score and determines if the sample is anomalous based on the threshold. The results
        are stored in the sample with keys for anomaly score, anomaly classification, anomaly map, and defect mask.

        If the engine is not available, it initializes a TorchInferencer and uses it to predict the outputs for each sample.
        It reads the image from the sample's filepath, predicts the output, calculates the confidence score, and determines
        if the sample is anomalous based on the threshold. The results are stored in the sample with keys for anomaly score,
        anomaly classification, anomaly map, and defect mask.
        """
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
                    logging.error(
                        f"Sample {sample.filepath} and output {output["image_path"][0]} are not aligned!"
                    )
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
                sample.save()

        else:
            inferencer = TorchInferencer(
                path=os.path.join(self.model_path), device="cuda"
            )
            self.inferencer = inferencer

            for sample in self.abnormal_data.iter_samples(autosave=True, progress=True):

                image = read_image(sample.filepath, as_tensor=True)
                output = self.inferencer.predict(image)

                # Classification
                conf = output.pred_score
                anomaly = "normal" if conf < threshold else "anomaly"
                sample[f"pred_anomaly_score_{self.model_name}"] = conf
                sample[f"pred_anomaly_{self.model_name}"] = fo.Classification(
                    label=anomaly
                )

                # Segmentation
                sample[f"pred_anomaly_map_{self.model_name}"] = fo.Heatmap(
                    map=output.anomaly_map
                )
                sample[f"pred_defect_mask_{self.model_name}"] = fo.Segmentation(
                    mask=output.pred_mask
                )
                sample.save()

    def eval_v51(self):
        """
        Evaluates the segmentations of abnormal data using the specified model.

        This method evaluates the segmentations of abnormal data by comparing the predicted defect mask
        with the ground truth anomaly mask. The evaluation results are stored and a report is printed.

        Parameters:
        None

        Returns:
        None

        Side Effects:
        - Evaluates the segmentations and stores the results in the `eval_seg_{self.model_name}` key.
        - Prints a report of the evaluation results for the specified classes [0, 255].
        """
        eval_seg = self.abnormal_data.evaluate_segmentations(
            f"pred_defect_mask_{self.model_name}",
            gt_field="anomaly_mask",
            eval_key=f"eval_seg_{self.model_name}",
        )
        eval_seg.print_report(classes=[0, 255])
