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
from huggingface_hub import HfApi, hf_hub_download
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.transforms.v2 import Compose, Resize

from config.config import GLOBAL_SEED, HF_DO_UPLOAD, HF_ROOT, NUM_WORKERS

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
            f"Using {len(self.abnormal_data)} images with anomalies for evaluation."
        )
        self.dataset_name = dataset_info["name"]
        self.TASK = TaskType.SEGMENTATION
        self.model_name = self.config["model_name"]
        self.image_size = self.config["image_size"]
        self.batch_size = self.config["batch_size"]
        self.tensorboard_output = os.path.abspath(tensorboard_output)
        self.anomalib_output_root = os.path.abspath(anomalib_output_root)
        self.model_path = os.path.join(
            anomalib_output_root,
            self.model_name,
            self.dataset_name,
            "weights/torch/model.pt",
        )

        data_root = os.path.abspath(self.config["data_root"])
        dataset_folder_ano_dec_masks = f"{self.dataset_name}_anomaly_detection_masks/"
        filepath_masks = os.path.join(data_root, dataset_folder_ano_dec_masks)

        filepath_train = self.normal_data.take(1).first().filepath
        filepath_val = self.abnormal_data.take(1).first().filepath

        self.normal_dir = os.path.dirname(filepath_train)
        self.abnormal_dir = os.path.dirname(filepath_val)
        self.mask_dir = os.path.dirname(filepath_masks)

        self.hf_repo_name = f"{HF_ROOT}/{self.dataset_name}_anomalib_{self.model_name}"

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

    def create_datamodule(self, transform):
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

        # Symlink the images and masks to the directory Anomalib expects.
        logging.info("Preparing images and masks for Anomalib")
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
        for sample in self.abnormal_data.iter_samples(progress=True):
            base_filename = sample.filename
            dir_name = os.path.dirname(sample.filepath).split("/")[-1]
            new_filename = f"{dir_name}_{base_filename}"

            try:
                os.unlink(os.path.join(self.abnormal_dir, new_filename))
            except Exception as e:
                logging.debug(
                    f"Unlinking of {os.path.join(self.abnormal_dir, new_filename)} failed: {e}"
                )

            try:
                os.unlink(os.path.join(self.mask_dir, new_filename))
            except Exception as e:
                logging.debug(
                    f"Unlinking of {os.path.join(self.mask_dir, new_filename)} failed: {e}"
                )

    def train_and_export_model(self):
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

        # Resize image if defined in config
        if self.image_size is not None:
            transform = Compose([Resize(self.image_size, antialias=True)])
        else:
            transform = None

        self.create_datamodule(transform=transform)
        if not os.path.exists(self.model_path):
            self.model = getattr(anomalib.models, self.model_name)()

            os.makedirs(self.anomalib_output_root, exist_ok=True)
            os.makedirs(self.tensorboard_output, exist_ok=True)
            self.unlink_symlinks()
            self.anomalib_logger = AnomalibTensorBoardLogger(
                save_dir=self.tensorboard_output,
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

            # Upload model to Hugging Face
            if HF_DO_UPLOAD == True:
                logging.info(f"Uploading model to Hugging Face: {self.hf_repo_name}")
                api = HfApi()
                api.create_repo(
                    self.hf_repo_name, private=True, repo_type="model", exist_ok=True
                )
                api.upload_file(
                    path_or_fileobj=self.model_path,
                    path_in_repo="model.pt",
                    repo_id=self.hf_repo_name,
                    repo_type="model",
                )

        else:
            logging.warning(
                f"Skipping model {self.model_name}, training results are already in {self.model_path}."
            )

    def validate_model(self):
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
        logging.info(f"Running inference")
        try:
            if os.path.exists(self.model_path):
                file_path = self.model_path
                logging.info(f"Loading model {self.model_name} from disk: {file_path}")
            else:
                download_dir = self.model_path.replace("model.pt", "")
                logging.info(
                    f"Downloading model {self.hf_repo_name} from Hugging Face to {download_dir}"
                )
                file_path = hf_hub_download(
                    repo_id=self.hf_repo_name,
                    filename="model.pt",
                    local_dir=download_dir,
                )
        except Exception as e:
            logging.error(f"Failed to load or download model: {str(e)}.")
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        inferencer = TorchInferencer(path=os.path.join(file_path), device=device)
        self.inferencer = inferencer

        for sample in self.abnormal_data.iter_samples(autosave=True, progress=True):

            image = read_image(sample.filepath, as_tensor=True)
            output = self.inferencer.predict(image)

            # Storing results in Voxel51 dataset
            # Sample Classifiction
            conf = output.pred_score
            anomaly = "normal" if conf < threshold else "anomaly"
            sample[f"pred_anomaly_score_{self.model_name}"] = conf
            sample[f"pred_anomaly_{self.model_name}"] = fo.Classification(label=anomaly)

            # Mask Segmentation
            sample[f"pred_anomaly_map_{self.model_name}"] = fo.Heatmap(
                map=output.anomaly_map
            )
            sample[f"pred_defect_mask_{self.model_name}"] = fo.Segmentation(
                mask=output.pred_mask
            )

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
