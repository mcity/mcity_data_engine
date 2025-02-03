import fiftyone.zoo as foz
import fiftyone as fo
import logging
import numpy as np
from PIL import Image
import torch
import os 
from sam2.sam2_image_predictor import SAM2ImagePredictor
from config.config import WORKFLOWS

# from depth_pro.depth_pro import create_model_and_transforms, DepthProConfig, load_rgb

class MaskTeacher:
    def __init__(self, dataset, dataset_info, model_name, task_type, model_config):
        self.dataset = dataset
        self.dataset_info = dataset_info
        self.model_name = model_name
        self.task_type = task_type
        self.model_config = model_config
        
    def run_inference(self):
        if self.task_type == "semantic_segmentation":
            self._run_semantic_segmentation()
        elif self.task_type == "depth_estimation":
            self._run_depth_estimation()
        else:
            logging.error(f"Task type {self.task_type} not supported")

    def inference_sam2(self, dataset, model_name, label_field, prompt_field = None):
        logging.info(f"Starting SAM2 inference for model {model_name}")
        model = foz.load_zoo_model(model_name)

        if prompt_field:
            logging.info(f"Running SAM2 with prompt field {prompt_field}")
            # Masks for the given bounding boxes with semantic classes
            dataset.apply_model(
                model,
                label_field=label_field,
                prompt_field=prompt_field,
            )
        else:
            logging.info(f"Running SAM2 without prompt field")
            # Mask for whole image but without semantic classes
            dataset.apply_model(
                model,
                label_field=label_field,
            )

    def _run_semantic_segmentation(self):
        if self.model_name == "sam2":
            logging.info(f"Loading SAM2 model")

            prompt_field = self.model_config["prompt_field"]
            sam_models = self.model_config["models"]

            for sam_model in sam_models:
                model_name_clear = sam_model.replace("-", "_")
                logging.info(f"Running semantic segmentation for model {sam_model}")

                if prompt_field:
                    label_field_bbox_prompt = f"pred_{model_name_clear}_prompt_field_{prompt_field}"
                else:
                    label_field_bbox_prompt = f"pred_{model_name_clear}_no_prompt_field"

                self.inference_sam2(self.dataset, sam_model, label_field_bbox_prompt, prompt_field)

        logging.info("Semantic segmentation completed for all models.")

    def _run_depth_estimation(self):
        pass
        # try:
        #     logging.info("Initializing DepthPro model")
        #     checkpoint_path = "/home/wizard/ml-depth-pro/checkpoints/depth_pro.pt"
        #     depth_map_save_dir = "depth_maps"
        #     os.makedirs(depth_map_save_dir, exist_ok=True)

        #     # Initialize DepthPro
        #     depthpro_config = DepthProConfig(
        #         checkpoint_uri=checkpoint_path,
        #         decoder_features=256,
        #         fov_encoder_preset="dinov2l16_384",
        #         image_encoder_preset="dinov2l16_384",
        #         patch_encoder_preset="dinov2l16_384",
        #         use_fov_head=True,
        #     )

        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     depth_model, transform = create_model_and_transforms(
        #         config=depthpro_config,
        #         device=device,
        #         precision=torch.float32,
        #     )
        #     depth_model.eval()

        #     for sample in self.dataset.iter_samples(progress=True):
        #         try:
        #             image_path = sample.filepath
        #             image, _, f_px = load_rgb(image_path)
        #             transformed_image = transform(image)

        #             with torch.no_grad():
        #                 prediction = depth_model.infer(transformed_image, f_px=f_px)
        #                 depth = prediction["depth"]

        #             depth_normalized = np.interp(
        #                 depth.cpu().numpy(),
        #                 (depth.min().cpu().numpy(), depth.max().cpu().numpy()),
        #                 (0, 255)
        #             ).astype(np.uint8)

        #             depth_map_path = os.path.join(
        #                 depth_map_save_dir, 
        #                 f"{os.path.basename(image_path)}_depth.png"
        #             )
        #             depth_image = Image.fromarray(depth_normalized)
        #             depth_image.save(depth_map_path)

        #             sample["depth_map_path"] = depth_map_path
        #             sample.save()

        #         except Exception as e:
        #             logging.error(f"Error processing depth for {image_path}: {e}")
        #             continue

        # except Exception as e:
        #     logging.error(f"Error in depth estimation setup: {e}")