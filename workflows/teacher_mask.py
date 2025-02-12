import fiftyone.zoo as foz
import fiftyone as fo
import logging
import numpy as np
from PIL import Image
import torch
import os 
from config.config import WORKFLOWS
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

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

    def inference_depth_estimation(self, dataset, model_name, label_field, prompt_field=None):
        logging.info(f"Starting depth estimation for model {model_name}")

        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name, ignore_mismatched_sizes=True)

        def apply_depth_model(data, model, image_processor, label_field):
            image = Image.open(data.filepath).convert("RGB")  
            inputs = image_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],  
                mode="bicubic",
                align_corners=False,
            )

            output = prediction.squeeze().cpu().numpy()

            formatted = (255 - output * 255 / np.max(output)).astype("uint8")

            data[label_field] = fo.Heatmap(map=formatted)
            data.save()

        for idx, data in enumerate(dataset.iter_samples(autosave=True, progress=True)):
            apply_depth_model(data, model, image_processor, label_field)

        logging.info("Depth estimation completed successfully!")

    def _run_depth_estimation(self):
        if self.model_name in ["dpt", "depth_anything", "glpn", "zoe_depth"]:
            logging.info(f"Loading depth model: {self.model_name}")

            depth_models = self.model_config["models"]
            prompt_field = self.model_config.get("prompt_field", None)  # None if not provided

            for depth_model in depth_models:
                model_name_clear = depth_model.replace("-", "_")
                logging.info(f"Running depth estimation for model {depth_model}")

                label_field_no_prompt = f"pred_{model_name_clear}_no_prompt"
                label_field_with_prompt = f"pred_{model_name_clear}_prompt_{prompt_field}" if prompt_field else None

                if prompt_field:
                    self.inference_depth_estimation(self.dataset, depth_model, label_field_with_prompt, prompt_field)
                else: 
                    self.inference_depth_estimation(self.dataset, depth_model, label_field_no_prompt)

        logging.info("Depth estimation completed for all models.")