import logging
import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class AutoLabelMask:
    def __init__(self, dataset, dataset_info, model_name, task_type, model_config):
        self.dataset = dataset
        self.dataset_info = dataset_info
        self.model_name = model_name
        self.task_type = task_type
        self.model_config = model_config

    def _sanitize_model_name(self, model_name: str) -> str:
        return (
            model_name
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )

    def run_inference(self):
        if self.task_type == "semantic_segmentation":
            self._run_semantic_segmentation()
        elif self.task_type == "depth_estimation":
            self._run_depth_estimation()
        else:
            logging.error(f"Task type '{self.task_type}' is not supported")

    # ---------------------------
    # SEMANTIC SEGMENTATION
    # ---------------------------
    def _run_semantic_segmentation(self):
        if self.model_name == "sam2":
            self._handle_sam2_segmentation()
        else:
            logging.error(
                f"Semantic segmentation with model '{self.model_name}' is not supported"
            )

    def _handle_sam2_segmentation(self):
        sam_config = self.model_config
        prompt_field = sam_config.get("prompt_field", None)
        sam_models = sam_config["models"]

        for sam_model in sam_models:
            model_name_clear = self._sanitize_model_name(sam_model)
            logging.info(f"Running SAM2 segmentation for model: {sam_model}")

            # `pred_ss_` for semantic segmentation
            if prompt_field:
                label_field = f"pred_ss_{model_name_clear}_prompt_{prompt_field}"
            else:
                label_field = f"pred_ss_{model_name_clear}_noprompt"

            self._inference_sam2(self.dataset, sam_model, label_field, prompt_field)

        logging.info("Semantic segmentation completed for all SAM2 models.")

    def _inference_sam2(self, dataset, sam_model, label_field, prompt_field=None):
        logging.info(f"Starting SAM2 inference for model '{sam_model}'")
        model = foz.load_zoo_model(sam_model)

        if prompt_field:
            logging.info(f"Running SAM2 with prompt_field '{prompt_field}'")
            dataset.apply_model(model, label_field=label_field, prompt_field=prompt_field)
        else:
            logging.info("Running SAM2 without prompt_field")
            dataset.apply_model(model, label_field=label_field)

    # ---------------------------
    # DEPTH ESTIMATION
    # ---------------------------
    def _run_depth_estimation(self):
        if self.model_name not in ["dpt", "depth_anything", "glpn", "zoe_depth"]:
            logging.error(f"Depth estimation model '{self.model_name}' not supported")
            return

        depth_models = self.model_config["models"]

        for depth_model in depth_models:
            depth_model_clear = self._sanitize_model_name(depth_model)
            logging.info(f"Running depth estimation for model: {depth_model}")

            # `pred_de_` for depth estimatoin
            label_field = f"pred_de_{depth_model_clear}"

            self._inference_depth_estimation(self.dataset, depth_model, label_field)

        logging.info(f"Depth estimation completed for all '{self.model_name}' models.")

    def _inference_depth_estimation(self, dataset, model_name, label_field):
        logging.info(f"Starting depth estimation for HF model '{model_name}'")
        image_processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True
        )

        def apply_depth_model(sample, depth_model, processor, out_field):

            pil_image = Image.open(sample.filepath).convert("RGB")
            depth_inputs = processor(images=pil_image, return_tensors="pt")

            with torch.no_grad():
                depth_outputs = depth_model(**depth_inputs)
                predicted_depth = depth_outputs.predicted_depth

            resized_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=pil_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            depth_map = resized_depth.squeeze().cpu().numpy()

            inverted_depth_map = (255 - depth_map * 255 / np.max(depth_map)).astype("uint8")

            sample[out_field] = fo.Heatmap(map=inverted_depth_map)
            sample.save()

        for sample in dataset.iter_samples(autosave=True, progress=True):
            apply_depth_model(sample, model, image_processor, label_field)

        logging.info(f"Depth estimation inference finished for '{model_name}'")
