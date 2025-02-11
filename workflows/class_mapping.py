# class_mapping.py
import logging
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
from tqdm import tqdm
from config.config import WORKFLOWS

class ClassMapper:
    def __init__(self, dataset, model_name, config=None):
        """
        Initialize the ClassMapper with dataset and model configuration.

        Args:
            dataset: FiftyOne dataset containing images and detections
            model_name: Name of the HuggingFace model to use
            config: Optional override configuration dictionary
        """
        self.dataset = dataset
        self.model_name = model_name
        # Get default config from WORKFLOWS and update with any provided config
        self.config = WORKFLOWS["class_mapping"].copy()
        if config:
            self.config.update(config)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.stats = {
            "total_processed": 0,
            "changes_made": 0,
            "class_counts": {}
        }

    def load_model(self):
        """Load the model and processor from HuggingFace."""
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            logging.info(f"Successfully loaded model {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

    def process_detection(self, image, detection, candidate_labels):
        """Process a single detection with the model."""
        # Convert bounding box to pixel coordinates
        img_width, img_height = image.size
        bbox = detection.bounding_box
        min_x, min_y, width, height = bbox
        x1, y1 = int(min_x * img_width), int(min_y * img_height)
        x2, y2 = int((min_x + width) * img_width), int((min_y + height) * img_height)

        # Crop image to detection region
        vehicle_patch = image.crop((x1, y1, x2, y2))

        # Prepare inputs for the model
        inputs = self.processor(
            images=vehicle_patch,
            text=candidate_labels,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate classification output
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image

        max_logit, predicted_idx = logits.max(dim=-1)
        predicted_label = candidate_labels[predicted_idx.item()]

        return predicted_label, max_logit.item()

    def run_mapping(self):
        """Run the class mapping process on the dataset."""
        if not self.model:
            self.load_model()

        candidate_labels = self.config["candidate_labels"]
        threshold = self.config["thresholds"]["confidence"]

        for sample in self.dataset.iter_samples(progress=True, autosave=True):
            image = Image.open(sample.filepath)
            detections = sample["ground_truth"]

            for det in detections.detections:
                old_label = det.label
                if old_label not in candidate_labels:
                    continue

                label_candidates = candidate_labels[old_label]
                predicted_label, confidence = self.process_detection(
                    image, det, label_candidates
                )

                if confidence > threshold and old_label != predicted_label:
                    tag = f"new_class_{self.model_name}_{predicted_label}"
                    if tag not in det.tags:
                        det.tags.append(tag)
                        self.stats["changes_made"] += 1

                self.stats["class_counts"][predicted_label] = \
                    self.stats["class_counts"].get(predicted_label, 0) + 1
                self.stats["total_processed"] += 1

        return self.stats