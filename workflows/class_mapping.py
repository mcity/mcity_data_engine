import logging
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForZeroShotImageClassification, AutoModel, AlignProcessor, AlignModel, AltCLIPModel, AltCLIPProcessor, CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers import SiglipModel, SiglipProcessor
import os
import datetime
from PIL import Image
from tqdm import tqdm
from config.config import WORKFLOWS
import fiftyone as fo
import wandb
from utils.dataset_loader import load_dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class ClassMapper:
    def __init__(self, dataset, model_name, config=None):
        """
        Initialize the ClassMapper with dataset and model configuration.

        Args:
            dataset: FiftyOne dataset containing images and detections.
            model_name: Name of the HuggingFace model to use.
            config: Optional override configuration dictionary.
        """
        self.dataset = dataset
        self.model_name = model_name
        # Get default config from WORKFLOWS and update with any provided config.
        self.config = WORKFLOWS["class_mapping"].copy()
        if config:
            self.config.update(config)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.stats = {
            "total_processed": 0,
            "changes_made": 0,
            "parent_class_counts": {},  # Count of processed detections per parent class (e.g., Car, Truck)
            "tags_added_per_category": {}  # Detailed counts for child tags added (e.g., Van, Pickup)
        }

    def load_model(self):
        """Load the model and processor from HuggingFace."""
        try:
            self.hf_model_config = AutoConfig.from_pretrained(self.model_name)
            self.hf_model_config_name = type(self.hf_model_config).__name__

            if self.hf_model_config_name == "SiglipConfig":
                if self.model_name == "google/siglip2-base-patch16-224":
                    self.model = AutoModel.from_pretrained(self.model_name, device_map="auto").eval()
                    self.processor = AutoProcessor.from_pretrained(self.model_name)
                else:
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.processor = AutoProcessor.from_pretrained(self.model_name)

            elif self.hf_model_config_name == "AlignConfig":
                self.processor = AlignProcessor.from_pretrained(self.model_name)
                self.model = AlignModel.from_pretrained(self.model_name)

            elif self.hf_model_config_name == "AltCLIPConfig":
                self.processor = AltCLIPProcessor.from_pretrained(self.model_name)
                self.model = AltCLIPModel.from_pretrained(self.model_name)

            elif self.hf_model_config_name == "CLIPSegConfig":
                self.processor = CLIPSegProcessor.from_pretrained(self.model_name)
                self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name)

            elif self.hf_model_config_name in ["Blip2Config", "CLIPConfig"]:
                self.model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_name)
                self.processor = AutoProcessor.from_pretrained(self.model_name)

            else:
                logging.error(f"Invalid Model Name : {self.model_name}")

            self.model.to(self.device)
            logging.info(f"Successfully loaded model {self.model_name}")

        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

    def process_detection(self, image, detection, candidate_labels):
        """Process a single detection with the model."""
        # Convert bounding box to pixel coordinates.
        prob_threshold = self.config["thresholds"]["confidence"]

        img_width, img_height = image.size
        bbox = detection.bounding_box
        min_x, min_y, width, height = bbox
        x1, y1 = int(min_x * img_width), int(min_y * img_height)
        x2, y2 = int((min_x + width) * img_width), int((min_y + height) * img_height)

        # Crop image to detection region.
        image_patch = image.crop((x1, y1, x2, y2))

        # Prepare inputs for the model.
        if self.hf_model_config_name == "SiglipConfig":
            if self.model_name == "google/siglip2-base-patch16-224":
                inputs = self.processor(text=candidate_labels, images=image_patch, padding="max_length", max_length=64, return_tensors="pt")
            else:
                inputs = self.processor(text=candidate_labels, images=image_patch, padding="max_length", return_tensors="pt")

        elif self.hf_model_config_name == "CLIPSegConfig":
            inputs = self.processor(text=candidate_labels, images=[image_patch]*len(candidate_labels), padding="max_length", return_tensors="pt")

        else:
            inputs = self.processor(images=image_patch, text=candidate_labels, return_tensors="pt", padding=True)

        # Ensure all tensors in the processed inputs are moved to the designated device.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate classification output.
        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_label = None
        confidence_score = None

        if self.hf_model_config_name == "SiglipConfig":
            # Apply sigmoid for probabilities
            logits = outputs.logits_per_image
            probs = torch.sigmoid(logits)
            probs = torch.softmax(probs, dim=1)
            max_prob, predicted_idx = probs[0].max(dim=-1)
            predicted_label = candidate_labels[predicted_idx.item()]
            confidence_score = max_prob.item()

        elif self.hf_model_config_name in ["AlignConfig", "AltCLIPConfig"]:
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=1)
            max_prob, predicted_idx = probs[0].max(dim=-1)
            predicted_label = candidate_labels[predicted_idx.item()]
            confidence_score = max_prob.item()

        elif self.hf_model_config_name == "CLIPSegConfig":
            # Get masks and ensure batch dimension exists
            masks = torch.sigmoid(outputs.logits)
            if masks.dim() == 2:  # Handle single-example edge case
                masks = masks.unsqueeze(0)  # Add batch dimension

            # Verify mask dimensions
            batch_size, mask_height, mask_width = masks.shape

            # Convert detection box to mask coordinates
            box_x1 = int(x1 * mask_width / img_width)
            box_y1 = int(y1 * mask_height / img_height)
            box_x2 = int(x2 * mask_width / img_width)
            box_y2 = int(y2 * mask_height / img_height)

            # Calculate scores for each candidate label
            scores = []
            for i in range(batch_size):
                # Extract relevant mask region
                label_mask = masks[i, box_y1:box_y2, box_x1:box_x2]

                # Handle empty regions gracefully
                if label_mask.numel() == 0:
                    scores.append(0.0)
                    continue
                scores.append(label_mask.mean().item())

            # Find best match
            confidence_score = max(scores)
            predicted_idx = scores.index(confidence_score)
            predicted_label = candidate_labels[predicted_idx]

        else:
            logits = outputs.logits_per_image
            max_logit, predicted_idx = logits.max(dim=-1)
            predicted_label = candidate_labels[predicted_idx.item()]
            confidence_score = max_logit.item()

        return predicted_label, confidence_score

    def run_mapping(self, label_field = "ground_truth"):
        """
        Run the class mapping process on the dataset.
        """
        if not self.model:
            self.load_model()

        dataset_source_name = self.config.get("dataset_source")
        dataset_target_name = self.config.get("dataset_target")

        if not dataset_source_name or not dataset_target_name:
            error_msg = "Both 'dataset_source' and 'dataset_target' must be specified in the config."
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Load the datasets from FiftyOne.
        try:
            SELECTED_DATASET = {
                "name": dataset_source_name,
                "n_samples": None,  # 'None' (full dataset) or 'int' (subset of the dataset)
            }
            source_dataset, source_dataset_info = load_dataset(SELECTED_DATASET)

        except Exception as e:
            error_msg = f"Failed to load dataset_source '{dataset_source_name}': {e}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        try:
            SELECTED_DATASET = {
                "name": dataset_target_name,
                "n_samples": None,  # 'None' (full dataset) or 'int' (subset of the dataset)
            }
            target_dataset, target_dataset_info = load_dataset(SELECTED_DATASET)
            #target_dataset = fo.load_dataset(dataset_target_name)
        except Exception as e:
            error_msg = f"Failed to load dataset_target '{dataset_target_name}': {e}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Get the distinct labels present in each dataset.
        source_labels = source_dataset.distinct(f"{label_field}.detections.label")
        target_labels = target_dataset.distinct(f"{label_field}.detections.label")

        # Access candidate labels from config
        candidate_labels = self.config["candidate_labels"]

        # Check that all parent labels from the candidate_labels exist in dataset_source.
        parent_labels = list(candidate_labels.keys())
        missing_parents = [p for p in parent_labels if p not in source_labels]
        if missing_parents:
            error_msg = (f"Missing parent labels in dataset_source '{dataset_source_name}': {missing_parents}\n"
                        f"Expected parents: {parent_labels}\n"
                        f"Found in dataset_source: {source_labels}")
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Check that all child labels from candidate_labels exist in dataset_target.
        all_child_labels = []
        for children in candidate_labels.values():
            all_child_labels.extend(children)
        missing_children = [child for child in all_child_labels if child not in target_labels]
        if missing_children:
            error_msg = (f"Missing child labels in dataset_target '{dataset_target_name}': {missing_children}\n"
                        f"Expected children: {all_child_labels}\n"
                        f"Found in dataset_target: {target_labels}")
            logging.error(error_msg)
            raise ValueError(error_msg)

        one_to_one_mapping = {
            parent: children[0]  # Only pick mappings with exactly 1 child
            for parent, children in candidate_labels.items()
            if len(children) == 1
        }

        experiment_name = f"class_mapping_{os.getpid()}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        tensorboard_root = self.config.get("tensorboard_root", "./tensorboard_logs")
        dataset_name = getattr(self.dataset, "name", "default_dataset")
        log_directory = os.path.join(tensorboard_root, dataset_name, experiment_name)

        # Unpatch any previous TensorBoard patching before applying a new one
        try:
            wandb.tensorboard.unpatch()
        except Exception as e:
            logging.warning(f"Error unpatching TensorBoard: {e}")

        wandb.tensorboard.patch(root_logdir=log_directory)
        wandb.init(
            name=f"class_mapping_{os.getpid()}",
            project="Class_mapping",
            config=self.config,
            sync_tensorboard=False  # Disable automatic syncing to avoid conflicts
        )
        tb_writer = SummaryWriter(log_dir=log_directory)

        threshold = self.config["thresholds"]["confidence"]
        sample_count = 0  # For logging steps

        for sample in self.dataset.iter_samples(progress=True, autosave=True):
            sample_count += 1
            try:
                image = Image.open(sample.filepath)
            except Exception as e:
                logging.error(f"Error opening image {sample.filepath}: {str(e)}")
                continue

            detections = sample[label_field]
            for det in detections.detections:
                current_label = det.label

                # Check if it is a one-to-one mapping
                if current_label in one_to_one_mapping:
                    new_label = one_to_one_mapping[current_label]
                    tag = f"new_class_{new_label}"

                    # Apply new label and add tag
                    if tag not in det.tags:
                        det.tags.append(tag)
                        det.label = new_label  # Replace label directly

                        self.stats["tags_added_per_category"][new_label] = (
                            self.stats["tags_added_per_category"].get(new_label, 0) + 1
                        )
                        # Log tag changes
                        tb_writer.add_scalar(f"Tags_Added/{new_label}",
                                            self.stats["tags_added_per_category"].get(new_label, 0),
                                            sample_count)
                    continue

                if current_label not in candidate_labels:
                    continue

                label_candidates = candidate_labels[current_label]
                predicted_label, confidence = self.process_detection(image, det, label_candidates)

                self.stats["parent_class_counts"][current_label] = self.stats["parent_class_counts"].get(current_label, 0) + 1
                self.stats["total_processed"] += 1

                # If prediction meets threshold and differs from the original label, add a tag.
                if confidence > threshold and current_label.lower() != predicted_label.lower():
                    tag = f"new_class_{self.model_name}_{predicted_label}"
                    if tag not in det.tags:
                        det.tags.append(tag)
                        self.stats["changes_made"] += 1
                        if predicted_label != current_label:
                            self.stats["tags_added_per_category"][predicted_label] = (
                                self.stats["tags_added_per_category"].get(predicted_label, 0) + 1
                            )

                        if tb_writer is not None:
                            tb_writer.add_scalar(f"{self.model_name}/Tags_Added/{predicted_label}",
                                                self.stats["tags_added_per_category"].get(predicted_label, 0),
                                                sample_count)
                            tb_writer.add_scalar(f"{self.model_name}/Total_Tags_Added",
                                                self.stats["changes_made"],
                                                sample_count)

            if tb_writer is not None:
                tb_writer.add_scalar(f"{self.model_name}/Total_Processed_Samples",
                                    self.stats["total_processed"],
                                    sample_count)
                tb_writer.add_scalar(f"{self.model_name}/Total_Tag_Changes",
                                    self.stats["changes_made"],
                                    sample_count)

                # Log parent class counts and child tag counts based on candidate_labels.
                for parent_label, child_labels in self.config["candidate_labels"].items():
                    tb_writer.add_scalar(f"{self.model_name}/Parent_Class_Count/{parent_label}",
                                        self.stats["parent_class_counts"].get(parent_label, 0),
                                        sample_count)
                    for child_label in child_labels:
                        tb_writer.add_scalar(f"{self.model_name}/Tags_Added/{child_label}",
                                            self.stats["tags_added_per_category"].get(child_label, 0),
                                            sample_count)


        if tb_writer is not None:
            tb_writer.close()
        return self.stats
