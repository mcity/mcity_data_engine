import logging
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, AutoModel, AlignProcessor, AlignModel, AltCLIPModel, AltCLIPProcessor, CLIPSegProcessor, CLIPSegForImageSegmentation
import os
import datetime
from PIL import Image
from tqdm import tqdm
from config.config import WORKFLOWS
import wandb
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
            self.is_siglip = "siglip" in self.model_name
            self.is_align = "align" in self.model_name
            self.is_altclip = "AltCLIP" in self.model_name
            self.is_clipseg = "clipseg" in self.model_name

            if self.is_siglip:
                self.model = AutoModel.from_pretrained(self.model_name)
                self.processor = AutoProcessor.from_pretrained(self.model_name)

            elif self.is_align:
                self.processor = AlignProcessor.from_pretrained(self.model_name)
                self.model = AlignModel.from_pretrained(self.model_name)

            elif self.is_altclip:
                self.processor = AltCLIPProcessor.from_pretrained(self.model_name)
                self.model = AltCLIPModel.from_pretrained(self.model_name)

            elif self.is_clipseg:
                self.processor = CLIPSegProcessor.from_pretrained(self.model_name)
                self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name)

            else:
                self.model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_name)
                self.processor = AutoProcessor.from_pretrained(self.model_name)

            self.model.to(self.device)

            logging.info(f"Successfully loaded model {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise

    def process_detection(self, image, detection, candidate_labels):
        """Process a single detection with the model."""
        # Convert bounding box to pixel coordinates.
        prob_threshold = 0.2
        img_width, img_height = image.size
        bbox = detection.bounding_box
        min_x, min_y, width, height = bbox
        x1, y1 = int(min_x * img_width), int(min_y * img_height)
        x2, y2 = int((min_x + width) * img_width), int((min_y + height) * img_height)

        # Crop image to detection region.
        vehicle_patch = image.crop((x1, y1, x2, y2))

        # Prepare inputs for the model.
        if self.is_siglip:
            inputs = self.processor(text=candidate_labels, images=vehicle_patch, padding="max_length", return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        elif self.is_clipseg:
            inputs = self.processor(text=candidate_labels, images=[vehicle_patch]*len(candidate_labels), padding="max_length", return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        elif self.is_align:
            inputs = self.processor(images=vehicle_patch, text=candidate_labels, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        else:
            inputs = self.processor(images=vehicle_patch, text=candidate_labels, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}


        # Generate classification output.
        with torch.no_grad():
            outputs = self.model(**inputs)


        if self.is_siglip:
            # Apply sigmoid for probabilities
            logits = outputs.logits_per_image
            probs = torch.sigmoid(logits)
            max_prob, predicted_idx = probs[0].max(dim=-1)
            predicted_label = candidate_labels[predicted_idx.item()]
            return predicted_label, max_prob.item()

        elif self.is_align or self.is_altclip:
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=1)
            max_prob, predicted_idx = probs[0].max(dim=-1)
            predicted_label = candidate_labels[predicted_idx.item()]
            return predicted_label, max_prob.item()

        elif self.is_clipseg:
            with torch.no_grad():
                outputs = self.model(**inputs)

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
            max_score = max(scores)
            predicted_idx = scores.index(max_score)
            return candidate_labels[predicted_idx], max_score


        else:
            logits = outputs.logits_per_image
            max_logit, predicted_idx = logits.max(dim=-1)
            predicted_label = candidate_labels[predicted_idx.item()]
            return predicted_label, max_logit.item()



    def run_mapping(self, interactive=True, wandb_logging=False):
        """
        Run the class mapping process on the dataset.

        If 'interactive' is True, a PrintLogger is used to print tag additions in real time.
        If 'wandb_logging' is True, metrics will be logged to Weights & Biases with TensorBoard syncing.
        """
        if not self.model:
            self.load_model()

        if wandb_logging:
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
                #job_type="inference",
                project="Class_mapping",
                config=self.config,
                sync_tensorboard=False  # Disable automatic syncing to avoid conflicts
            )
            tb_writer = SummaryWriter(log_dir=log_directory)
        else:
            tb_writer = None

        # Validate parent labels
        parent_labels = list(self.config["candidate_labels"].keys())
        try:
            existing_labels = self.dataset.distinct("ground_truth.detections.label")
        except Exception as e:
            logging.error("Could not retrieve detection labels from dataset")
            raise ValueError("Invalid detection field path") from e

        missing_parents = [label for label in parent_labels if label not in existing_labels]
        if len(missing_parents) == len(parent_labels):
            error_msg = (f"No parent labels from config found in dataset!\n"
                         f"Config parents: {parent_labels}\nDataset labels: {existing_labels}")
            logging.error(error_msg)
            raise ValueError("Parent label mismatch")
        if missing_parents:
            logging.warning(f"Some parent labels not found in dataset: {missing_parents}")

        candidate_labels = self.config["candidate_labels"]
        threshold = self.config["thresholds"]["confidence"]

        # Set up interactive logging if enabled
        if interactive:
            class PrintLogger:
                def __init__(self):
                    self.count = 0

                def log(self, tag, confidence):
                    self.count += 1
                    print(f"Tag added: {tag} (Confidence: {confidence:.2f})")
            print_logger = PrintLogger()
        else:
            print_logger = None

        sample_count = 0  # For logging steps

        for sample in tqdm(self.dataset.iter_samples(progress=True, autosave=True), desc="Processing samples"):
            sample_count += 1
            try:
                image = Image.open(sample.filepath)
            except Exception as e:
                logging.error(f"Error opening image {sample.filepath}: {str(e)}")
                continue

            detections = sample["ground_truth"]
            for det in detections.detections:
                old_label = det.label
                if old_label not in candidate_labels:
                    continue

                label_candidates = candidate_labels[old_label]
                predicted_label, confidence = self.process_detection(image, det, label_candidates)

                self.stats["parent_class_counts"][old_label] = self.stats["parent_class_counts"].get(old_label, 0) + 1
                self.stats["total_processed"] += 1

                # If prediction meets threshold and differs from the original label, add a tag.
                if confidence > threshold and old_label != predicted_label:
                    tag = f"new_class_{self.model_name}_{predicted_label}"
                    if tag not in det.tags:
                        det.tags.append(tag)
                        self.stats["changes_made"] += 1
                        if predicted_label != old_label:
                            self.stats["tags_added_per_category"][predicted_label] = (
                                self.stats["tags_added_per_category"].get(predicted_label, 0) + 1
                            )

                        if print_logger:
                            print_logger.log(tag, confidence)

                        if wandb_logging:
                            wandb.log({
                                f"{self.model_name}/Tags_Added/{predicted_label}": self.stats["tags_added_per_category"].get(predicted_label, 0),
                                f"{self.model_name}/Total_Tags_Added": self.stats["changes_made"]
                            })

            if wandb_logging:
                wandb.log({
                    f"{self.model_name}/Total_Processed_Samples": self.stats["total_processed"],
                    f"{self.model_name}/Total_Tag_Changes": self.stats["changes_made"],
                    f"{self.model_name}/Parent_Class_Count/Car": self.stats["parent_class_counts"].get("Car", 0),
                    f"{self.model_name}/Parent_Class_Count/Truck": self.stats["parent_class_counts"].get("Truck", 0),
                    f"{self.model_name}/Tags_Added/Van": self.stats["tags_added_per_category"].get("Van", 0),
                    f"{self.model_name}/Tags_Added/Pickup": self.stats["tags_added_per_category"].get("Pickup", 0)
                }, step=sample_count)

            if tb_writer is not None:
                tb_writer.add_scalar('Metrics/Total_Processed_Detections', self.stats["total_processed"], sample_count)
                tb_writer.add_scalar('Metrics/Total_Tag_Changes', self.stats["changes_made"], sample_count)
                tb_writer.add_scalar('Parent_Class_Count/Car', self.stats["parent_class_counts"].get("Car", 0), sample_count)
                tb_writer.add_scalar('Parent_Class_Count/Truck', self.stats["parent_class_counts"].get("Truck", 0), sample_count)
                tb_writer.add_scalar('Tags_Added/Van', self.stats["tags_added_per_category"].get("Van", 0), sample_count)
                tb_writer.add_scalar('Tags_Added/Pickup', self.stats["tags_added_per_category"].get("Pickup", 0), sample_count)

        if tb_writer is not None:
            tb_writer.close()
        if wandb_logging:
            wandb.finish(exit_code=0)
        return self.stats
