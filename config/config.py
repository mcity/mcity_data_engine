import os

SELECTED_WORKFLOW = "learn_normality"  # Selection from WORKFLOWS
SELECTED_DATASET = "mcity_fisheye_ano_ped"  # Choose from datasets.yaml
SELECTED_SPLITS = ["train", "val"]  # Choose from datasets.yaml

PERSISTENT = False  # If V51 database is stored

# Select from V51 embedding model zoo
V51_EMBEDDING_MODELS = (
    "clip-vit-base32-torch",
    "open-clip-torch",
    "dinov2-vitl14-torch",
    "mobilenet-v2-imagenet-torch",
    "resnet152-imagenet-torch",
    "vgg19-imagenet-torch",
    "classification-transformer-torch",
    "detection-transformer-torch",
    "zero-shot-detection-transformer-torch",
    "zero-shot-classification-transformer-torch",
)

# Workflows and associated parameters
WORKFLOWS = {
    "brain_selection": {},
    "learn_normality": {},
}

# Define the global variable
NUM_WORKERS = os.cpu_count()
GLOBAL_SEED = 0
