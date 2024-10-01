import os

# Define the global variable
NUM_WORKERS = os.cpu_count()

# Selected dataset
SELECTED_DATASET = "mcity_fisheye_2000"  # Choose from datasets.yaml

# V51 embedding models
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

PERSISTENT = True  # If V51 database is stored
