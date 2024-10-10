import os

SELECTED_WORKFLOW = "learn_normality"  # Selection from WORKFLOWS
SELECTED_DATASET = "mcity_fisheye_ano_ped"  # Choose from datasets.yaml
SELECTED_SPLITS = ["train", "val"]  # Choose from datasets.yaml

PERSISTENT = False  # If V51 database is stored

# Select from V51 "Embeddings" models https://docs.voxel51.com/model_zoo/models.html
V51_EMBEDDING_MODELS = [
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
]

# Choose from https://anomalib.readthedocs.io/en/v1.1.1/markdown/guides/reference/models/image/index.html
ANOMALIB_IMAGE_MODELS = [
    "Padim",
    "Patchcore",
    "Draem",
    "Cfa",
    "Cflow",
    "Csflow",
    "Dfkde",
    "Dfm",
    "Dsr",
    "EfficientAd",
    "Fastflow",
    "Ganomaly",
    "ReverseDistillation",
    "Rkde",
    "Stfpm",
    "Uflow",
    "WinClip",
]

# Choose from https://anomalib.readthedocs.io/en/v1.1.1/markdown/guides/reference/metrics/index.html
ANOMALIB_EVAL_METRICS = [
    "AUPR",
    # "AUPRO",
    "AUROC",
    # "AnomalyScoreDistribution",
    # "BinaryPrecisionRecallCurve",
    # "F1AdaptiveThreshold",
    "F1Max",
    # "F1Score",
    # "ManualThreshold",
    # "MinMax",
    # "PRO",
]

# Workflows and associated parameters
WORKFLOWS = {
    "brain_selection": {},
    "learn_normality": {},
}

# Define the global variable
NUM_WORKERS = os.cpu_count()
GLOBAL_SEED = 0
