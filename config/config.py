import os

SELECTED_WORKFLOW = "train_teacher"  # Selection from WORKFLOWS
SELECTED_DATASET = "mcity_fisheye_2000"  # Choose from datasets.yaml

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
    "Draem",
    "Cfa",
    "Cflow",
    "Csflow",
    "Dfm",
    "Dsr",
    "EfficientAd",
    "Fastflow",
    "ReverseDistillation",
    "Rkde",
    "Stfpm",
    "Uflow",
]
# "WinClip",    # Requires language input
# "Dfkde",  # Has no pixel metrics
# "Ganomaly",   # Has no pixel metrics
# "Patchcore",  # Ineffiecient algorithm, cannot load whole dataset


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
    "train_teacher": {
        "hf_models": [
            "microsoft/conditional-detr-resnet-50",
            "Omnifact/conditional-detr-resnet-101-dc5",
            "facebook/detr-resnet-50",
            "facebook/detr-resnet-50-dc5",
            "facebook/detr-resnet-101",
            "facebook/detr-resnet-101-dc5",
            "facebook/deformable-detr-detic",
            "facebook/deformable-detr-box-supervised",
            "SenseTime/deformable-detr",
            "SenseTime/deformable-detr-with-box-refine-two-stage",
            "SenseTime/deformable-detr-with-box-refine",
            "disco-eth/cue-detr",
            "PekingU/rtdetr_r50vd",
            "PekingU/rtdetr_r50vd_coco_o365",
            "jozhang97/deta-swin-large",
            "jozhang97/deta-swin-large-o365",
            "jozhang97/deta-resnet-50",
            "Ultralytics/YOLOv8",
            "Ultralytics/YOLO11",
            "hustvl/yolos-base",
            "jameslahm/yolov10x",
            "onnx-community/yolov10x",
            # "IDEA-Research/grounding-dino-tiny",
            # "google/owlvit-base-patch32",
            # "google/owlv2-base-patch16-ensemble",
            # "omlab/omdet-turbo-swin-tiny-hf",
        ]
    },
}

# Define the global variable
WANDB_CONFIG = {
    "docker_image": "dbogdollresearch/mcity_data_engine:latest",
    "docker_file": "Dockerfile.wandb",
    "queue": "data-engine",  # TODO Make this a command line argument
    "entity": "mcity",
    "github": "git@github.com:daniel-bogdoll/mcity_data_engine.git",  # TODO https might be necessary for jobs on lighthouse
}

NUM_WORKERS = os.cpu_count()
GLOBAL_SEED = 0
