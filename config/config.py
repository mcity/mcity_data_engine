import os

SELECTED_WORKFLOW = [
    "zero_shot_teacher",
    "ensemble_exploration",
]  # Selection from WORKFLOWS
SELECTED_DATASET = "mcity_fisheye_3_months"  # Choose from datasets.yaml

PERSISTENT = True  # If V51 database is stored

# Workflows and associated parameters
WORKFLOWS = {
    "brain_selection": {
        "embedding_models": [  # Select from V51 "Embeddings" models https://docs.voxel51.com/model_zoo/models.html
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
    },
    "learn_normality": {
        "anomalib_image_models": [  # Choose from https://anomalib.readthedocs.io/en/v1.1.1/markdown/guides/reference/models/image/index.html
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
            # "WinClip",    # Requires language input
            # "Dfkde",  # Has no pixel metrics
            # "Ganomaly",   # Has no pixel metrics
            # "Patchcore",  # Ineffiecient algorithm, cannot load whole dataset
        ],
        "anomalib_eval_metrics": [  # Choose from https://anomalib.readthedocs.io/en/v1.1.1/markdown/guides/reference/metrics/index.html
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
        ],
    },
    "train_teacher": {
        "hf_models_objectdetection": [
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
            "PekingU/rtdetr_r50vd",
            "PekingU/rtdetr_r50vd_coco_o365",
            "jozhang97/deta-swin-large",
            "jozhang97/deta-swin-large-o365",
            "jozhang97/deta-resnet-50",
            "jozhang97/deta-resnet-50-24-epochs",
            "hustvl/yolos-base",
        ],
    },
    "zero_shot_teacher": {
        "hf_models_zeroshot_objectdetection": [
            "IDEA-Research/grounding-dino-tiny",
            "IDEA-Research/grounding-dino-base",
            # "google/owlvit-base-patch16",
            # "google/owlvit-base-patch32",
            "google/owlvit-large-patch14",
            "google/owlv2-base-patch16",
            "google/owlv2-base-patch16-ensemble",
            "google/owlv2-base-patch16-finetuned",
            "google/owlv2-large-patch14",
            "google/owlv2-large-patch14-ensemble",
            "google/owlv2-large-patch14-finetuned",
            "omlab/omdet-turbo-swin-tiny-hf",
        ]
    },
    "ensemble_exploration": {},
}

# Define the global variable
WANDB_CONFIG = {
    "docker_image": "dbogdollresearch/mcity_data_engine:latest",
    "docker_file": "Dockerfile.wandb",
    "queue": "data-engine",  # TODO Make this a command line argument
    "entity": "mcity",
    "github": "git@github.com:daniel-bogdoll/mcity_data_engine.git",  # TODO https might be necessary for jobs on lighthouse
}

cpu_count = os.cpu_count()
NUM_WORKERS = 32 if cpu_count > 32 else cpu_count
GLOBAL_SEED = 0
V51_ADDRESS = "localhost"
V51_PORT = 5151
