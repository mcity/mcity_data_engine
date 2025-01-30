import os

# Selection from WORKFLOWS
SELECTED_WORKFLOW = ["mask_teacher"]

# Choose from config/datasets.yaml
SELECTED_DATASET = "Voxel51/fisheye8k"

PERSISTENT = True  # If V51 database is stored

# Workflows and associated parameters
WORKFLOWS = {
    "aws_download": {
        "source": "mcity_gridsmart",
        "start_date": "2023-11-19",
        "end_date": "2023-11-25",
        "sample_rate_hz": 1,
        "test_run": True,
        "selected_dataset_overwrite": True,
    },
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
            # "Dfkde",      # Has no pixel metrics
            # "Ganomaly",   # Has no pixel metrics
            # "Patchcore",  # Ineffiecient algorithm, cannot load whole dataset
        ],
        "anomalib_eval_metrics": [  # Choose from https://anomalib.readthedocs.io/en/v1.1.1/markdown/guides/reference/metrics/index.html
            "AUPR",
            "AUROC",
            "F1Max",
            # "AUPRO",      # Focus on standard metrics, computation of others can be expensive
            # "AnomalyScoreDistribution",
            # "BinaryPrecisionRecallCurve",
            # "F1AdaptiveThreshold",
            # "F1Score",
            # "ManualThreshold",
            # "MinMax",
            # "PRO",
        ],
    },
    "train_teacher": {
        "model_source": "custom_codetr",    # Pick from one of the options below (hf_models_objectdetection, custom_codetr, ultralytics)
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
        "custom_codetr": {
            "train_model": True,    # Set false if model file should be loaded without training
            "export_dataset_root": "/media/dbogdoll/Datasets/codetr_data/",
            "configs": [
                "projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py",
                "projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py"
                ],
            "n_gpus": "1",
            "container_tool": "docker"
        },
        "ultralytics": {}
    },
    "zero_shot_teacher": {
        "n_post_processing_worker_per_inference_worker": 2, 
        "n_worker_dataloader": 3,
        "prefetch_factor_dataloader": 2,
        "hf_models_zeroshot_objectdetection": {
            # dataset_chunks: Number of chunks to split the dataset into for parallel processing       # batch_size
            "omlab/omdet-turbo-swin-tiny-hf": {"batch_size": 180, "n_dataset_chunks": 1},              # RTX 4090: 64 ; H 100: 128
            "IDEA-Research/grounding-dino-tiny": {"batch_size": 32, "n_dataset_chunks": 1},            # RTX 4090: 8 ;  H 100: 32
            #"IDEA-Research/grounding-dino-base": {"batch_size": 8, "n_dataset_chunks": 1},            # RTX 4090: 8 ;  H 100: ?
            #"google/owlvit-base-patch16": {"batch_size": 8, "n_dataset_chunks": 1},                   # RTX 4090: 8 ;  H 100: ?
            #"google/owlvit-base-patch32": {"batch_size": 8, "n_dataset_chunks": 1},                   # RTX 4090: 8 ;  H 100: ?
            "google/owlvit-large-patch14": {"batch_size": 24, "n_dataset_chunks": 8},                  # RTX 4090: 4 ;  H 100: 16
            #"google/owlv2-base-patch16": {"batch_size": 32, "n_dataset_chunks": 1},                    # RTX 4090: 8 ;  H 100: 32
            #"google/owlv2-base-patch16-ensemble": {"batch_size": 8, "n_dataset_chunks": 1},           # RTX 4090: 8 ;  H 100: ?
            "google/owlv2-base-patch16-finetuned": {"batch_size": 32, "n_dataset_chunks": 8},          # RTX 4090: 8 ;  H 100: 16
            #"google/owlv2-large-patch14": {"batch_size": 8, "n_dataset_chunks": 8},                   # RTX 4090: 2 ;  H 100: 8
            "google/owlv2-large-patch14-ensemble": {"batch_size": 12, "n_dataset_chunks": 8},          # RTX 4090: 2 ;  H 100: 8
            #"google/owlv2-large-patch14-finetuned": {"batch_size": 2, "n_dataset_chunks": },          # RTX 4090: 2 ;  H 100: ?
        },
        "detection_threshold": 0.2,
        "object_classes": [
            "skater",
            "child",
            "bicycle",
            "bicyclist",
            "cyclist",
            "bike",
            "rider",
            "motorcycle",
            "motorcyclist",
            "pedestrian",
            "person",
            "walker",
            "jogger",
            "runner",
            "skateboarder",
            "scooter",
            "vehicle",
            "car",
            "bus",
            "truck",
            "taxi",
            "van",
            "pickup truck",
            "trailer",
            "emergency vehicle",
            "delivery driver"
        ],
    },
    "mask_teacher": {},
    "ensemble_exploration": {},
}

# Configuration for Weights and Biases
WANDB_CONFIG = {
    "docker_image": "dbogdollresearch/mcity_data_engine:latest",
    "docker_file": "Dockerfile.wandb",
    "queue": "data-engine",  # TODO Make this a command line argument to support cluster
    "entity": "mcity",
    "github": "git@github.com:daniel-bogdoll/mcity_data_engine.git",  # TODO https might be necessary for jobs on lighthouse
}

cpu_count = os.cpu_count()
NUM_WORKERS = 32 if cpu_count > 32 else cpu_count
GLOBAL_SEED = 0
V51_ADDRESS = "localhost"
V51_PORT = 5151
