import os

# Selection from WORKFLOWS
SELECTED_WORKFLOW = ["embedding_selection"]

# Choose from config/datasets.yaml
SELECTED_DATASET = {
    "name": "fisheye8k",
    "n_samples": 100,  # 'None' (full dataset) or 'int' (subset of the dataset)
}

PERSISTENT = True  # If V51 database is stored

# Workflows and associated parameters
WORKFLOWS = {
    "aws_download": {
        "bucket": "mcity-data-engine",
        "prefix": "",
        "download_path": "output/datasets/annarbor_rolling",
        "test_run": True,
        "selected_dataset_overwrite": True,
    },
    "embedding_selection": {
        "mode": "load",  # "compute" or "load"
        "parameters": {
            "compute_representativeness": 0.99,
            "compute_unique_images_greedy": 0.01,
            "compute_unique_images_deterministic": 0.99,
            "compute_similar_images": 0.03,
            "neighbour_count": 3,
        },
        "embedding_models": [  # Select from V51 "Embeddings" models https://docs.voxel51.com/model_zoo/models.html
            "clip-vit-base32-torch",
            # "open-clip-torch",
            # "dinov2-vits14-torch",        # Issue with query IDs
            # "dinov2-vitl14-torch",        # Issue with query IDs
            # "dinov2-vits14-reg-torch",
            # "dinov2-vitl14-reg-torch",    # High GPU memory requirements
            "mobilenet-v2-imagenet-torch",
            # "resnet152-imagenet-torch",
            # "vgg19-imagenet-torch",
            # "classification-transformer-torch",
            # "detection-transformer-torch",
            # "zero-shot-detection-transformer-torch",
            # "zero-shot-classification-transformer-torch",
        ],
    },
    "anomaly_detection": {
        "mode": ["train", "inference"],  # "train" and "inference" supported
        "epochs": 1,
        "early_stop_patience": 5,
        "anomalib_image_models": {  # Choose from https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/models/image/index.html
            "Padim": {"batch_size": 1, "image_size": [960, 960]},
            "EfficientAd": {"batch_size": 1, "image_size": [960, 960]},
            # "Draem": {"batch_size": 1, "image_size": [960, 960]},
            "Cfa": {"batch_size": 1, "image_size": [960, 960]},
            "Cflow": {"batch_size": 1, "image_size": [960, 960]},
            # "Csflow": {"batch_size": 1, "image_size": [960, 960]},    # Node fc1_0 error
            "Dfm": {"batch_size": 1, "image_size": [960, 960]},
            "Dsr": {"batch_size": 1, "image_size": [960, 960]},
            "Fastflow": {"batch_size": 1, "image_size": [960, 960]},
            "ReverseDistillation": {"batch_size": 1, "image_size": [960, 960]},
            "Rkde": {"batch_size": 1, "image_size": [960, 960]},
            "Stfpm": {"batch_size": 1, "image_size": [960, 960]},
            "Uflow": {
                "batch_size": 1,
                "image_size": [448, 448],
            },  # Inflexible model w.r.t image size
            # "WinClip",    # Requires language input
            # "Dfkde",      # Has no pixel metrics
            # "Ganomaly",   # Has no pixel metrics
            # "Patchcore",  # Ineffiecient algorithm, cannot load whole dataset
        },
        "anomalib_eval_metrics": [  # Choose from https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/metrics/index.html
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
        "data_preparation": {"fisheye8k": {"location": "cam1", "rare_class": "Truck"}},
    },
    "auto_labeling": {
        "mode": ["train", "inference"],  # "train" and "inference" supported
        "model_source": [
            "hf_models_objectdetection"
        ],  # "hf_models_objectdetection" and "custom_codetr" and "ultralytics" supported
        "n_worker_dataloader": 3,
        "epochs": 2,
        "early_stop_patience": 5,
        "early_stop_threshold": 0,
        "learning_rate": 5e-05,
        "weight_decay": 0.0001,
        "max_grad_norm": 0.01,
        "hf_models_objectdetection": {
            "microsoft/conditional-detr-resnet-50": {"batch_size": 1},
            "Omnifact/conditional-detr-resnet-101-dc5": {"batch_size": 1},
            "facebook/detr-resnet-50": {"batch_size": 1},
            "facebook/detr-resnet-50-dc5": {"batch_size": 1, "image_size": [960, 960]},
            "facebook/detr-resnet-101": {"batch_size": 1, "image_size": [960, 960]},
            "facebook/detr-resnet-101-dc5": {"batch_size": 1, "image_size": [960, 960]},
            "facebook/deformable-detr-detic": {
                "batch_size": 1,
                "image_size": [960, 960],
            },
            "facebook/deformable-detr-box-supervised": {
                "batch_size": 1,
                "image_size": [960, 960],
            },
            "SenseTime/deformable-detr": {"batch_size": 1, "image_size": [960, 960]},
            "SenseTime/deformable-detr-with-box-refine-two-stage": {
                "batch_size": 1,
                "image_size": [960, 960],
            },
            "SenseTime/deformable-detr-with-box-refine": {
                "batch_size": 1,
                "image_size": [960, 960],
            },
            "jozhang97/deta-swin-large": {
                "batch_size": 1,
                "image_size": [960, 960],
            },  # Ranks best on HF Leaderboard: https://huggingface.co/spaces/hf-vision/object_detection_leaderboard
            "jozhang97/deta-swin-large-o365": {
                "batch_size": 1,
                "image_size": [960, 960],
            },
            "jozhang97/deta-resnet-50": {"batch_size": 1, "image_size": [960, 960]},
            "jozhang97/deta-resnet-50-24-epochs": {
                "batch_size": 1,
                "image_size": [960, 960],
            },
            "hustvl/yolos-base": {"batch_size": 1},
            # "PekingU/rtdetr_r50vd": {},              # Tensor size mismatch error
            # "PekingU/rtdetr_r50vd_coco_o365": {},    # Tensor size mismatch error
        },
        "custom_codetr": {
            "train_model": True,  # Set false if model file should be loaded without training
            "export_dataset_root": "/media/dbogdoll/Datasets/codetr_data/",
            "configs": [
                "projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py",
                "projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py",
            ],
            "n_gpus": "1",
            "container_tool": "docker",
        },
        "ultralytics": {},
    },
    "auto_labeling_zero_shot": {
        "n_post_processing_worker_per_inference_worker": 2,
        "n_worker_dataloader": 3,
        "prefetch_factor_dataloader": 2,
        "hf_models_zeroshot_objectdetection": {
            # dataset_chunks: Number of chunks to split the dataset into for parallel processing       # batch_size
            "omlab/omdet-turbo-swin-tiny-hf": {
                "batch_size": 64,
                "n_dataset_chunks": 1,
            },  # RTX 4090: 64 ; H 100: 128
            "IDEA-Research/grounding-dino-tiny": {
                "batch_size": 8,
                "n_dataset_chunks": 1,
            },  # RTX 4090: 8 ;  H 100: 32
            # "IDEA-Research/grounding-dino-base": {"batch_size": 8, "n_dataset_chunks": 1},            # RTX 4090: 8 ;  H 100: ?
            # "google/owlvit-base-patch16": {"batch_size": 8, "n_dataset_chunks": 1},                   # RTX 4090: 8 ;  H 100: ?
            # "google/owlvit-base-patch32": {"batch_size": 8, "n_dataset_chunks": 1},                   # RTX 4090: 8 ;  H 100: ?
            "google/owlvit-large-patch14": {
                "batch_size": 4,
                "n_dataset_chunks": 8,
            },  # RTX 4090: 4 ;  H 100: 16
            # "google/owlv2-base-patch16": {"batch_size": 32, "n_dataset_chunks": 1},                    # RTX 4090: 8 ;  H 100: 32
            # "google/owlv2-base-patch16-ensemble": {"batch_size": 8, "n_dataset_chunks": 1},           # RTX 4090: 8 ;  H 100: ?
            "google/owlv2-base-patch16-finetuned": {
                "batch_size": 8,
                "n_dataset_chunks": 8,
            },  # RTX 4090: 8 ;  H 100: 16
            # "google/owlv2-large-patch14": {"batch_size": 8, "n_dataset_chunks": 8},                   # RTX 4090: 2 ;  H 100: 8
            "google/owlv2-large-patch14-ensemble": {
                "batch_size": 2,
                "n_dataset_chunks": 8,
            },  # RTX 4090: 2 ;  H 100: 8
            # "google/owlv2-large-patch14-finetuned": {"batch_size": 2, "n_dataset_chunks": },          # RTX 4090: 2 ;  H 100: ?
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
            "delivery driver",
        ],
    },
    "ensemble_exploration": {
        "field_includes": "pred_zsod_",  # V51 field used for detections, "pred_zsod_" default for zero-shot object detection models
        "agreement_threshold": 3,  # Threshold for n models necessary for agreement between models
        "iou_threshold": 0.5,  # Threshold for IoU between bboxes to consider them as overlapping
        "max_bbox_size": 0.01,  # Value between [0,1] for the max size of considered bboxes
        "positive_classes": [  # Classes to consider, must be subset of available classes in the detections. Example for Vulnerable Road Users.
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
            "delivery driver",
        ],
    },
}

ACCEPTED_SPLITS = {"train", "val", "test"}
HF_ROOT = "mcity-data-engine"  # https://huggingface.co/mcity-data-engine
HF_DO_UPLOAD = False

cpu_count = os.cpu_count()
NUM_WORKERS = 32 if cpu_count > 32 else cpu_count
GLOBAL_SEED = 0
V51_ADDRESS = "localhost"
V51_PORT = 5151
V51_REMOTE = True
