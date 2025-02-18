import os

# Selection from WORKFLOWS
SELECTED_WORKFLOW = ["class_mapping"]

# Choose from config/datasets.yaml
SELECTED_DATASET = {
    "name": "fisheye8k",
    "n_samples": None,  # 'None' (full dataset) or 'int' (subset of the dataset)
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
        "thresholds": {
            "compute_representativeness": 0.99,
            "compute_unique_images_greedy": 0.01,
            "compute_unique_images_deterministic": 0.99,
            "compute_similar_images": 0.03,
            "neighbour_count": 3,
        },
        "embedding_models": [  # Select from V51 "Embeddings" models https://docs.voxel51.com/model_zoo/models.html
            "clip-vit-base32-torch",
            "open-clip-torch",
            # "dinov2-vits14-torch",        # Issue with query IDs
            # "dinov2-vitl14-torch",        # Issue with query IDs
            "dinov2-vits14-reg-torch",
            # "dinov2-vitl14-reg-torch",    # High GPU memory requirements
            "mobilenet-v2-imagenet-torch",
            "resnet152-imagenet-torch",
            "vgg19-imagenet-torch",
            "classification-transformer-torch",
            "detection-transformer-torch",
            "zero-shot-detection-transformer-torch",
            "zero-shot-classification-transformer-torch",
        ],
    },
    "anomaly_detection": {
        "mode": "train",  # "train" or "inference"
        "epochs": 30,
        "early_stop_patience": 5,
        "anomalib_image_models": {  # Choose from https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/models/image/index.html
            "Padim": {"batch_size": 1, "image_size": [960, 960]},
            "EfficientAd": {"batch_size": 1, "image_size": [960, 960]},
            "Draem": {"batch_size": 1, "image_size": [960, 960]},
            "Cfa": {"batch_size": 1, "image_size": [960, 960]},
            "Cflow": {"batch_size": 1, "image_size": [960, 960]},
            "Csflow": {"batch_size": 1, "image_size": [960, 960]},
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
        "mode": "train",  # "train" or "inference"
        "model_source": "hf_models_objectdetection",  # Pick from one of the options below (hf_models_objectdetection, custom_codetr, ultralytics)
        "hf_models_objectdetection": [
            # "microsoft/conditional-detr-resnet-50",
            # "Omnifact/conditional-detr-resnet-101-dc5",
            # "facebook/detr-resnet-50",
            # "facebook/detr-resnet-50-dc5",
            # "facebook/detr-resnet-101",
            # "facebook/detr-resnet-101-dc5",
            # "facebook/deformable-detr-detic",
            # "facebook/deformable-detr-box-supervised",
            # "SenseTime/deformable-detr",
            # "SenseTime/deformable-detr-with-box-refine-two-stage",
            # "SenseTime/deformable-detr-with-box-refine",
            # "PekingU/rtdetr_r50vd",
            # "PekingU/rtdetr_r50vd_coco_o365",
            "jozhang97/deta-swin-large",  # Ranks best on HF Leaderboard: https://huggingface.co/spaces/hf-vision/object_detection_leaderboard
            # "jozhang97/deta-swin-large-o365",
            # "jozhang97/deta-resnet-50",
            # "jozhang97/deta-resnet-50-24-epochs",
            # "hustvl/yolos-base",
        ],
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
                "batch_size": 180,
                "n_dataset_chunks": 1,
            },  # RTX 4090: 64 ; H 100: 128
            "IDEA-Research/grounding-dino-tiny": {
                "batch_size": 32,
                "n_dataset_chunks": 1,
            },  # RTX 4090: 8 ;  H 100: 32
            # "IDEA-Research/grounding-dino-base": {"batch_size": 8, "n_dataset_chunks": 1},            # RTX 4090: 8 ;  H 100: ?
            # "google/owlvit-base-patch16": {"batch_size": 8, "n_dataset_chunks": 1},                   # RTX 4090: 8 ;  H 100: ?
            # "google/owlvit-base-patch32": {"batch_size": 8, "n_dataset_chunks": 1},                   # RTX 4090: 8 ;  H 100: ?
            "google/owlvit-large-patch14": {
                "batch_size": 24,
                "n_dataset_chunks": 8,
            },  # RTX 4090: 4 ;  H 100: 16
            # "google/owlv2-base-patch16": {"batch_size": 32, "n_dataset_chunks": 1},                    # RTX 4090: 8 ;  H 100: 32
            # "google/owlv2-base-patch16-ensemble": {"batch_size": 8, "n_dataset_chunks": 1},           # RTX 4090: 8 ;  H 100: ?
            "google/owlv2-base-patch16-finetuned": {
                "batch_size": 32,
                "n_dataset_chunks": 8,
            },  # RTX 4090: 8 ;  H 100: 16
            # "google/owlv2-large-patch14": {"batch_size": 8, "n_dataset_chunks": 8},                   # RTX 4090: 2 ;  H 100: 8
            "google/owlv2-large-patch14-ensemble": {
                "batch_size": 12,
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
    "ensemble_exploration": {},
    "class_mapping": {
        "model_source": "hf_models_zeroshot_classification",  # Choose any number of models from the options below hf_models_zeroshot_classification, to not include a model for class mapping, just comment it out
        "hf_models_zeroshot_classification": [
            "Salesforce/blip2-itm-vit-g",
            "openai/clip-vit-large-patch14",
            "google/siglip-so400m-patch14-384",
            "kakaobrain/align-base",
            "BAAI/AltCLIP",
            "CIDAS/clipseg-rd64-refined"
        ],
        "thresholds": {
            "confidence": 0.2
        },
        "candidate_labels": {
            #Parent class(Generalized class) : Children classes(specific categories)
            "Car": ["Car", "Van", "Pickup"],
            "Truck": ["Truck", "Pickup"]
            #Can add other class mappings in here
        }
    }
}

ACCEPTED_SPLITS = {"train", "val", "test"}

cpu_count = os.cpu_count()
NUM_WORKERS = 32 if cpu_count > 32 else cpu_count
GLOBAL_SEED = 0
V51_ADDRESS = "localhost"
V51_PORT = 5151
