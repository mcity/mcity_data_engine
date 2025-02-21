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
        "mcity": {
            "bucket": "mcity-data-engine",
            "prefix": "",
            "download_path": "output/datasets/annarbor_rolling",
            "test_run": True,
            "selected_dataset_overwrite": True,
        }
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
            "open-clip-torch",
            "dinov2-vits14-torch",
            "dinov2-vits14-reg-torch",
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
        "mode": ["train"],  # "train" and "inference" supported
        "epochs": 36,
        "early_stop_patience": 5,
        "anomalib_image_models": {  # Choose from https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/models/image/index.html
            "Padim": {},
            "EfficientAd": {},
            "Draem": {},
            "Cfa": {},
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
        "epochs": 1,
        "early_stop_patience": 5,
        "early_stop_threshold": 0,
        "learning_rate": 5e-05,
        "weight_decay": 0.0001,
        "max_grad_norm": 0.01,
        "hf_models_objectdetection": {  # HF Leaderboard: https://huggingface.co/spaces/hf-vision/object_detection_leaderboard
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
            },
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
        },
        "custom_codetr": {
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
    "mask_teacher": {
        "semantic_segmentation": {
            "sam2": {
                "prompt_field": None,  # None or Voxel51 field with bbox detections
                "models": [
                    "segment-anything-2-hiera-tiny-image-torch",
                    "segment-anything-2-hiera-small-image-torch",
                    "segment-anything-2-hiera-base-plus-image-torch",
                    "segment-anything-2-hiera-large-image-torch",
                    "segment-anything-2.1-hiera-tiny-image-torch",
                    "segment-anything-2.1-hiera-small-image-torch",
                    "segment-anything-2.1-hiera-base-plus-image-torch",
                    "segment-anything-2.1-hiera-large-image-torch",
                ],
            },
        },
        "depth_estimation": {
            "dpt": {
                "prompt_field": None,
                "models": {
                    "Intel/dpt-swinv2-tiny-256",
                    "Intel/dpt-swinv2-large-384",
                    "Intel/dpt-beit-large-384",
                    "Intel/dpt-beit-large-512",
                    "Intel/dpt-large-ade",
                    "Intel/dpt-large",
                    "Intel/dpt-hybrid-midas",
                    "Intel/dpt-swinv2-base-384",
                    "Intel/dpt-beit-base-384",
                },
            },
            "depth_anything": {
                "prompt_field": None,
                "models": {
                    "LiheYoung/depth-anything-base-hf",
                    "LiheYoung/depth-anything-large-hf",
                    "LiheYoung/depth-anything-small-hf",
                },
            },
            "glpn": {
                "prompt_field": None,
                "models": {"vinvino02/glpn-nyu", "vinvino02/glpn-kitti"},
            },
            "zoe_depth": {
                "prompt_field": None,
                "models": {
                    "Intel/zoedepth-nyu-kitti",
                    "Intel/zoedepth-nyu",
                    "Intel/zoedepth-kitti",
                },
            },
        },
    },
    "ensemble_selection": {
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

    "class_mapping": {
        # get the source and target dataset names from datasets.yaml
        "dataset_source": "fisheye8k",
        #"dataset_targets": [
            #"target_dataset_name_1",
         #   "mcity_fisheye_2000"
        #],
        "dataset_target": "mcity_fisheye_2000",
         # Choose any number of models from the options below hf_models_zeroshot_classification, to not include a model for class mapping, just comment it out
         #https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForZeroShotImageClassification
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
            "Car": ["car", "van", "pickup"],
            "Truck": ["truck", "pickup"]
            #Can add other class mappings in here
        },

        #"one_to_one_mapping":{
        #    "Bike" : ["motorbike/cycler"]

        #}
    }
}

# Global settings
ACCEPTED_SPLITS = ["train", "val", "test"]
cpu_count = os.cpu_count()
NUM_WORKERS = 32 if cpu_count > 32 else cpu_count
GLOBAL_SEED = 0

# Hugging Face Config
HF_ROOT = "mcity-data-engine"  # https://huggingface.co/mcity-data-engine
HF_DO_UPLOAD = False

# Weights and Biases Config
WANDB_ACTIVE = True

# Voxel51 Config
V51_ADDRESS = "localhost"
V51_PORT = 5151
V51_REMOTE = True
