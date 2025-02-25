import os

# Selection from WORKFLOWS
SELECTED_WORKFLOW = ["embedding_selection"]

# Choose from config/datasets.yaml
SELECTED_DATASET = {
    "name": "mcity_fisheye_2000",
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
            # "open-clip-torch",
            # "dinov2-vits14-torch",
            # "dinov2-vits14-reg-torch",
            # "mobilenet-v2-imagenet-torch",
            # "resnet152-imagenet-torch",
            # "vgg19-imagenet-torch",
            # "classification-transformer-torch",
            # "detection-transformer-torch",
            # "zero-shot-detection-transformer-torch",
            # "zero-shot-classification-transformer-torch",
        ],
    },
    "anomaly_detection": {
        "mode": ["inference"],  # "train" and "inference" supported
        "epochs": 36,
        "early_stop_patience": 5,
        "anomalib_image_models": {  # Choose from https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/models/image/index.html
            "Padim": {},
            "EfficientAd": {},
            "Draem": {},
            "Cfa": {},
        },
        "anomalib_eval_metrics": [  # Choose from https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/metrics/index.html. Focus on standard metrics, computation of others can be expensive
            "AUPR",
            "AUROC",
            "F1Max",
        ],
        "data_preparation": {"fisheye8k": {"location": "cam1", "rare_class": "Truck"}},
    },
    "auto_labeling": {
        "mode": ["inference"],  # "train" and "inference" supported
        "model_source": [
            "ultralytics"
        ],  # "hf_models_objectdetection" and "custom_codetr" and "ultralytics" supported
        "n_worker_dataloader": 3,
        "epochs": 32,
        "early_stop_patience": 5,
        "early_stop_threshold": 0,
        "learning_rate": 5e-05,
        "weight_decay": 0.0001,
        "max_grad_norm": 0.01,
        "inference_settings": {
            "do_eval": True,
            "inference_on_evaluation": True,
            "model_hf": None,  # None (automatic selection) or Hugging Face ID
        },
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
        "ultralytics": {
            "export_dataset_root": "/media/dbogdoll/Datasets/ultralytics_data/",
            "models": {  # Pick from https://docs.ultralytics.com/models/
                # "yolo11n": {"batch_size": 32, "img_size": 640},
                "yolo11x": {"batch_size": 1, "img_size": 640},
                # "yolo12n": {"batch_size": 32, "img_size": 640},
                # "yolo12x": {"batch_size": 1, "img_size": 640},
            },
        },
    },
    "auto_labeling_zero_shot": {
        "n_post_processing_worker_per_inference_worker": 5,
        "n_worker_dataloader": 3,
        "prefetch_factor_dataloader": 2,
        "hf_models_zeroshot_objectdetection": {
            "omlab/omdet-turbo-swin-tiny-hf": {  # https://huggingface.co/models?pipeline_tag=zero-shot-object-detection&sort=trending&search=omlab%2Fomdet
                "batch_size": 1,
                "n_dataset_chunks": 1,  # Number of chunks to split the dataset into for parallel processing
            },
            "IDEA-Research/grounding-dino-tiny": {  # https://huggingface.co/models?pipeline_tag=zero-shot-object-detection&sort=trending&search=IDEA-Research%2Fgrounding
                "batch_size": 1,
                "n_dataset_chunks": 1,
            },
            "google/owlvit-large-patch14": {  # https://huggingface.co/models?pipeline_tag=zero-shot-object-detection&sort=trending&search=google%2Fowlvit
                "batch_size": 1,
                "n_dataset_chunks": 1,
            },
            "google/owlv2-base-patch16-finetuned": {  # https://huggingface.co/models?pipeline_tag=zero-shot-object-detection&sort=trending&search=google%2Fowlv2
                "batch_size": 1,
                "n_dataset_chunks": 1,
            },
            "google/owlv2-large-patch14-ensemble": {
                "batch_size": 1,
                "n_dataset_chunks": 1,
            },
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
}

# Global settings
ACCEPTED_SPLITS = ["train", "val", "test"]
cpu_count = os.cpu_count()
NUM_WORKERS = 32 if cpu_count > 32 else cpu_count
GLOBAL_SEED = 0

# Hugging Face Config
HF_ROOT = "mcity-data-engine"  # https://huggingface.co/mcity-data-engine
HF_DO_UPLOAD = True

# Weights and Biases Config
WANDB_ACTIVE = True

# Voxel51 Config
V51_ADDRESS = "localhost"
V51_PORT = 5151
V51_REMOTE = True
