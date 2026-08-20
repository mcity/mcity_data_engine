import os
import psutil

#: Select workflow list from 'WORKFLOWS = {...}' dictionary
SELECTED_WORKFLOW = [""]

#: Select dataset from config/datasets.yaml
SELECTED_DATASET = {
    "name": "",
    "n_samples": None,
    "custom_view": None,
}

#: Runtime session state — managed by the agent via WorkflowState, do not edit manually
WORKFLOW_STATE = {'workflow_name': '', 'dataset_name': '', 'dataset_confirmed': False, 'labeled_dataset_name': '', 'auto_labeling': None, 'class_mapping': None, 'anomaly_detection': None, 'embedding_selection': None, 'auto_labeling_zero_shot': None, 'ensemble_selection': None, 'last_run': None, 'workflow_just_reset': False, 'delete_awaiting_confirmation': False, 'delete_confirmed': False, 'delete_pending_name': ''}

#: Workflows and associated parameters
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
        "mode": "compute",  # "compute" or "load"
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
        "mode": ["train", "inference"],  # "train" and "inference" supported
        "epochs": 12,
        "early_stop_patience": 5,
        "anomalib_image_models": {  # Choose from https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/models/image/index.html
            "Padim": {},
            # "EfficientAd": {},
            # "Draem": {},
            # "Cfa": {},
        },
        "anomalib_eval_metrics": [  # Choose from https://anomalib.readthedocs.io/en/v1.2.0/markdown/guides/reference/metrics/index.html. Focus on standard metrics, computation of others can be expensive
            "AUPR",
            "AUROC",
            "F1Max",
        ],
        "data_preparation": {
            "fisheye8k": {
                "location": "cam1",
                "rare_class": "Truck"
            }
        },
    },
    "auto_labeling": {
        "mode": ['inference'],
        "model_source": [
        # "hf_models_objectdetection",
        "ultralytics",
        # "custom_codetr",
        # "roboflow",
        # "roboflow_keypoint",
        # "vitpose",
        # "roi_keypoint",
        ],
        "n_worker_dataloader": 20,
        "epochs": 10,
        "early_stop_patience": 2,
        "early_stop_threshold": 0.0,
        "learning_rate": 2e-05,
        "weight_decay": 0.0001,
        "max_grad_norm": 0.01,
        "inference_settings": {
            "do_eval": False,
            "inference_on_test": False,
            # None (automatic selection) or overwrite with Hugging Face ID. Assumes same model as selected below.
            # Shared by every model source — reset to None before switching to a different source.
            "model_hf": "mcity-data-engine/fisheye8k_yolo12x",
            "detection_threshold": 0.2,
            "pedestrian_class_name": "pedestrian",  # class name to use for pedestrian tracking
        },
        "hf_models_objectdetection": {  # HF Leaderboard: https://huggingface.co/spaces/hf-vision/object_detection_leaderboard
            # "microsoft/conditional-detr-resnet-50": {"batch_size": 4},
            # "Omnifact/conditional-detr-resnet-101-dc5": {"batch_size": 1},
            # "facebook/detr-resnet-50": {"batch_size": 1},
            # "facebook/detr-resnet-50-dc5": {"batch_size": 1, "image_size": [960, 960]},
            # "facebook/detr-resnet-101": {"batch_size": 4, "image_size": [960, 960]},
            # "facebook/detr-resnet-101-dc5": {"batch_size": 1, "image_size": [960, 960]},
            # "facebook/deformable-detr-detic": {"batch_size": 4, "image_size": [960, 960] },
            # "facebook/deformable-detr-box-supervised": {"batch_size": 1, "image_size": [960, 960]},
            # "SenseTime/deformable-detr": {"batch_size": 4, "image_size": [960, 960]},
            # "SenseTime/deformable-detr-with-box-refine": {"batch_size": 1, "image_size": [960, 960]},
            # "jozhang97/deta-swin-large": {"batch_size": 1, "image_size": [960, 960]},
            # "jozhang97/deta-swin-large-o365": {"batch_size": 4, "image_size": [960, 960]},
            # "hustvl/yolos-base": {"batch_size": 4},
            "IDEA-Research/dab-detr-resnet-50": {"batch_size": 4, "image_size": [960, 960]},
            # "PekingU/rtdetr_v2_r18vd": {"batch_size": 4, "image_size": [960, 960]},
        },
        "custom_codetr": {
            "export_dataset_root": "output/datasets/codetr_data/",
            "configs": [
#                "projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py",
#"projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py",
            ],
            "n_gpus": "1",
            "container_tool": "docker",
        },
        "roboflow": {  # Roboflow RF-DETR configuration
            "export_dataset_root": "output/datasets/roboflow_data/",
            "configs": [
                # "rfdetr_nano",
                # "rfdetr_small",
                 "rfdetr_medium",
                #"rfdetr_large",
		#"rfdetr_xlarge",
  # "rfdetr_2xlarge",
            ],
            # Fallback HuggingFace repo to download best.pt from when no local model is found.
            # Maps config name → HF repo ID. Set to None to skip HF fallback for that config.
            "fallback_hf_repo": {
                "rfdetr_2xlarge": "mcity-ai/rfdetr_2xlarge_mcity_31k",
            },
            # Class names for inference-only datasets that have no ground_truth or annotation files.
            # Order must match the class IDs the model was trained with (0-indexed).
            # Set to None to auto-detect from ground_truth field or exported annotation files.
            "class_names": [
                "bus",
                "car",
                "motorbike/cycler",
                "pedestrian",
                "pickup",
                "trailer",
                "truck",
                "van",
            ],
            # RF-DETR specific parameters only
            "batch_size": 10,                     # Override default batch size
            "grad_accum_steps": 4,                # Gradient accumulation steps
            "lr_encoder": None,                   # Encoder-specific learning rate (optional)
            "resolution": None,                   # Image resolution, must be divisible by 56 (optional)
            "use_ema": True,                      # Exponential moving average
            "gradient_checkpointing": False,      # Memory optimization
            "early_stopping_min_delta": 0.0001,    # Minimum improvement for early stopping
            "early_stopping_use_ema": True,       # Use EMA model for early stopping
        },
        "roboflow_keypoint": {  # RF-DETR with dual bbox + keypoint heads
            "export_dataset_root": "output/datasets/roboflow_kp_data/",
            "configs": [
                # "rfdetr_base",
                # "rfdetr_large",
                # "rfdetr_xlarge",
                "rfdetr_2xlarge",
            ],
            # ── FiftyOne-native mode (no COCO export needed) ──────────
            "fo_native": True,                      # read directly from FiftyOne
            "detection_field": "ground_truth",      # fo.Detections field name
            "keypoint_field":  "pedestrian_points", # fo.Keypoints field name
            "target_label":    "pedestrian",        # only this label is used
            "class_names":     ["pedestrian"],
            "num_classes":     1,
            # ── Keypoint configuration ────────────────────────────────
            "keypoint_names": ["ankle_center"],     # midpoint between left+right ankles
            "kp_xy_coef": 5.0,                      # weight for OKS coordinate loss
            "kp_l1_coef": 2.0,                      # weight for L1 coordinate loss (prevents OKS dead zone when predictions are far from GT)
            "kp_vis_coef": 1.0,                     # weight for visibility loss
            "kp_sigma": 0.089,                      # OKS sigma for ankle_center (COCO ankle standard)
            "val_iou_thresh": 0.2,                  # IoU threshold for TP matching during validation (0.3 tolerates frozen bbox head domain shift)
            "freeze_backbone_epochs": 5,
            "freeze_bbox_head": True,               # keep bbox/class/transformer frozen; train keypoint_embed only
            # ── Training parameters ───────────────────────────────────
            "batch_size": 8,
            "lr_encoder": None,
            "resolution": 880,  # rfdetr_2xlarge=880, xlarge=700, base/large=560
            # ── Pre-trained weights ───────────────────────────────────
            # Start from the COCO-pretrained checkpoint to benefit from
            # the bbox head warm-start; keypoint head is re-initialised.
            # Path was previously /home/dataengine/... (wrong machine path → silently
            # skipped, model started randomly → TPs=0 all training). Fixed to NFS path.
            "pretrain_weights": "/nfs/turbo/coe-mcity/rpatnaik/Mcity/rf-detr-data-engine/mcity_data_engine/output/models/rfdetr_kp/coco-2017-keypoints/rfdetr_2xlarge/best.pt",
        },
        "vitpose": {  # Standalone ViTPose-B fine-tuning on GT RoI crops
            # ── ViTPose-B pretrained init (HuggingFace ID or local path) ───────────
            # Downloaded automatically on first run; cached in vitpose_save_dir.
            "vitpose_pretrain_weights": "usyd-community/vitpose-base-simple",
            "model_name": "vitpose_base",  # used in FiftyOne field names: pred_kp_{model_name}-{dataset}
            "resume_weights": None,        # set to vitpose_best.pt path to continue training from a checkpoint
            "vitpose_save_dir": "output/models/vitpose/",
            # ── Dataset / annotation ───────────────────────────────────────────────
            "detection_field": "ground_truth", #"pred_od_rfdetr_2xlarge-ashley-huron-jan-2026",  # RF-DETR predicted boxes
            "detection_conf_thresh": 0.2,           # min confidence for RF-DETR pedestrian boxes
            "keypoint_field":  "pedestrian_points", # fo.Keypoints with ankle keypoints (GT)
            "target_label":    "pedestrian",             #"class_3",           # label used in detection_field for pedestrians
            "keypoint_names":  ["ankle_center"],
            "kp_sigma": 0.089,
            # ── Training parameters ────────────────────────────────────────────────
            "val_split_fallback": 0.1,    # fraction of train used as val when no val split exists
            "batch_size": 32,
            "freeze_backbone": True,      # freeze ViT-B encoder for first N epochs
            "freeze_backbone_epochs": 5,  # then unfreeze at 0.1× learning_rate
        },
        "roi_keypoint": {  # Two-head model: RF-DETR (Head 1) → ViTPose (Head 2)
            # ── Head 1: RF-DETR detector ───────────────────────────────────────────
            # Runs live on each image; detections saved to detection_field in FiftyOne.
            "rfdetr_model":          "rfdetr_2xlarge",
            "pretrain_weights":      "/nfs/turbo/coe-mcity/rpatnaik/Mcity/rf-detr-data-engine/mcity_data_engine/output/models/rfdetr_kp/coco-2017-keypoints/rfdetr_2xlarge/best.pt",
            "detection_conf_thresh": 0.3,
            "detection_field":       "pred_od_rfdetr_2xlarge-mcity-person-kp_bb",  # Head 1 output field
            # ── Head 2: Fine-tuned ViTPose-B ───────────────────────────────────────
            # Receives RoIs from Head 1; set to vitpose_best.pt from vitpose workflow.
            "vitpose_weights": "output/models/vitpose/<dataset>/vitpose_best.pt",
            # ── Shared annotation config ───────────────────────────────────────────
            "target_label":   "pedestrian",
            "keypoint_names": ["ankle_center"],
            "kp_sigma": 0.089,
        },
        "ultralytics": {
            "export_dataset_root": "output/datasets/ultralytics_data/",
            "multi_scale": False,
            "cos_lr": True,
            "models": {  # Pick from https://docs.ultralytics.com/models/
                # "yolo11n": {"batch_size": 8, "img_size": 1280},
                # "yolo11x": {"batch_size": 1, "img_size": 960},
                # "yolo12n": {"batch_size": 8, "img_size": 1280},
                "yolo12x": {"batch_size": 1, "img_size": 960},
                # "yolo26x": {"batch_size": 1, "img_size": 960},
                # "yolo26l": {"batch_size": 1, "img_size": 960},
                # "yolo26m": {"batch_size": 1, "img_size": 960},
            },
        },
    },
    "auto_labeling_zero_shot": {
        "n_post_processing_worker_per_inference_worker": 5,
        "n_worker_dataloader": 3,
        "prefetch_factor_dataloader": 2,
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
        "hf_models_zeroshot_objectdetection": {
            "omlab/omdet-turbo-swin-tiny-hf": {"batch_size": 1, "n_dataset_chunks": 1, },
            "IDEA-Research/grounding-dino-tiny": {"batch_size": 1, "n_dataset_chunks": 1, },
            "google/owlvit-large-patch14": {"batch_size": 1, "n_dataset_chunks": 1, },
            "google/owlv2-base-patch16-finetuned": {"batch_size": 1, "n_dataset_chunks": 1, },
            "google/owlv2-large-patch14-ensemble": {"batch_size": 1, "n_dataset_chunks": 1, },
        },
    },
    "auto_label_mask": {
        "semantic_segmentation": {
            "sam2": {
                "prompt_field": None,
                "models": [
                    "segment-anything-2-hiera-tiny-image-torch",
                    "segment-anything-2-hiera-small-image-torch",
                    "segment-anything-2-hiera-base-plus-image-torch",
                    "segment-anything-2.1-hiera-tiny-image-torch",
                    "segment-anything-2.1-hiera-small-image-torch",
                    "segment-anything-2.1-hiera-base-plus-image-torch",
                    "segment-anything-2.1-hiera-large-image-torch",
                ],
            },
        },
        "depth_estimation": {
            "dpt": {
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
                "models": {
                    "LiheYoung/depth-anything-base-hf",
                    "LiheYoung/depth-anything-large-hf",
                    "LiheYoung/depth-anything-small-hf",
                },
            },
            "depth_pro": {
                "models": {
                    "apple/DepthPro-hf",
                }
            },
            "glpn": {
                "models": {"vinvino02/glpn-nyu",
                           "vinvino02/glpn-kitti"},
            },
            "zoe_depth": {
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
        "iou_threshold": 0.3,  # Threshold for IoU between bboxes to consider them as overlapping
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
        "dataset_target": "mcity_fisheye_2000",
        # Set to True to change detection labels in the dataset, Set to False to just add tags without changing labels in the dataset.
        "change_labels": False,

         # Choose any number of models from the options below hf_models_zeroshot_classification, to not include a model for class mapping, just comment it out
         #https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForZeroShotImageClassification
        "hf_models_zeroshot_classification": [
            "Salesforce/blip2-itm-vit-g",
            "openai/clip-vit-large-patch14",
            "google/siglip-so400m-patch14-384",
            "kakaobrain/align-base",
            "BAAI/AltCLIP",
            "CIDAS/clipseg-rd64-refined",
        ],

        "candidate_labels": {
            "Truck": ["truck", "pickup"],
        },
        "thresholds": {"confidence": 0.2},
    },
    "vitpose_download": {
        # Download ViTPose-B pretrained weights from HuggingFace.
        # Run this once before starting roi_keypoint training.
        "vitpose_model": "usyd-community/vitpose-base-simple",
        "save_dir":      "output/models/vitpose/",
    },
    "data_ingest": {
        "dataset_name": "gs_gen_zina_msight",
        "annotation_format": "auto",  # Options: "auto", "coco", "voc", "yolo", "image_only", "video"
        "dataset_dir": "/home/dataengine/Mcity/sip_s3_data/gs_gen_zina",
        "split_percentages": [0.7, 0.15, 0.15],  # Optional train/val/test
        "fps": 20, #Frames per second to convert a Video dataset to Fiftyone Image Dataset
    }
}

MSIGHT_CONFIG = {
    # Set to True only for a dataset of the camera that the calibration files
    # below describe. The detection_field must exist in that dataset.
    "run_localization": False,
    "detection_field": "pred_od_rfdetr_2xlarge-"+SELECTED_DATASET['name'],
    "loc_maps": "MSight/data/calibration_results_ashley_huron.npz",
    "intrinsics": "MSight/data/ashley_huron_intrinsic.json",
}

"""Global settings"""
#: Non-persistent datasets are deleted from the database each time the database is shut down
PERSISTENT = True
#: Accepted splits for data processing
ACCEPTED_SPLITS = ["train", "val", "test"]
cpu_count = len(psutil.Process().cpu_affinity())
#: Max. number of CPU workers
NUM_WORKERS_MAX = 32
NUM_WORKERS = NUM_WORKERS_MAX if cpu_count > NUM_WORKERS_MAX else cpu_count
#: SEED for reproducability
GLOBAL_SEED = 0

"""Hugging Face Config"""
#: Hugging Face name or Organization
HF_ROOT = "mcity-engineering"  # https://huggingface.co/mcity-data-engine
#: Determins if model weights should be uploaded to Hugging Face
HF_DO_UPLOAD = False

"""Weights and Biases Config"""
#: Determines if tracking with Weights and Biases is activated
WANDB_ACTIVE = False

"""Voxel51 Config"""
#: Address for Voxel51 connection
V51_ADDRESS = "0.0.0.0"
#: Port for Voxel51 connection
V51_PORT = 5151
#: Remote app sessions will listen to any connection to their ports
V51_REMOTE = True
