"""
Train MMPose (RTMPose-s) on lower body keypoints from FiftyOne dataset: coco-person-2017
With comprehensive OKS performance validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Export strategy
  Uses FiftyOne's built-in COCODetectionDataset exporter to produce the
  standard COCO JSON structure, then patches the resulting labels.json
  in-place to:
    • Extract only lower body keypoints (hips, knees, ankles) from the full
      17-joint COCO keypoints
    • add keypoint coordinates + visibility flags from the fo.Keypoints field
    • inject the "keypoints" / "skeleton" fields into the category entry
  This means all bounding-box and image metadata come directly from FiftyOne
  (no manual pixel-math required) and keypoints are appended cleanly on top.

Dataset assumptions
  • fo.load_dataset("coco-person-2017")
  • Detection field : "ground_truth"   (fo.Detections – bounding boxes)
  • Keypoint field  : "keypoints"      (fo.Keypoints  – 17 body joints, we extract 6)
  • Keypoint label  : "person"
  • Sample tags     : "train" / "validation"
  • Lower body only : left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

What this script does
  1. Loads configuration from YAML file
  2. Determines job type (train/inference/both)
  3. Inspects the dataset (field schema, first-sample peek)
  4. Exports each split with fo.export(dataset_type=COCODetectionDataset)
  5. Patches the exported JSON to embed ONLY lower body keypoints + skeleton metadata
  6. Builds a full MMPose Config (RTMPose-s, SimCC head) for 6 keypoints
  7. Trains (if job_type includes training)
  8. Runs inference and validation on best/specified checkpoint
  9. Generates detailed performance reports and visualizations

Dependencies
  pip install torch torchvision
  pip install openmim && mim install mmengine "mmcv>=2.0" mmdet mmpose
  pip install fiftyone matplotlib seaborn pandas pyyaml
"""

import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import fiftyone as fo
import fiftyone.types as fot
from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.utils import register_all_modules
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import PoseDataSample

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

register_all_modules()


# ── Configuration loading ─────────────────────────────────────────────────────
def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    log.info("━━━ Configuration loaded ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  Config file: %s", config_path)
    log.info("  Dataset: %s", config['dataset']['name'])
    log.info("  Job type: %s", config['job']['type'])
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return config


def find_best_checkpoint(work_dir: str) -> Optional[str]:
    """
    Find the best checkpoint in work_dir.
    Priority: best.pt > best_coco_AP_epoch_*.pth > epoch_*.pth (latest) > last.pth > None
    """
    work_path = Path(work_dir)

    if not work_path.exists():
        log.warning("  Work directory does not exist: %s", work_dir)
        return None

    log.info("Searching for checkpoints in: %s", work_path.absolute())

    # 1. Check for best.pt (new standard format)
    best_pt = work_path / "best.pt"
    if best_pt.exists():
        log.info("  ✓ Found best.pt")
        return str(best_pt)

    # 2. Check for best_coco_AP_epoch_*.pth
    best_checkpoints = sorted(work_path.glob("best_coco_AP_epoch_*.pth"))
    if best_checkpoints:
        log.info("  ✓ Using best checkpoint: %s", best_checkpoints[-1].name)
        return str(best_checkpoints[-1])

    # 3. Check for epoch_*.pth (latest)
    epoch_checkpoints = sorted(work_path.glob("epoch_*.pth"))
    if epoch_checkpoints:
        log.info("  ✓ Using latest epoch checkpoint: %s", epoch_checkpoints[-1].name)
        return str(epoch_checkpoints[-1])

    # 4. Check for last.pth
    last_checkpoint = work_path / "last.pth"
    if last_checkpoint.exists():
        log.info("  ✓ Using last checkpoint: last.pth")
        return str(last_checkpoint)

    log.warning("  No checkpoints found")
    return None


def download_from_huggingface(repo_id: str, filename: str, revision: str = "main", cache_dir: Optional[str] = None) -> str:
    """
    Download checkpoint from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "username/model-repo")
        filename: File to download (e.g., "best.pt")
        revision: Branch/tag/commit (default: "main")
        cache_dir: Cache directory (default: HF default cache)

    Returns:
        Path to downloaded checkpoint
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for downloading from HuggingFace.\n"
            "Install it with: pip install huggingface-hub"
        )

    log.info("━━━ Downloading from HuggingFace ━━━━━━━━━━━━━━━━━━━━")
    log.info("  Repository: %s", repo_id)
    log.info("  File: %s", filename)
    log.info("  Revision: %s", revision)

    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir
        )
        log.info("  ✓ Downloaded to: %s", checkpoint_path)
        return checkpoint_path
    except Exception as e:
        log.error("Failed to download from HuggingFace: %s", e)
        raise


def resolve_pretrained_checkpoint(config: Dict) -> Optional[str]:
    """
    Resolve pretrained checkpoint path based on finetune configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Path to pretrained checkpoint, or None if training from scratch
    """
    finetune_config = config.get('finetune', {})

    if not finetune_config.get('enabled', False):
        log.info("Training from scratch (finetuning disabled)")
        return None

    source = finetune_config.get('source', 'local')

    if source == 'huggingface':
        hf_config = finetune_config.get('huggingface', {})
        return download_from_huggingface(
            repo_id=hf_config['repo_id'],
            filename=hf_config.get('filename', 'best.pt'),
            revision=hf_config.get('revision', 'main'),
            cache_dir=hf_config.get('cache_dir')
        )

    elif source == 'local':
        checkpoint_path = finetune_config.get('local', {}).get('checkpoint_path')
        if not checkpoint_path:
            raise ValueError("finetune.local.checkpoint_path is required when source='local'")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

        log.info("Using local pretrained checkpoint: %s", checkpoint_path)
        return checkpoint_path

    else:
        raise ValueError(
            f"Invalid finetune source: {source}. Must be 'huggingface' or 'local'"
        )


    # List all .pth files for debugging
    all_checkpoints = list(work_path.glob("*.pth"))
    log.info("  Found %d .pth files in %s", len(all_checkpoints), work_dir)
    for ckpt in all_checkpoints[:5]:  # Show first 5
        log.info("    - %s", ckpt.name)

    # Look for best checkpoint with metrics
    best_checkpoints = sorted(work_path.glob("best_coco_AP_epoch_*.pth"))
    if best_checkpoints:
        log.info("  Using best checkpoint: %s", best_checkpoints[-1].name)
        return str(best_checkpoints[-1])

    # Look for epoch checkpoints (when validation is disabled)
    epoch_checkpoints = sorted(work_path.glob("epoch_*.pth"))
    if epoch_checkpoints:
        log.info("  Using latest epoch checkpoint: %s", epoch_checkpoints[-1].name)
        return str(epoch_checkpoints[-1])

    # Look for last checkpoint
    last_checkpoint = work_path / "last.pth"
    if last_checkpoint.exists():
        log.info("  Using last checkpoint: last.pth")
        return str(last_checkpoint)

    log.warning("  No checkpoints found (looked for: best_coco_AP_epoch_*.pth, epoch_*.pth, last.pth)")
    return None


def resolve_checkpoint_path(config: Dict) -> Optional[str]:
    """
    Resolve checkpoint path based on configuration.
    Returns the checkpoint path to use, or None if training from scratch.
    """
    checkpoint_path = config['model'].get('checkpoint_path')
    auto_checkpoint = config['model'].get('auto_checkpoint', True)
    work_dir = config['paths']['work_dir']
    job_type = config['job']['type']

    # If checkpoint explicitly specified, use it
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_path}")
        log.info("Using specified checkpoint: %s", checkpoint_path)
        return checkpoint_path

    # If inference or both, try to find checkpoint automatically
    if job_type in ['inference', 'both'] and auto_checkpoint:
        found_checkpoint = find_best_checkpoint(work_dir)
        if found_checkpoint:
            log.info("Auto-detected checkpoint: %s", found_checkpoint)
            return found_checkpoint
        elif job_type == 'inference':
            raise FileNotFoundError(
                f"No checkpoint found in {work_dir} for inference. "
                "Please specify checkpoint_path in config or train first."
            )

    # Training from scratch
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Lower body keypoint constants (6 joints: hips, knees, ankles)
# ─────────────────────────────────────────────────────────────────────────────
KEYPOINT_NAMES = [
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
]
NUM_KP = len(KEYPOINT_NAMES)   # 6

# Skeleton connections for lower body (1-indexed)
SKELETON = [
    [1, 2],   # left_hip - right_hip
    [1, 3],   # left_hip - left_knee
    [2, 4],   # right_hip - right_knee
    [3, 5],   # left_knee - left_ankle
    [4, 6],   # right_knee - right_ankle
]

# Sigmas for lower body keypoints (from COCO)
SIGMAS = [
    0.107, 0.107,  # hips
    0.087, 0.087,  # knees
    0.089, 0.089,  # ankles
]

# Flip indices for horizontal flipping (swap left/right)
# When flipping, left_hip↔right_hip, left_knee↔right_knee, left_ankle↔right_ankle
FLIP_INDICES = [
    1, 0,  # left_hip(0) ↔ right_hip(1)
    3, 2,  # left_knee(2) ↔ right_knee(3)
    5, 4,  # left_ankle(4) ↔ right_ankle(5)
]

# Mapping from COCO 17-joint indices to our 6-joint indices
# COCO indices: left_hip=11, right_hip=12, left_knee=13, right_knee=14, left_ankle=15, right_ankle=16
COCO_TO_LOWER_BODY = {
    11: 0,  # left_hip
    12: 1,  # right_hip
    13: 2,  # left_knee
    14: 3,  # right_knee
    15: 4,  # left_ankle
    16: 5,  # right_ankle
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Dataset inspection
# ─────────────────────────────────────────────────────────────────────────────
def inspect_dataset(dataset: fo.Dataset) -> None:
    log.info("━━━ Dataset inspection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  Name         : %s", dataset.name)
    log.info("  Total samples: %d", len(dataset))
    log.info("  Distinct tags: %s", dataset.distinct("tags"))
    log.info("  Field schema :")
    for fname, ftype in dataset.get_field_schema().items():
        log.info("    %-28s %s", fname, ftype)

    sample = dataset.first()
    kp_label = sample.get_field("keypoints")
    if kp_label is not None:
        instances = (
            kp_label.keypoints
            if hasattr(kp_label, "keypoints")
            else [kp_label]
        )
        log.info("  First sample keypoint groups : %d", len(instances))
        if instances:
            kp0 = instances[0]
            log.info(
                "    label='%s'  num_points=%d  has_confidence=%s",
                kp0.label,
                len(kp0.points or []),
                kp0.confidence is not None,
            )
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – FiftyOne native COCO export  +  keypoint patch
# ─────────────────────────────────────────────────────────────────────────────
def export_split_coco(
    view: fo.DatasetView,
    export_dir: str,
    det_field: str = "ground_truth",
    kp_field: str  = "keypoints",
) -> str:
    """
    Export *view* using FiftyOne's COCODetectionDataset exporter, then patch
    the resulting labels.json in-place to add keypoint data.

    FiftyOne writes:
        <export_dir>/
            data/          <- symlinks / copies of images
            labels.json    <- COCO-format annotation file

    After the patch the JSON is MMPose-ready.

    Returns the absolute path to the patched labels.json.
    """
    os.makedirs(export_dir, exist_ok=True)

    # ── 2a. Native FiftyOne COCODetectionDataset export ───────────────
    log.info("  fo.export → %s  (label_field='%s')", export_dir, det_field)
    view.export(
        export_dir=export_dir,
        dataset_type=fot.COCODetectionDataset,
        label_field=det_field,        # exports bboxes + category info
        overwrite=True,
    )

    # FiftyOne writes the JSON as "labels.json" inside export_dir
    ann_path = os.path.join(export_dir, "labels.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"Expected FiftyOne to write {ann_path} but it was not found. "
            "Check that export_dir is writable and det_field is correct."
        )

    with open(ann_path) as f:
        coco = json.load(f)

    # ── 2b. Fix image paths to absolute ───────────────────────────────
    # FiftyOne exports images with relative paths, convert to absolute
    data_dir = os.path.abspath(os.path.join(export_dir, "data"))
    log.info("  Converting image paths to absolute (data_dir=%s)", data_dir)

    fixed_count = 0
    for img in coco.get("images", []):
        old_path = img["file_name"]

        # If already absolute, skip
        if os.path.isabs(old_path):
            continue

        # Convert to absolute path
        filename = os.path.basename(old_path)
        abs_path = os.path.join(data_dir, filename)
        img["file_name"] = abs_path
        fixed_count += 1

        # Log first few to verify
        if fixed_count <= 3:
            log.info("    Example: '%s' -> '%s'", old_path, abs_path)
            if not os.path.exists(abs_path):
                log.warning("    WARNING: File does not exist: %s", abs_path)

    log.info("  ✓ Fixed %d image paths to absolute", fixed_count)

    # ── 2c. Patch categories with keypoint + skeleton metadata ────────
    # Define flip pairs: when flipping horizontally, swap these keypoint indices
    # Format: [[left_idx, right_idx], ...]
    flip_pairs = [
        [0, 1],  # left_hip (0) ↔ right_hip (1)
        [2, 3],  # left_knee (2) ↔ right_knee (3)
        [4, 5],  # left_ankle (4) ↔ right_ankle (5)
    ]

    for cat in coco.get("categories", []):
        if cat.get("name") == "person":
            cat["keypoints"] = KEYPOINT_NAMES
            cat["skeleton"]  = SKELETON
            cat["flip_pairs"] = flip_pairs  # Tell MMPose which keypoints to swap

    # ── 2d. Build lookup: image file_name (basename) → image_id ──────
    #    and  image_id → (width, height)
    fname_to_id: dict[str, int] = {}
    img_meta: dict[int, tuple[int, int]] = {}
    for img in coco.get("images", []):
        basename = os.path.basename(img["file_name"])
        fname_to_id[basename] = img["id"]
        img_meta[img["id"]]   = (img["width"], img["height"])

    # ── 2e. Build lookup: annotation_id → index in coco["annotations"] ─
    ann_idx: dict[int, int] = {
        ann["id"]: i for i, ann in enumerate(coco.get("annotations", []))
    }

    # ── 2f. Group annotation ids by image_id (preserving FiftyOne order)
    from collections import defaultdict
    img_to_ann_ids: dict[int, list[int]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_to_ann_ids[ann["image_id"]].append(ann["id"])

    # ── 2g. Walk samples and inject keypoints ─────────────────────────
    patched = 0
    nan_count = 0
    total_keypoints = 0

    for sample in view.iter_samples(progress=True):
        base   = os.path.basename(sample.filepath)
        img_id = fname_to_id.get(base)
        if img_id is None:
            continue

        W, H = img_meta.get(img_id, (1, 1))

        kp_label  = sample.get_field(kp_field)
        det_label = sample.get_field(det_field)
        if kp_label is None or det_label is None:
            continue

        # Ordered lists – must match the order FiftyOne used for annotation ids
        kp_instances = (
            kp_label.keypoints
            if hasattr(kp_label, "keypoints")
            else [kp_label]
        )

        ann_ids_for_img = img_to_ann_ids.get(img_id, [])

        for i, ann_id in enumerate(ann_ids_for_img):
            idx = ann_idx.get(ann_id)
            if idx is None:
                continue

            kp = kp_instances[i] if i < len(kp_instances) else None
            if kp is None:
                continue

            raw_pts = kp.points or []
            flat: list[float] = []

            # Extract only lower body keypoints from COCO 17-joint format
            # COCO indices: left_hip=11, right_hip=12, left_knee=13, right_knee=14, left_ankle=15, right_ankle=16
            coco_indices = [11, 12, 13, 14, 15, 16]

            for coco_idx in coco_indices:
                total_keypoints += 1
                if coco_idx < len(raw_pts):
                    # Get raw coordinates
                    x_norm = raw_pts[coco_idx][0]
                    y_norm = raw_pts[coco_idx][1]

                    # Check for NaN or invalid values
                    if x_norm is None or y_norm is None or \
                       (isinstance(x_norm, float) and (x_norm != x_norm)) or \
                       (isinstance(y_norm, float) and (y_norm != y_norm)):
                        # NaN or None detected - mark as invisible
                        px, py, v = 0.0, 0.0, 0
                        nan_count += 1
                    else:
                        # Valid coordinates - convert to pixel space
                        px = float(x_norm) * W
                        py = float(y_norm) * H

                        # Use confidence for visibility if available
                        if kp.confidence and coco_idx < len(kp.confidence):
                            conf = kp.confidence[coco_idx]
                            # Check for NaN in confidence too
                            if conf is None or (isinstance(conf, float) and conf != conf):
                                v = 0
                            else:
                                v = 2 if float(conf) > 0.0 else 1
                        else:
                            v = 2
                else:
                    px, py, v = 0.0, 0.0, 0    # pad missing joints

                flat.extend([round(px, 2), round(py, 2), v])

            num_vis = sum(1 for j in range(NUM_KP) if flat[j * 3 + 2] > 0)
            coco["annotations"][idx]["keypoints"]     = flat
            coco["annotations"][idx]["num_keypoints"] = num_vis
            patched += 1

    # ── 2h. Write patched JSON back in place ──────────────────────────
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    log.info(
        "  Patched %d / %d annotations with keypoints → %s",
        patched, len(coco["annotations"]), ann_path,
    )

    # Report NaN statistics
    if nan_count > 0:
        log.warning("  ⚠️  Found %d NaN keypoints out of %d total (%.1f%%)",
                   nan_count, total_keypoints, 100 * nan_count / total_keypoints)
        log.warning("  These keypoints were marked as invisible (v=0)")
    else:
        log.info("  ✓ No NaN keypoints found - all data valid")

    # Verify image paths are absolute in the saved file
    log.info("  Verifying saved image paths...")
    with open(ann_path) as f:
        saved = json.load(f)
        if saved.get("images"):
            first_path = saved["images"][0]["file_name"]
            log.info("    First image path in saved file: %s", first_path)
            if os.path.isabs(first_path):
                log.info("    ✓ Paths are absolute in labels.json")
            else:
                log.error("    ✗ ERROR: Paths are still relative!")

    return ann_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – MMPose Config  (RTMPose-s, SimCC head)
# ─────────────────────────────────────────────────────────────────────────────
def build_config(
    train_ann: str,
    val_ann: str,
    work_dir: str,
    img_size: tuple[int, int] = (192, 256),   # (W, H)
    batch_size: int = 16,
    max_epochs: int = 50,
    lr: float = 5e-4,
    num_workers: int = 4,
    pretrained_checkpoint: Optional[str] = None,  # NEW: for finetuning
    model_size: str = 's',  # NEW: 's', 'm', 'l', or 'x'
) -> Config:

    W, H = img_size
    feat_w, feat_h = W // 32, H // 32   # 6 x 8 for 192 x 256

    # ── RTMPose Model Size Configurations ─────────────────────────────
    # Each model size has different:
    # - deepen_factor: depth of network
    # - widen_factor: width of network
    # - out_channels: output channels from backbone
    # - checkpoint: pretrained weights

    MODEL_CONFIGS = {
        's': {  # Small (default)
            'name': 'RTMPose-s',
            'deepen_factor': 0.167,
            'widen_factor': 0.375,
            'out_channels': 384,  # CSPNeXt-s output
            'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth',
        },
        'm': {  # Medium
            'name': 'RTMPose-m',
            'deepen_factor': 0.33,
            'widen_factor': 0.75,
            'out_channels': 768,  # CSPNeXt-m output
            'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth',
        },
        'l': {  # Large
            'name': 'RTMPose-l',
            'deepen_factor': 0.67,
            'widen_factor': 1.0,
            'out_channels': 1024,  # CSPNeXt-l output
            'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth',
        },
        'x': {  # Extra Large
            'name': 'RTMPose-x',
            'deepen_factor': 1.0,
            'widen_factor': 1.25,
            'out_channels': 1280,  # CSPNeXt-x output
            # RTMPose-x pretrained weights not publicly available
            # Training will start from random initialization for backbone
            'checkpoint': None,  # No pretrained weights available
        },
    }

    # Validate and get model config
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model_size: {model_size}. Must be one of: {list(MODEL_CONFIGS.keys())}")

    model_cfg = MODEL_CONFIGS[model_size]
    log.info(f"Building config for {model_cfg['name']} (deepen={model_cfg['deepen_factor']}, widen={model_cfg['widen_factor']})")

    # Warn if no pretrained weights available
    if model_cfg['checkpoint'] is None:
        log.warning(f"⚠️  No pretrained weights available for {model_cfg['name']}")
        log.warning("   Backbone will be trained from random initialization")
        log.warning("   This will require more epochs and may result in lower accuracy")
        log.warning("   Consider using size 'l' (large) instead for pretrained weights")

    simcc_encoder = dict(
        type="SimCCLabel",
        input_size=(W, H),
        smoothing_type="gaussian",
        sigma=(4.9, 5.66),
        simcc_split_ratio=2.0,
        normalize=False,
        use_dark=False,
    )

    train_pipeline = [
        dict(type="LoadImage"),
        dict(type="GetBBoxCenterScale"),
        # dict(type="RandomFlip", direction="horizontal"),  # Temporarily disabled due to flip_indices bug
        dict(type="RandomHalfBody"),
        dict(type="RandomBBoxTransform", scale_factor=[0.6, 1.4], rotate_factor=80),
        dict(type="TopdownAffine", input_size=(W, H)),
        dict(type="mmdet.YOLOXHSVRandomAug"),
        dict(
            type="Albumentation",
            transforms=[
                dict(type="Blur", p=0.1),
                dict(type="MedianBlur", p=0.1),
                dict(
                    type="CoarseDropout",
                    max_holes=1, max_height=0.4, max_width=0.4,
                    min_holes=1, min_height=0.2, min_width=0.2,
                    p=0.5,
                ),
            ],
        ),
        dict(type="GenerateTarget", encoder=simcc_encoder),
        dict(type="PackPoseInputs"),
    ]

    val_pipeline = [
        dict(type="LoadImage"),
        dict(type="GetBBoxCenterScale"),
        dict(type="TopdownAffine", input_size=(W, H)),
        dict(type="PackPoseInputs"),
    ]

    def _dataloader(ann_file: str, pipeline: list, shuffle: bool, bs: int) -> dict:
        return dict(
            batch_size=bs,
            num_workers=num_workers if shuffle else max(num_workers // 2, 2),
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type="DefaultSampler", shuffle=shuffle),
            dataset=dict(
                type="CocoDataset",
                data_root="",
                data_mode="topdown",
                ann_file=ann_file,
                data_prefix=dict(img=""),   # file_name in JSON is absolute
                pipeline=pipeline,
                # Let MMPose auto-detect metainfo from COCO JSON
            ),
        )

    cfg_dict = dict(
        default_scope="mmpose",

        env_cfg=dict(
            cudnn_benchmark=True,  # Enable for speed (was False)
            mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
            dist_cfg=dict(backend="nccl"),
        ),
        vis_backends=[dict(type="LocalVisBackend")],
        visualizer=dict(
            type="PoseLocalVisualizer",
            vis_backends=[dict(type="LocalVisBackend")],
            name="visualizer",
        ),
        log_processor=dict(type="LogProcessor", window_size=50, by_epoch=True),
        log_level="INFO",
        load_from=pretrained_checkpoint,  # Load pretrained weights if provided
        resume=False,

        # ── RTMPose-s model ───────────────────────────────────────────
        model=dict(
            type="TopdownPoseEstimator",
            data_preprocessor=dict(
                type="PoseDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
            ),
            backbone=dict(
                type="CSPNeXt",
                arch="P5",
                expand_ratio=0.5,
                deepen_factor=model_cfg['deepen_factor'],
                widen_factor=model_cfg['widen_factor'],
                out_indices=(4,),
                channel_attention=True,
                norm_cfg=dict(type="BN"),
                act_cfg=dict(type="SiLU"),
                init_cfg=dict(
                    type="Pretrained",
                    prefix="backbone.",
                    checkpoint=model_cfg['checkpoint'],
                ) if model_cfg['checkpoint'] else None,  # Only load pretrained if available
            ),
            neck=dict(
                type="ChannelMapper",
                in_channels=[model_cfg['out_channels']],  # Use model-specific output channels
                out_channels=256,
                kernel_size=1,
            ),
            head=dict(
                type="RTMCCHead",
                in_channels=256,
                out_channels=NUM_KP,  # 6 keypoints (lower body only)
                input_size=(W, H),
                in_featuremap_size=(feat_w, feat_h),
                simcc_split_ratio=2.0,
                final_layer_kernel_size=7,
                gau_cfg=dict(
                    hidden_dims=256, s=128,
                    expansion_factor=2, dropout_rate=0.0, drop_path=0.0,
                    act_fn="SiLU", use_rel_bias=False, pos_enc=False,
                ),
                loss=dict(
                    type="KLDiscretLoss",
                    use_target_weight=True,
                    beta=10.0,
                    label_softmax=True,
                ),
                decoder=simcc_encoder,
            ),
            test_cfg=dict(flip_test=False),  # Disable TTA flipping due to flip_indices issue
        ),

        # ── Data loaders ─────────────────────────────────────────────
        train_dataloader = _dataloader(train_ann, train_pipeline, shuffle=True,  bs=batch_size),
        val_dataloader   = _dataloader(val_ann,   val_pipeline,   shuffle=False, bs=batch_size),
        test_dataloader  = _dataloader(val_ann,   val_pipeline,   shuffle=False, bs=batch_size),

        # ── OKS evaluator ─────────────────────────────────────────────
        val_evaluator=dict(
            type="CocoMetric",
            ann_file=val_ann,
            nms_mode="none",
            score_mode="bbox",
        ),
        test_evaluator=dict(
            type="CocoMetric",
            ann_file=val_ann,
            nms_mode="none",
            score_mode="bbox",
        ),

        # ── Schedule ──────────────────────────────────────────────────
        train_cfg=dict(by_epoch=True, max_epochs=max_epochs, val_interval=999),  # Disable built-in validation
        val_cfg=dict(),
        test_cfg=dict(),

        optim_wrapper=dict(
            type="OptimWrapper",
            optimizer=dict(type="AdamW", lr=lr, weight_decay=0.05),
            paramwise_cfg=dict(
                norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True,
            ),
            clip_grad=dict(max_norm=35, norm_type=2),
        ),

        param_scheduler=[
            dict(type="LinearLR", begin=0, end=500,
                 start_factor=1e-5, by_epoch=False),
            dict(type="CosineAnnealingLR",
                 eta_min=lr * 0.05,
                 begin=max_epochs // 2, end=max_epochs,
                 T_max=max_epochs // 2,
                 by_epoch=True, convert_to_iter_based=True),
        ],

        # ── Hooks ─────────────────────────────────────────────────────
        default_hooks=dict(
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", interval=50),
            param_scheduler=dict(type="ParamSchedulerHook"),
            checkpoint=dict(
                type="CheckpointHook",
                interval=1,
                save_best="coco/AP",    # highest OKS mAP triggers save
                rule="greater",
                max_keep_ckpts=3,
                save_last=True,
            ),
            sampler_seed=dict(type="DistSamplerSeedHook"),
            visualization=dict(type="PoseVisualizationHook", enable=False),
        ),

        work_dir=work_dir,
    )

    return Config(cfg_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – OKS calculator (for custom validation)
# ─────────────────────────────────────────────────────────────────────────────
def compute_oks(
    pred_kps:  np.ndarray,                       # (N, K, 2)
    gt_kps:    np.ndarray,                       # (N, K, 2)
    gt_vis:    np.ndarray,                       # (N, K)
    gt_areas:  np.ndarray,                       # (N,)
    sigmas:    np.ndarray = np.array(SIGMAS),
) -> np.ndarray:
    """
    OKS_i = sum_k [ exp(-d2_k / (2 * area_i * sigma2_k)) * delta(v_k > 0) ]
            ──────────────────────────────────────────────────────────────────
                              sum_k delta(v_k > 0)
    Returns shape (N,) OKS scores in [0, 1].
    """
    N   = len(gt_kps)
    oks = np.zeros(N, dtype=np.float32)
    for i in range(N):
        s2 = float(gt_areas[i])
        if s2 <= 0:
            continue
        d2  = np.sum((pred_kps[i] - gt_kps[i]) ** 2, axis=-1)
        e   = d2 / (2.0 * s2 * sigmas ** 2 + 1e-9)
        vis = gt_vis[i] > 0
        if vis.sum() == 0:
            continue
        oks[i] = (np.exp(-e) * vis).sum() / vis.sum()
    return oks


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Comprehensive OKS validation
# ─────────────────────────────────────────────────────────────────────────────
class OKSValidator:
    """Comprehensive OKS performance validation and reporting."""

    def __init__(
        self,
        model,
        config: Config,
        ann_file: str,
        work_dir: str,
        sigmas: np.ndarray = np.array(SIGMAS),
    ):
        self.model = model
        self.config = config
        self.ann_file = ann_file
        self.work_dir = work_dir
        self.sigmas = sigmas

        # Load ground truth annotations
        with open(ann_file) as f:
            self.coco_data = json.load(f)

        # Build lookups
        self.img_id_to_path = {
            img['id']: img['file_name']
            for img in self.coco_data['images']
        }
        self.img_id_to_dims = {
            img['id']: (img['width'], img['height'])
            for img in self.coco_data['images']
        }

        # Group annotations by image
        self.img_to_anns = defaultdict(list)
        for ann in self.coco_data['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)

    def run_inference(self, max_samples: Optional[int] = None) -> Dict:
        """Run inference on validation set and collect predictions."""
        log.info("━━━ Running inference for OKS validation ━━━━━━━━━━━")

        predictions = []
        ground_truths = []
        oks_scores = []
        per_keypoint_errors = defaultdict(list)

        img_ids = list(self.img_to_anns.keys())
        if max_samples:
            img_ids = img_ids[:max_samples]

        for idx, img_id in enumerate(img_ids):
            if (idx + 1) % 100 == 0:
                log.info(f"  Processed {idx + 1}/{len(img_ids)} images")

            img_path = self.img_id_to_path[img_id]
            anns = self.img_to_anns[img_id]

            if not anns:
                continue

            W, H = self.img_id_to_dims[img_id]

            # Get bounding boxes from annotations
            bboxes = []
            for ann in anns:
                bbox = ann['bbox']  # [x, y, w, h]
                # Convert to [x1, y1, x2, y2]
                bboxes.append([
                    bbox[0], bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3]
                ])

            if not bboxes:
                continue

            # Check if image file exists
            if not os.path.exists(img_path):
                log.warning(f"  Image file not found: {img_path}")
                continue

            # Run inference
            try:
                # inference_topdown expects bboxes as list of [x1,y1,x2,y2] or numpy arrays
                results = inference_topdown(
                    self.model,
                    img_path,
                    bboxes=np.array(bboxes)  # Convert to numpy array
                )
            except Exception as e:
                import traceback
                log.warning(f"  Inference failed for {img_path}")
                log.warning(f"  Error type: {type(e).__name__}")
                log.warning(f"  Error message: {str(e)}")
                if idx < 5:  # Show full traceback for first 5 errors
                    log.warning(f"  Traceback:\n{traceback.format_exc()}")
                continue

            # Process each person annotation
            for i, (ann, result) in enumerate(zip(anns, results)):
                # Extract ground truth - should already be 6 keypoints from our export
                gt_kps_flat = np.array(ann['keypoints']).reshape(-1, 3)
                gt_kps = gt_kps_flat[:, :2]
                gt_vis = gt_kps_flat[:, 2]

                # Verify we have 6 keypoints
                if len(gt_kps) != NUM_KP:
                    log.warning(f"  Unexpected number of GT keypoints: {len(gt_kps)}, expected {NUM_KP}")
                    continue

                # Compute area from bbox
                bbox = ann['bbox']
                gt_area = bbox[2] * bbox[3]

                # Extract prediction
                pred_instances = result.pred_instances
                if len(pred_instances.keypoints) == 0:
                    continue

                # Model predicts 6 keypoints (lower body only)
                # Handle both torch tensors and numpy arrays
                pred_kps_raw = pred_instances.keypoints[0]
                pred_scores_raw = pred_instances.keypoint_scores[0]

                # Convert to numpy if needed
                if hasattr(pred_kps_raw, 'cpu'):
                    pred_kps = pred_kps_raw.cpu().numpy()  # (6, 2)
                    pred_scores = pred_scores_raw.cpu().numpy()  # (6,)
                else:
                    pred_kps = np.array(pred_kps_raw)  # Already numpy
                    pred_scores = np.array(pred_scores_raw)

                # Verify prediction shape
                if len(pred_kps) != NUM_KP:
                    log.warning(f"  Unexpected number of predicted keypoints: {len(pred_kps)}, expected {NUM_KP}")
                    continue

                # Compute OKS
                oks = compute_oks(
                    pred_kps[np.newaxis],
                    gt_kps[np.newaxis],
                    gt_vis[np.newaxis],
                    np.array([gt_area]),
                    self.sigmas
                )[0]

                oks_scores.append(oks)
                predictions.append(pred_kps)
                ground_truths.append((gt_kps, gt_vis, gt_area))

                # Compute per-keypoint errors
                for kp_idx in range(NUM_KP):
                    if gt_vis[kp_idx] > 0:
                        error = np.linalg.norm(pred_kps[kp_idx] - gt_kps[kp_idx])
                        per_keypoint_errors[kp_idx].append(error)

        log.info(f"  Completed inference on {len(predictions)} instances")

        return {
            'predictions': predictions,
            'ground_truths': ground_truths,
            'oks_scores': np.array(oks_scores),
            'per_keypoint_errors': per_keypoint_errors,
        }

    def compute_metrics(self, results: Dict) -> Dict:
        """Compute comprehensive OKS metrics."""
        log.info("━━━ Computing OKS metrics ━━━━━━━━━━━━━━━━━━━━━━━━━")

        oks_scores = results['oks_scores']

        # Check if we have any predictions
        if len(oks_scores) == 0:
            log.error("No predictions generated! Cannot compute metrics.")
            log.error("This could mean:")
            log.error("  1. The model hasn't been trained yet")
            log.error("  2. No valid detections in the validation set")
            log.error("  3. Issue with the checkpoint file")

            # Return dict with all expected keys (zeros)
            metrics = {
                'error': 'No predictions generated',
                'mean_oks': 0.0,
                'median_oks': 0.0,
                'std_oks': 0.0,
                'min_oks': 0.0,
                'max_oks': 0.0,
                'mAP': 0.0,
                # F1 scores
                'precision@0.50': 0.0,
                'recall@0.50': 0.0,
                'F1@0.50': 0.0,
                'precision@0.75': 0.0,
                'recall@0.75': 0.0,
                'F1@0.75': 0.0,
                'precision@0.90': 0.0,
                'recall@0.90': 0.0,
                'F1@0.90': 0.0,
                'mean_F1': 0.0,
                # Performance breakdown counts
                'excellent_count': 0,
                'good_count': 0,
                'fair_count': 0,
                'poor_count': 0,
                'total_count': 0,
                # Performance breakdown percentages
                'excellent_pct': 0.0,
                'good_pct': 0.0,
                'fair_pct': 0.0,
                'poor_pct': 0.0,
            }
            # Add AP thresholds
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            for thresh in thresholds:
                metrics[f'AP@{thresh:.2f}'] = 0.0

            # Add per-keypoint errors (all zeros)
            for i, kp_name in enumerate(KEYPOINT_NAMES):
                metrics[f'{kp_name}_error'] = 0.0

            return metrics

        log.info("Computing metrics for %d predictions...", len(oks_scores))

        metrics = {
            'mean_oks': float(np.mean(oks_scores)),
            'median_oks': float(np.median(oks_scores)),
            'std_oks': float(np.std(oks_scores)),
            'min_oks': float(np.min(oks_scores)),
            'max_oks': float(np.max(oks_scores)),
        }

        # Compute AP at different OKS thresholds
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        for thresh in thresholds:
            ap = np.mean(oks_scores >= thresh)
            metrics[f'AP@{thresh:.2f}'] = float(ap)

        # Compute mAP (average over thresholds 0.5:0.05:0.95)
        aps = [np.mean(oks_scores >= t/100) for t in range(50, 100, 5)]
        metrics['mAP'] = float(np.mean(aps))

        # Compute Precision, Recall, and F1 Score at different thresholds
        # Standard COCO thresholds for pose estimation
        f1_thresholds = [0.5, 0.75, 0.9]

        for thresh in f1_thresholds:
            # True Positives: predictions with OKS >= threshold
            tp = np.sum(oks_scores >= thresh)
            # False Positives: predictions with OKS < threshold
            fp = np.sum(oks_scores < thresh)
            # False Negatives: ground truth instances without predictions
            # (approximated as same as total predictions for this implementation)
            fn = len(oks_scores) - tp

            # Precision: TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall: TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[f'precision@{thresh:.2f}'] = float(precision)
            metrics[f'recall@{thresh:.2f}'] = float(recall)
            metrics[f'F1@{thresh:.2f}'] = float(f1)

        # Compute mean F1 score across thresholds
        f1_scores = [metrics[f'F1@{t:.2f}'] for t in f1_thresholds]
        metrics['mean_F1'] = float(np.mean(f1_scores))

        # Per-keypoint average errors
        per_kp_errors = results['per_keypoint_errors']
        for kp_idx, errors in per_kp_errors.items():
            if errors:
                kp_name = KEYPOINT_NAMES[kp_idx]
                metrics[f'mean_error_{kp_name}'] = float(np.mean(errors))
                metrics[f'median_error_{kp_name}'] = float(np.median(errors))

        # Performance by OKS range
        ranges = [
            ('excellent', 0.9, 1.0),
            ('good', 0.75, 0.9),
            ('fair', 0.5, 0.75),
            ('poor', 0.0, 0.5),
        ]

        for name, low, high in ranges:
            count = np.sum((oks_scores >= low) & (oks_scores < high))
            pct = 100 * count / len(oks_scores)
            metrics[f'{name}_count'] = int(count)
            metrics[f'{name}_pct'] = float(pct)

        # Log summary
        log.info(f"  Mean OKS:   {metrics['mean_oks']:.4f}")
        log.info(f"  Median OKS: {metrics['median_oks']:.4f}")
        log.info(f"  mAP:        {metrics['mAP']:.4f}")
        log.info(f"  AP@0.50:    {metrics['AP@0.50']:.4f}")
        log.info(f"  AP@0.75:    {metrics['AP@0.75']:.4f}")
        log.info(f"  F1@0.50:    {metrics['F1@0.50']:.4f}")
        log.info(f"  F1@0.75:    {metrics['F1@0.75']:.4f}")
        log.info(f"  Mean F1:    {metrics['mean_F1']:.4f}")

        return metrics

    def generate_visualizations(self, results: Dict, metrics: Dict):
        """Generate visualization plots for OKS analysis."""
        log.info("━━━ Generating visualizations ━━━━━━━━━━━━━━━━━━━━━")

        viz_dir = Path(self.work_dir) / "oks_validation"
        viz_dir.mkdir(parents=True, exist_ok=True)

        oks_scores = results['oks_scores']
        per_kp_errors = results['per_keypoint_errors']

        # Check if we have any valid data to visualize
        if len(oks_scores) == 0 or metrics.get('error'):
            log.warning("No valid predictions to visualize - skipping visualization generation")
            log.info("Metrics summary: mean_oks={:.3f}, mAP={:.3f}".format(
                metrics.get('mean_oks', 0.0),
                metrics.get('mAP', 0.0)
            ))
            return

        log.info("Generating visualizations for %d predictions...", len(oks_scores))

        # Set style
        sns.set_style("whitegrid")

        # 1. OKS distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(oks_scores, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(metrics['mean_oks'], color='r', linestyle='--',
                   label=f"Mean: {metrics['mean_oks']:.3f}")
        plt.axvline(metrics['median_oks'], color='g', linestyle='--',
                   label=f"Median: {metrics['median_oks']:.3f}")
        plt.xlabel('OKS Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of OKS Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "oks_distribution.png", dpi=150)
        plt.close()

        # 2. Cumulative OKS curve
        plt.figure(figsize=(10, 6))
        sorted_oks = np.sort(oks_scores)
        cumulative = np.arange(1, len(sorted_oks) + 1) / len(sorted_oks)
        plt.plot(sorted_oks, cumulative, linewidth=2)
        plt.xlabel('OKS Score')
        plt.ylabel('Cumulative Proportion')
        plt.title('Cumulative Distribution of OKS Scores')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / "oks_cumulative.png", dpi=150)
        plt.close()

        # 3. Per-keypoint error analysis
        kp_names = []
        kp_mean_errors = []
        kp_median_errors = []

        for kp_idx in range(NUM_KP):
            if kp_idx in per_kp_errors and per_kp_errors[kp_idx]:
                kp_names.append(KEYPOINT_NAMES[kp_idx])
                kp_mean_errors.append(np.mean(per_kp_errors[kp_idx]))
                kp_median_errors.append(np.median(per_kp_errors[kp_idx]))

        plt.figure(figsize=(14, 6))
        x = np.arange(len(kp_names))
        width = 0.35
        plt.bar(x - width/2, kp_mean_errors, width, label='Mean Error', alpha=0.8)
        plt.bar(x + width/2, kp_median_errors, width, label='Median Error', alpha=0.8)
        plt.xlabel('Keypoint')
        plt.ylabel('Error (pixels)')
        plt.title('Per-Keypoint Prediction Error')
        plt.xticks(x, kp_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / "per_keypoint_errors.png", dpi=150)
        plt.close()

        # 4. AP at different thresholds
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        aps = [metrics[f'AP@{t:.2f}'] for t in thresholds]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, aps, marker='o', linewidth=2, markersize=8)
        plt.xlabel('OKS Threshold')
        plt.ylabel('Average Precision')
        plt.title('AP vs OKS Threshold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / "ap_vs_threshold.png", dpi=150)
        plt.close()

        # 5. Performance breakdown pie chart
        labels = ['Excellent\n(≥0.9)', 'Good\n(0.75-0.9)',
                 'Fair\n(0.5-0.75)', 'Poor\n(<0.5)']
        sizes = [
            metrics['excellent_count'],
            metrics['good_count'],
            metrics['fair_count'],
            metrics['poor_count'],
        ]
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12})
        plt.title('OKS Performance Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_breakdown.png", dpi=150)
        plt.close()

        # 6. F1, Precision, Recall comparison
        f1_thresholds = [0.5, 0.75, 0.9]
        f1_scores = [metrics[f'F1@{t:.2f}'] for t in f1_thresholds]
        precision_scores = [metrics[f'precision@{t:.2f}'] for t in f1_thresholds]
        recall_scores = [metrics[f'recall@{t:.2f}'] for t in f1_thresholds]

        x = np.arange(len(f1_thresholds))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)

        plt.xlabel('OKS Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall, and F1 Score at Different OKS Thresholds')
        plt.xticks(x, [f'{t:.2f}' for t in f1_thresholds])
        plt.ylim([0, 1.0])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(viz_dir / "f1_precision_recall.png", dpi=150)
        plt.close()

        log.info(f"  Visualizations saved to: {viz_dir}")

    def save_report(self, metrics: Dict):
        """Save detailed metrics report."""
        report_path = Path(self.work_dir) / "oks_validation" / "metrics_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        log.info(f"  Metrics report saved to: {report_path}")

        # Also save human-readable text report
        txt_path = report_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("OKS PERFORMANCE VALIDATION REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("OVERALL METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean OKS:        {metrics['mean_oks']:.4f}\n")
            f.write(f"Median OKS:      {metrics['median_oks']:.4f}\n")
            f.write(f"Std Dev:         {metrics['std_oks']:.4f}\n")
            f.write(f"Min OKS:         {metrics['min_oks']:.4f}\n")
            f.write(f"Max OKS:         {metrics['max_oks']:.4f}\n")
            f.write(f"mAP (0.5:0.95):  {metrics['mAP']:.4f}\n")
            f.write(f"Mean F1:         {metrics['mean_F1']:.4f}\n\n")

            f.write("F1 SCORES, PRECISION, AND RECALL\n")
            f.write("-" * 70 + "\n")
            f.write(f"F1@0.50:         {metrics['F1@0.50']:.4f}  (P={metrics['precision@0.50']:.4f}, R={metrics['recall@0.50']:.4f})\n")
            f.write(f"F1@0.75:         {metrics['F1@0.75']:.4f}  (P={metrics['precision@0.75']:.4f}, R={metrics['recall@0.75']:.4f})\n")
            f.write(f"F1@0.90:         {metrics['F1@0.90']:.4f}  (P={metrics['precision@0.90']:.4f}, R={metrics['recall@0.90']:.4f})\n\n")

            f.write("AVERAGE PRECISION AT DIFFERENT THRESHOLDS\n")
            f.write("-" * 70 + "\n")
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            for thresh in thresholds:
                f.write(f"AP@{thresh:.2f}:  {metrics[f'AP@{thresh:.2f}']:.4f}\n")
            f.write("\n")

            f.write("PERFORMANCE DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Excellent (≥0.9):    {metrics['excellent_count']:5d} ({metrics['excellent_pct']:5.1f}%)\n")
            f.write(f"Good (0.75-0.9):     {metrics['good_count']:5d} ({metrics['good_pct']:5.1f}%)\n")
            f.write(f"Fair (0.5-0.75):     {metrics['fair_count']:5d} ({metrics['fair_pct']:5.1f}%)\n")
            f.write(f"Poor (<0.5):         {metrics['poor_count']:5d} ({metrics['poor_pct']:5.1f}%)\n")
            f.write("\n")

            f.write("PER-KEYPOINT MEAN ERRORS (pixels)\n")
            f.write("-" * 70 + "\n")
            for kp_name in KEYPOINT_NAMES:
                key = f'mean_error_{kp_name}'
                if key in metrics:
                    f.write(f"{kp_name:20s}: {metrics[key]:7.2f}\n")

        log.info(f"  Text report saved to: {txt_path}")


def validate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    val_ann: str,
    work_dir: str,
    max_samples: Optional[int] = None,
):
    """Run comprehensive OKS validation on a trained checkpoint."""
    log.info("━━━ Loading checkpoint for validation ━━━━━━━━━━━━━━━")
    log.info(f"  Checkpoint: {checkpoint_path}")
    log.info(f"  Config:     {config_path}")

    # Initialize model
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    config = Config.fromfile(config_path)

    # Create validator
    validator = OKSValidator(
        model=model,
        config=config,
        ann_file=val_ann,
        work_dir=work_dir,
    )

    # Run inference
    results = validator.run_inference(max_samples=max_samples)

    # Compute metrics
    metrics = validator.compute_metrics(results)

    # Generate visualizations
    validator.generate_visualizations(results, metrics)

    # Save report
    validator.save_report(metrics)

    log.info("━━━ OKS validation complete! ━━━━━━━━━━━━━━━━━━━━━━━")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(config_path: str = "config.yaml") -> None:
    """
    Main training/inference pipeline controlled by configuration.

    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)

    # Extract configuration
    dataset_name = config['dataset']['name']
    det_field = config['dataset']['detection_field']
    kp_field = config['dataset']['keypoint_field']
    train_tag = config['dataset']['train_tag']
    val_tag = config['dataset']['validation_tag']
    val_split = config['dataset']['validation_split']

    job_type = config['job']['type']
    skip_export = config['job']['skip_export']
    max_val_samples = config['job']['max_validation_samples']

    # Construct dataset-specific paths
    output_root = config['paths'].get('output_root', './output')
    data_dir = os.path.join(output_root, dataset_name, "data")
    model_dir = os.path.join(output_root, dataset_name, "models")
    validation_dir = os.path.join(model_dir, config['paths'].get('validation_dir', 'validation_results'))

    # Legacy path support (for backward compatibility)
    if 'export_root' in config['paths']:
        export_root = config['paths']['export_root']
        work_dir = config['paths']['work_dir']
        log.warning("Using legacy paths (export_root, work_dir). Consider updating to output_root structure.")
    else:
        export_root = data_dir
        work_dir = model_dir

    img_w = config['training']['image_size']['width']
    img_h = config['training']['image_size']['height']
    batch_size = config['training']['batch_size']
    max_epochs = config['training']['max_epochs']
    lr = config['training']['learning_rate']
    num_workers = config['training']['num_workers']

    # Model size configuration
    model_size = config.get('model', {}).get('size', 's').lower()
    if model_size not in ['s', 'm', 'l', 'x']:
        log.warning(f"Invalid model size '{model_size}', defaulting to 's'")
        model_size = 's'

    # Update model_dir to include size subfolder
    # Structure: ./output/{dataset-name}/models/{size}/
    model_dir = os.path.join(model_dir, model_size)
    validation_dir = os.path.join(model_dir, config['paths'].get('validation_dir', 'validation_results'))

    # Update work_dir for legacy compatibility
    if 'export_root' not in config['paths']:
        work_dir = model_dir

    run_validation = config['validation']['run_after_training']

    # Validate job type
    if job_type not in ['train', 'inference', 'both']:
        raise ValueError(
            f"Invalid job type: {job_type}. Must be 'train', 'inference', or 'both'"
        )

    # Determine what to run
    should_train = job_type in ['train', 'both']
    should_infer = job_type in ['inference', 'both']

    log.info("\n" + "=" * 70)
    log.info("PATH CONFIGURATION")
    log.info("=" * 70)
    log.info("  Dataset: %s", dataset_name)
    log.info("  Model size: %s", model_size.upper())
    log.info("  Data directory:  %s", data_dir)
    log.info("  Model directory: %s", model_dir)
    log.info("  Validation dir:  %s", validation_dir)
    log.info("=" * 70)
    log.info("\n" + "=" * 70)
    log.info("JOB CONFIGURATION")
    log.info("=" * 70)
    log.info("Job type: %s", job_type)
    log.info("  Training: %s", "YES" if should_train else "NO")
    log.info("  Inference: %s", "YES" if should_infer else "NO")
    log.info("=" * 70 + "\n")

    # 1. Load dataset ─────────────────────────────────────────────────
    log.info("Loading FiftyOne dataset: %s", dataset_name)
    try:
        dataset = fo.load_dataset(dataset_name)
    except ValueError:
        log.error(f"Dataset '{dataset_name}' not found in FiftyOne.")
        log.error("Available datasets:")
        for ds_name in fo.list_datasets():
            log.error(f"  - {ds_name}")
        sys.exit(1)

    inspect_dataset(dataset)

    # 2. Split dataset ────────────────────────────────────────────────
    train_view = dataset.match_tags(train_tag)
    val_view = dataset.match_tags(val_tag)

    if len(train_view) == 0:
        raise RuntimeError(
            f"No samples tagged '{train_tag}'. "
            f"Run: dataset.take(N).tag_samples('{train_tag}') to tag them first."
        )

    if len(val_view) == 0:
        log.warning("No '%s' tag found – using %.0f%% of train as validation.",
                   val_tag, val_split * 100)
        n_val = max(1, int(len(train_view) * val_split))
        val_view = train_view.take(n_val)
        train_view = train_view.skip(n_val)

    log.info("Train: %d samples  |  Val: %d samples", len(train_view), len(val_view))

    # 3. Export data (if needed) ──────────────────────────────────────
    train_ann = os.path.join(export_root, "train", "labels.json")
    val_ann = os.path.join(export_root, "val", "labels.json")

    if should_train and not skip_export:
        log.info("Exporting train split via COCODetectionDataset …")
        train_ann = export_split_coco(
            view=train_view,
            export_dir=os.path.join(export_root, "train"),
            det_field=det_field,
            kp_field=kp_field,
        )

        log.info("Exporting val split via COCODetectionDataset …")
        val_ann = export_split_coco(
            view=val_view,
            export_dir=os.path.join(export_root, "val"),
            det_field=det_field,
            kp_field=kp_field,
        )
    elif should_train:
        log.info("Skipping export (skip_export=True), using existing data")
        if not os.path.exists(train_ann) or not os.path.exists(val_ann):
            raise FileNotFoundError(
                f"skip_export=True but exported data not found at:\n"
                f"  {train_ann}\n  {val_ann}\n"
                "Run with skip_export=false first."
            )

    # 4. Build MMPose config (if training) ────────────────────────────
    checkpoint_path = None

    if should_train:
        # Resolve pretrained checkpoint for finetuning
        pretrained_checkpoint = resolve_pretrained_checkpoint(config)

        if pretrained_checkpoint:
            log.info("\n" + "=" * 70)
            log.info("FINETUNING MODE")
            log.info("=" * 70)
            log.info("  Pretrained checkpoint: %s", pretrained_checkpoint)

            # Adjust learning rate if specified
            lr_scale = config.get('finetune', {}).get('learning_rate_scale', 1.0)
            if lr_scale != 1.0:
                lr = lr * lr_scale
                log.info("  Adjusted learning rate: %.6f (scale=%.2f)", lr, lr_scale)

            freeze_backbone = config.get('finetune', {}).get('freeze_backbone', False)
            if freeze_backbone:
                log.info("  Backbone freezing: ENABLED (Note: not yet implemented)")
            log.info("=" * 70 + "\n")
        else:
            log.info("\n" + "=" * 70)
            log.info("TRAINING FROM SCRATCH")
            log.info("=" * 70 + "\n")

        log.info("Building MMPose config …")
        cfg = build_config(
            train_ann=train_ann,
            val_ann=val_ann,
            work_dir=work_dir,
            img_size=(img_w, img_h),
            batch_size=batch_size,
            max_epochs=max_epochs,
            lr=lr,
            num_workers=num_workers,
            pretrained_checkpoint=pretrained_checkpoint,  # Pass pretrained checkpoint
            model_size=model_size,  # Pass model size
        )

        # 5. Train ────────────────────────────────────────────────────
        log.info("Starting training  ->  work_dir: %s", work_dir)
        runner = Runner.from_cfg(cfg)
        runner.train()

        # 6. Find and save best checkpoint ────────────────────────────
        checkpoint_path = find_best_checkpoint(work_dir)
        if checkpoint_path:
            # Save as best.pt
            best_path = os.path.join(work_dir, "best.pt")
            if not os.path.exists(best_path) or not checkpoint_path.endswith("best.pt"):
                import shutil
                shutil.copy(checkpoint_path, best_path)
                log.info("✓ Saved best model to: %s", best_path)
            log.info("Training complete! Best checkpoint: %s", best_path)
            checkpoint_path = best_path
        else:
            log.warning("No checkpoint found after training")

    # 7. Run inference and validation ─────────────────────────────────
    if should_infer:
        # Resolve checkpoint path
        if checkpoint_path is None:
            checkpoint_path = resolve_checkpoint_path(config)

        if checkpoint_path and os.path.exists(checkpoint_path):
            log.info("\n" + "=" * 70)
            log.info("STARTING INFERENCE AND VALIDATION")
            log.info("=" * 70 + "\n")

            # Get or create config file
            config_file = os.path.join(work_dir, "config.py")

            if not os.path.exists(config_file):
                # Config doesn't exist - need to rebuild it
                log.info("Config file not found, rebuilding MMPose config...")
                cfg = build_config(
                    train_ann=train_ann,
                    val_ann=val_ann,
                    work_dir=work_dir,
                    img_size=(img_w, img_h),
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    lr=lr,
                    num_workers=num_workers,
                    model_size=model_size,  # Pass model size
                )
                cfg.dump(config_file)
                log.info("Config saved to: %s", config_file)

            metrics = validate_checkpoint(
                checkpoint_path=checkpoint_path,
                config_path=config_file,
                val_ann=val_ann,
                work_dir=work_dir,
                max_samples=max_val_samples,
            )

            log.info("\n" + "=" * 70)
            log.info("VALIDATION SUMMARY")
            log.info("=" * 70)
            log.info("Checkpoint: %s", os.path.basename(checkpoint_path))
            log.info("Mean OKS:   %.4f", metrics['mean_oks'])
            log.info("Median OKS: %.4f", metrics['median_oks'])
            log.info("mAP:        %.4f", metrics['mAP'])
            log.info("AP@0.50:    %.4f", metrics['AP@0.50'])
            log.info("AP@0.75:    %.4f", metrics['AP@0.75'])
            log.info("F1@0.50:    %.4f", metrics['F1@0.50'])
            log.info("F1@0.75:    %.4f", metrics['F1@0.75'])
            log.info("Mean F1:    %.4f", metrics['mean_F1'])
            log.info("=" * 70 + "\n")
        else:
            log.warning("No checkpoint available for inference")
            if job_type == 'inference':
                log.error("Cannot run inference without checkpoint!")
                sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Train/Inference RTMPose on FiftyOne dataset with OKS validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and validate using config file
  python train_mmpose_with_oks_validation.py --config config.yaml

  # Override config with command line args
  python train_mmpose_with_oks_validation.py --config config.yaml --job-type inference

  # Quick test run
  python train_mmpose_with_oks_validation.py --config config.yaml --max-val-samples 100

Job Types:
  train      : Export data + train model + validate
  inference  : Only run inference on existing checkpoint
  both       : Full pipeline (export + train + inference + validate)
        """
    )

    # Primary input: configuration file
    p.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )

    # Optional overrides (override config file if specified)
    p.add_argument(
        "--dataset",
        type=str,
        help="FiftyOne dataset name (overrides config)"
    )
    p.add_argument(
        "--job-type",
        type=str,
        choices=['train', 'inference', 'both'],
        help="Job type: train, inference, or both (overrides config)"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for inference (overrides config)"
    )
    p.add_argument(
        "--work-dir",
        type=str,
        help="Working directory for outputs (overrides config)"
    )
    p.add_argument(
        "--max-val-samples",
        type=int,
        help="Limit validation to N samples (overrides config)"
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip data export, use existing (overrides config)"
    )

    # Backward compatibility flags (map to job types)
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="DEPRECATED: Use --job-type inference instead"
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="DEPRECATED: Use --job-type inference instead"
    )

    args = p.parse_args()

    # Load base configuration
    if not os.path.exists(args.config):
        log.error(f"Configuration file not found: {args.config}")
        log.error("Please create a config.yaml file or specify --config path")
        sys.exit(1)

    config = load_config(args.config)

    # Apply command-line overrides
    if args.dataset:
        config['dataset']['name'] = args.dataset
        log.info("Override: dataset = %s", args.dataset)

    if args.job_type:
        config['job']['type'] = args.job_type
        log.info("Override: job_type = %s", args.job_type)
    elif args.skip_training or args.validate_only:
        log.warning("--skip-training and --validate-only are deprecated")
        log.warning("Use --job-type inference instead")
        config['job']['type'] = 'inference'

    if args.checkpoint:
        config['model']['checkpoint_path'] = args.checkpoint
        log.info("Override: checkpoint_path = %s", args.checkpoint)

    if args.work_dir:
        config['paths']['work_dir'] = args.work_dir
        log.info("Override: work_dir = %s", args.work_dir)

    if args.max_val_samples is not None:
        config['job']['max_validation_samples'] = args.max_val_samples
        log.info("Override: max_validation_samples = %d", args.max_val_samples)

    if args.skip_export:
        config['job']['skip_export'] = True
        log.info("Override: skip_export = True")

    # Run main pipeline
    try:
        main(args.config)
    except KeyboardInterrupt:
        log.info("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        log.error(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)