"""
FiftyOne-native keypoint dataset for RF-DETR keypoint training.

Design for DDP safety
---------------------
FiftyOne (MongoDB) cannot be accessed from DDP worker processes spawned via
mp.spawn.  This module solves that in two steps:

  Step 1 — prefetch (runs in the MAIN process, before mp.spawn):
      data = prefetch_fo_split(fo_view, detection_field, keypoint_field, ...)
      # Returns a plain Python list of dicts — fully picklable.

  Step 2 — dataset (runs inside EACH worker, no FiftyOne needed):
      ds = FiftyOneKeypointDataset(prefetched=data, ...)
      # Only reads image files; no MongoDB connection required.

Quick start
-----------
    from workflows.keypoint.fo_keypoint_dataset import (
        prefetch_fo_split, FiftyOneKeypointDataset, fo_kp_collate_fn
    )

    train_data = prefetch_fo_split(dataset.match_tags("train"), ...)
    val_data   = prefetch_fo_split(dataset.match_tags("val"),   ...)

    # Inside each DDP worker:
    train_ds = FiftyOneKeypointDataset(prefetched=train_data, augment=True,  ...)
    val_ds   = FiftyOneKeypointDataset(prefetched=val_data,   augment=False, ...)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — prefetch (main process only)
# ─────────────────────────────────────────────────────────────────────────────

def prefetch_fo_split(
    fo_view,
    detection_field: str = "ground_truth",
    keypoint_field:  str = "pedestrian_points",
    target_label:    str = "pedestrian",
    num_keypoints:   int = 1,
    conf_field:      Optional[str] = "confidence",
) -> List[Dict]:
    """
    Walk a FiftyOne view and extract all detection + keypoint data into a
    plain Python list that can be pickled and passed to DDP workers.

    Each element of the returned list is a dict:
      {
        "filepath": str,
        "instances": [
          {
            "box":  [x, y, w, h],          # FiftyOne normalised [0,1]
            "kp":   [[x, y, vis], ...],     # K keypoints, normalised [0,1]
          },
          ...
        ]
      }

    Keypoints are matched to bboxes spatially: the keypoint whose centre
    lies inside the bbox (or up to 30 % below its bottom edge) is assigned
    to that pedestrian.  Pedestrians without a matching keypoint get
    vis=0 at the bbox centre.
    """
    records = []
    n_matched = 0
    n_unmatched = 0

    for sample in fo_view.iter_samples(progress=True, autosave=False):
        filepath = sample.filepath

        # ── Pedestrian bboxes ─────────────────────────────────────────
        raw_boxes: List[List[float]] = []
        det_field_val = getattr(sample, detection_field, None)
        if det_field_val is not None:
            for det in det_field_val.detections:
                if det.label == target_label:
                    raw_boxes.append(list(det.bounding_box))  # [x,y,w,h] norm

        # ── Keypoints ─────────────────────────────────────────────────
        raw_kps:    List[List[float]] = []   # [[x,y], ...]
        raw_confs:  List[float]       = []

        kp_field_val = getattr(sample, keypoint_field, None)
        if kp_field_val is not None:
            for kp_ann in kp_field_val.keypoints:
                if kp_ann.points and len(kp_ann.points) > 0:
                    pt = kp_ann.points[0]
                    raw_kps.append([float(pt[0]), float(pt[1])])
                    conf = 1.0
                    if conf_field:
                        c = getattr(kp_ann, conf_field, None)
                        if c is not None:
                            conf = float(c[0]) if hasattr(c, "__len__") else float(c)
                    raw_confs.append(conf)

        # ── Match bbox → keypoint ─────────────────────────────────────
        instances = []
        used = set()
        for bx, by, bw, bh in raw_boxes:
            margin_y = bh * 0.3   # allow ankle to sit slightly below box
            best_idx, best_dist = -1, float("inf")
            for ki, (kx, ky) in enumerate(raw_kps):
                if ki in used:
                    continue
                if bx <= kx <= bx + bw and by <= ky <= by + bh + margin_y:
                    cx_b = bx + bw / 2
                    cy_b = by + bh / 2
                    d = (kx - cx_b) ** 2 + (ky - cy_b) ** 2
                    if d < best_dist:
                        best_dist, best_idx = d, ki

            kp_list = []
            for k in range(num_keypoints):
                if k == 0 and best_idx >= 0:
                    kx, ky = raw_kps[best_idx]
                    # FIX: put 1.0 first so NaN confidence doesn't propagate.
                    # Python's max(nan, 1.0)=nan (NaN comparison is always False),
                    # but max(1.0, nan)=1.0. A NaN vis causes vis_mask=(nan>0)=False,
                    # which silently disables both OKS/PCK metrics AND the kp XY loss.
                    vis = max(1.0, raw_confs[best_idx])
                    kp_list.append([kx, ky, vis])
                    used.add(best_idx)
                    n_matched += 1
                else:
                    # No explicit annotation — fall back to bottom-centre of the
                    # bbox (ankle_center ≈ foot contact point for upright pedestrians).
                    # vis=1.0 so OKS/PCK are still computed; the model learns to
                    # predict the bottom-centre when no finer label is available.
                    kp_list.append([bx + bw / 2, by + bh, 1.0])
                    if k == 0:
                        n_unmatched += 1

            instances.append({"box": [bx, by, bw, bh], "kp": kp_list})

        records.append({"filepath": filepath, "instances": instances})

    logging.info(
        f"prefetch_fo_split: {len(records)} samples | "
        f"matched {n_matched} keypoints | "
        f"{n_unmatched} defaulted to bbox-bottom-centre (vis=1)"
    )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — dataset (runs in every DDP worker, no FiftyOne access)
# ─────────────────────────────────────────────────────────────────────────────

class FiftyOneKeypointDataset(Dataset):
    """
    Picklable PyTorch Dataset built from pre-fetched FiftyOne data.

    Accepts the list returned by `prefetch_fo_split`.  No FiftyOne /
    MongoDB connection is made after construction — safe for DDP workers.
    """

    def __init__(
        self,
        prefetched: List[Dict],
        num_keypoints:  int  = 1,
        keypoint_names: Optional[List[str]] = None,
        resolution:     int  = 560,
        augment:        bool = False,
    ):
        self._data          = prefetched          # list of dicts from prefetch_fo_split
        self.num_keypoints  = num_keypoints
        self.keypoint_names = keypoint_names or [f"kp_{i}" for i in range(num_keypoints)]
        self.resolution     = resolution
        self.augment        = augment

        # ImageNet normalisation (same as RF-DETR)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self._aug = _build_augmentation(resolution) if augment else None

        logging.info(
            f"FiftyOneKeypointDataset: {len(self._data)} samples | "
            f"K={num_keypoints} | augment={augment}"
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        record    = self._data[idx]
        filepath  = record["filepath"]
        instances = record["instances"]  # list of {"box":..., "kp":...}

        # ── Load image ────────────────────────────────────────────────
        img = Image.open(filepath).convert("RGB")
        img_np = np.array(img)   # H, W, 3

        raw_boxes     = [inst["box"] for inst in instances]
        raw_keypoints = [np.array(inst["kp"], dtype=np.float32)
                         for inst in instances]   # each [K, 3]

        # ── Augmentation ──────────────────────────────────────────────
        if self._aug is not None and raw_boxes:
            img_np, raw_boxes, raw_keypoints = _apply_augmentation(
                self._aug, img_np, raw_boxes, raw_keypoints
            )

        # ── Resize ────────────────────────────────────────────────────
        img_pil = Image.fromarray(img_np).resize(
            (self.resolution, self.resolution), Image.BILINEAR
        )
        img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std

        # ── Build target tensors ──────────────────────────────────────
        if raw_boxes:
            boxes_np = np.array(raw_boxes, dtype=np.float32)
            # FiftyOne [x,y,w,h] → model [cx,cy,w,h], both normalised
            boxes_np[:, 0] = boxes_np[:, 0] + boxes_np[:, 2] / 2   # cx
            boxes_np[:, 1] = boxes_np[:, 1] + boxes_np[:, 3] / 2   # cy
            boxes_np = np.clip(boxes_np, 0.0, 1.0)
            kp_np    = np.stack(raw_keypoints, axis=0)               # [N, K, 3]
            lbl_np   = np.zeros(len(raw_boxes), dtype=np.int64)
        else:
            boxes_np = np.array([[0.5, 0.5, 0.01, 0.01]], dtype=np.float32)
            kp_np    = np.zeros((1, self.num_keypoints, 3), dtype=np.float32)
            lbl_np   = np.zeros(1, dtype=np.int64)

        target = {
            "boxes":     torch.from_numpy(boxes_np),
            "labels":    torch.from_numpy(lbl_np),
            "keypoints": torch.from_numpy(kp_np),
            "image_id":  torch.tensor([idx]),
        }
        return img_tensor, target


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_augmentation(resolution: int):
    try:
        import albumentations as A
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3,
                              saturation=0.2, hue=0.05, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.RandomScale(scale_limit=0.2, p=0.3),
                A.PadIfNeeded(min_height=resolution, min_width=resolution,
                               border_mode=0, value=0, p=1.0),
                A.RandomCrop(height=resolution, width=resolution, p=1.0),
            ],
            bbox_params=A.BboxParams(
                format="albumentations",
                label_fields=["bbox_labels"],
                min_visibility=0.2,
            ),
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,
            ),
        )
    except ImportError:
        logging.warning("albumentations not installed — augmentation disabled")
        return None


def _apply_augmentation(aug, img_np, raw_boxes, raw_keypoints):
    """Apply Albumentations; returns (img_np, raw_boxes, raw_keypoints)."""
    H, W = img_np.shape[:2]

    alb_boxes = []
    for bx, by, bw, bh in raw_boxes:
        alb_boxes.append([
            float(np.clip(bx, 0, 1)),
            float(np.clip(by, 0, 1)),
            float(np.clip(bx + bw, 0, 1)),
            float(np.clip(by + bh, 0, 1)),
        ])

    kp_flat = []
    kp_meta = []
    for i, kp_arr in enumerate(raw_keypoints):
        for k in range(kp_arr.shape[0]):
            kp_flat.append((float(kp_arr[k, 0] * W), float(kp_arr[k, 1] * H)))
            kp_meta.append((i, k))

    try:
        result = aug(
            image=img_np,
            bboxes=alb_boxes,
            bbox_labels=list(range(len(alb_boxes))),
            keypoints=kp_flat,
        )
    except Exception as e:
        logging.debug(f"Augmentation failed ({e}); skipping")
        return img_np, raw_boxes, raw_keypoints

    out_img = result["image"]
    out_H, out_W = out_img.shape[:2]
    # Albumentations may return bbox_labels as floats even when ints were passed in.
    # Cast to int so they can be used as list indices.
    surviving_indices = [int(x) for x in result["bbox_labels"]]
    out_alb_boxes     = result["bboxes"]
    out_kps           = result["keypoints"]

    new_kp_dict: Dict[int, np.ndarray] = {}
    for flat_idx, (inst_idx, k_idx) in enumerate(kp_meta):
        if inst_idx not in new_kp_dict:
            new_kp_dict[inst_idx] = raw_keypoints[inst_idx].copy()
        if flat_idx < len(out_kps):
            ox, oy = out_kps[flat_idx][:2]
            new_kp_dict[inst_idx][k_idx, 0] = float(ox) / out_W
            new_kp_dict[inst_idx][k_idx, 1] = float(oy) / out_H

    out_boxes, out_keypoints = [], []
    for alb_box, orig_idx in zip(out_alb_boxes, surviving_indices):
        x_min, y_min, x_max, y_max = alb_box
        bw = max(x_max - x_min, 0.001)
        bh = max(y_max - y_min, 0.001)
        out_boxes.append([x_min, y_min, bw, bh])
        out_keypoints.append(new_kp_dict.get(orig_idx, raw_keypoints[orig_idx]))

    if not out_boxes:
        return out_img, raw_boxes, raw_keypoints

    return out_img, out_boxes, out_keypoints


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def fo_kp_collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)

