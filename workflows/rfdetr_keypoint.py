"""
RF-DETR with dual heads: bounding-box detection + keypoint estimation.

Architecture
------------
  Backbone  : RF-DETR (DINOv2 windowed ViT) – shared, pre-trained
  Transformer: RF-DETR decoder – shared, pre-trained
  Head 1    : bounding-box head (class_embed + bbox_embed) – from RF-DETR
  Head 2    : keypoint head (keypoint_embed MLP) – newly added

The model inherits LWDETR so every pre-trained weight loads cleanly;
only keypoint_embed is randomly initialised.

Training targets (per image, dict):
  boxes      : [N, 4]  normalised cx-cy-w-h
  labels     : [N]     int class ids
  keypoints  : [N, K, 3]  (x_norm, y_norm, visibility)
                           visibility: 0=unlabelled, 1=occluded, 2=visible

Output dict (forward pass):
  pred_logits   : [B, Q, num_classes]
  pred_boxes    : [B, Q, 4]   cx-cy-w-h, normalised
  pred_keypoints: [B, Q, K, 3]  (x, y) in [0,1], visibility logit
  aux_outputs   : list of dicts (same keys, one per decoder layer)
"""

import copy
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rfdetr.models.lwdetr import LWDETR, MLP
from rfdetr.util.misc import nested_tensor_from_tensor_list, NestedTensor
from rfdetr.models.matcher import build_matcher
from rfdetr.util import box_ops

from config.config import GLOBAL_SEED, HF_DO_UPLOAD, HF_ROOT, WANDB_ACTIVE


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Core model
# ─────────────────────────────────────────────────────────────────────────────

class LWDETRWithKeypointHead(LWDETR):
    """
    Extends LWDETR with a second MLP head for keypoint prediction.

    Each query predicts:
      • class logits   via class_embed  (inherited)
      • bbox           via bbox_embed   (inherited)
      • K keypoints    via keypoint_embed  (new)

    keypoint_embed output shape: [B, Q, K * 3]
      → reshaped to [B, Q, K, 3] where dim-2 = (x, y, visibility)
      x, y are passed through sigmoid → [0, 1] normalised coordinates.
      visibility is a raw logit (apply sigmoid for probability).
    """

    def __init__(
        self,
        backbone,
        transformer,
        segmentation_head,
        num_classes: int,
        num_queries: int,
        num_keypoints: int,
        keypoint_names: Optional[List[str]] = None,
        aux_loss: bool = False,
        group_detr: int = 1,
        two_stage: bool = False,
        lite_refpoint_refine: bool = False,
        bbox_reparam: bool = False,
    ):
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=segmentation_head,
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=aux_loss,
            group_detr=group_detr,
            two_stage=two_stage,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
        )
        self.num_keypoints = num_keypoints
        self.keypoint_names = keypoint_names or [f"kp_{i}" for i in range(num_keypoints)]

        hidden_dim = transformer.d_model
        # Keypoint MLP: hidden_dim → hidden_dim → num_keypoints * 3
        self.keypoint_embed = MLP(hidden_dim, hidden_dim, num_keypoints * 3, num_layers=3)
        nn.init.constant_(self.keypoint_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.keypoint_embed.layers[-1].bias.data, 0)

    # ------------------------------------------------------------------
    def forward(self, samples: NestedTensor, targets=None):
        """Returns dict with pred_logits, pred_boxes, pred_keypoints (+aux)."""
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, poss = self.backbone(samples)

        srcs, masks = [], []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
            query_feat_weight = self.query_feat.weight[: self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight
        )

        out = {}

        if hs is not None:
            # ── Bbox head ──────────────────────────────────────────────
            if self.bbox_reparam:
                delta = self.bbox_embed(hs)
                cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.cat([cxcy, wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            # ── Keypoint head ──────────────────────────────────────────
            # hs: [num_layers, B, Q, hidden_dim]
            kp_raw = self.keypoint_embed(hs)  # [num_layers, B, Q, K*3]
            num_layers, B, Q, _ = kp_raw.shape
            kp = kp_raw.view(num_layers, B, Q, self.num_keypoints, 3)
            # normalise x, y to [0,1]; keep visibility as logit
            kp = torch.cat([kp[..., :2].sigmoid(), kp[..., 2:3]], dim=-1)

            out = {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
                "pred_keypoints": kp[-1],  # [B, Q, K, 3]
            }

            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss_kp(
                    outputs_class, outputs_coord, kp
                )

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = torch.cat(
                [self.transformer.enc_out_class_embed[g](hs_enc_list[g]) for g in range(group_detr)],
                dim=1,
            )
            if hs is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}

        return out

    @torch.jit.unused
    def _set_aux_loss_kp(self, outputs_class, outputs_coord, outputs_kp):
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_keypoints": c}
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_kp[:-1])
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Keypoint loss
# ─────────────────────────────────────────────────────────────────────────────

class KeypointCriterion(nn.Module):
    """
    Computes keypoint loss on Hungarian-matched prediction/GT pairs.

    Loss components:
      loss_kp_xy  : L1 on (x, y) for labelled keypoints (visibility > 0)
      loss_kp_vis : BCE on visibility flag (binary: labelled vs unlabelled)
    """

    def __init__(
        self,
        matcher,
        num_keypoints: int,
        kp_xy_coef: float = 5.0,
        kp_vis_coef: float = 1.0,
        group_detr: int = 1,
    ):
        super().__init__()
        self.matcher = matcher
        self.num_keypoints = num_keypoints
        self.kp_xy_coef = kp_xy_coef
        self.kp_vis_coef = kp_vis_coef
        self.group_detr = group_detr

    # ------------------------------------------------------------------
    def _compute_kp_loss(self, outputs, targets, indices, num_boxes):
        """Core loss computation for one set of outputs."""
        if "pred_keypoints" not in outputs:
            return {}

        pred_kp = outputs["pred_keypoints"]  # [B, Q, K, 3]
        if pred_kp is None:
            return {}

        src_idx = self._get_src_permutation_idx(indices)

        # Gather predicted keypoints for matched queries
        src_kp = pred_kp[src_idx]  # [N_matched, K, 3]

        # Gather GT keypoints for matched GT objects
        tgt_kp = torch.cat(
            [t["keypoints"][j] for t, (_, j) in zip(targets, indices)], dim=0
        )  # [N_matched, K, 3] – (x_norm, y_norm, visibility)

        if src_kp.shape[0] == 0:
            device = pred_kp.device
            return {
                "loss_kp_xy": torch.tensor(0.0, device=device, requires_grad=True),
                "loss_kp_vis": torch.tensor(0.0, device=device, requires_grad=True),
            }

        # Visibility mask: labelled keypoints have visibility > 0
        vis_mask = (tgt_kp[..., 2] > 0).float()  # [N_matched, K]

        # ── xy loss (L1, only on labelled keypoints) ─────────────────
        pred_xy = src_kp[..., :2]  # [N_matched, K, 2]
        tgt_xy = tgt_kp[..., :2]   # [N_matched, K, 2]

        xy_loss = F.l1_loss(pred_xy, tgt_xy, reduction="none")  # [N_matched, K, 2]
        xy_loss = (xy_loss.sum(-1) * vis_mask).sum()

        num_kp = vis_mask.sum().clamp(min=1)
        loss_kp_xy = xy_loss / num_kp

        # ── visibility loss (BCE, all keypoints) ─────────────────────
        pred_vis_logit = src_kp[..., 2]  # [N_matched, K]
        tgt_vis_binary = (tgt_kp[..., 2] > 0).float()

        loss_kp_vis = F.binary_cross_entropy_with_logits(
            pred_vis_logit, tgt_vis_binary, reduction="sum"
        ) / max(num_boxes, 1)

        return {"loss_kp_xy": loss_kp_xy, "loss_kp_vis": loss_kp_vis}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    # ------------------------------------------------------------------
    def forward(self, outputs, targets):
        """
        Compute keypoint losses for main outputs and auxiliary outputs.

        Returns a flat dict of losses, e.g.:
          loss_kp_xy, loss_kp_vis,
          loss_kp_xy_0, loss_kp_vis_0,  ...  (aux layers)
        """
        # Count total labelled keypoints across the batch
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(num_boxes, 1)

        # Main decoder layer
        indices = self.matcher(outputs, targets)
        losses = self._compute_kp_loss(outputs, targets, indices, num_boxes)

        # Auxiliary decoder layers
        if "aux_outputs" in outputs:
            for i, aux_out in enumerate(outputs["aux_outputs"]):
                aux_indices = self.matcher(aux_out, targets)
                aux_losses = self._compute_kp_loss(aux_out, targets, aux_indices, num_boxes)
                losses.update({f"{k}_{i}": v for k, v in aux_losses.items()})

        return losses


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Validation metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou_xyxy(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of boxes in x1y1x2y2 format. Returns [N, M] matrix."""
    x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def _compute_ap(ap_entries: list, n_gt: int) -> float:
    """
    Compute Average Precision from a list of (score, is_tp) pairs and total GT count.
    Uses the 101-point COCO interpolation.
    """
    if n_gt == 0 or not ap_entries:
        return 0.0
    ap_entries.sort(key=lambda x: -x[0])
    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    for _, is_tp in ap_entries:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)

    # 101-point interpolation (COCO style)
    ap = 0.0
    for t in [r / 100 for r in range(101)]:
        p_at_t = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p_at_t) if p_at_t else 0.0
    return ap / 101


def _val_metrics_batch(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_kp: torch.Tensor,
    targets: list,
    # Lowered from 0.5 → 0.3: early in training the model's max confidence is
    # often 0.3–0.4, so conf_thresh=0.5 produced TPs=0 and OKS=N/A for many
    # epochs even though the model was actively learning.  0.3 lets OKS/PCK
    # be computed while the model is still warming up.
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    kp_sigma: float = 0.089,
    pck_thresh: float = 0.2,
    scale_min: float = 0.05,
) -> dict:
    """
    Compute all validation metrics for one batch.

    pred_logits : [B, Q, C]
    pred_boxes  : [B, Q, 4]   cx,cy,w,h normalised
    pred_kp     : [B, Q, K, 3]  x,y in [0,1], vis logit
    targets     : list of dicts with 'boxes' [N,4], 'keypoints' [N,K,3]

    Returns a dict of accumulators:
      Bbox  : tp, fp, fn, n_gt, ap_entries [(score, is_tp)]
      OKS   : oks_sum, n_oks, oks_per_kp [K], n_oks_per_kp [K]
      PCK   : pck_correct, pck_total, pck_per_kp [K], pck_total_per_kp [K]
    """
    K = pred_kp.shape[2]
    acc = dict(
        tp=0, fp=0, fn=0, n_gt=0, ap_entries=[],
        oks_sum=0.0, n_oks=0,
        oks_per_kp=[0.0] * K, n_oks_per_kp=[0] * K,
        pck_correct=0, pck_total=0,
        pck_per_kp=[0] * K, pck_total_per_kp=[0] * K,
    )

    scores = pred_logits.sigmoid().max(dim=-1).values  # [B, Q]

    for b in range(pred_logits.shape[0]):
        sc = scores[b]                      # [Q]
        pb = pred_boxes[b]                  # [Q, 4]
        pk = pred_kp[b]                     # [Q, K, 3]
        gt_boxes = targets[b]["boxes"]      # [N, 4] cx,cy,w,h
        gt_kp    = targets[b]["keypoints"]  # [N, K, 3]

        N_gt = gt_boxes.shape[0]
        acc["n_gt"] += N_gt

        keep = sc > conf_thresh
        pb_f = pb[keep]
        pk_f = pk[keep]
        sc_f = sc[keep]
        N_pred = pb_f.shape[0]

        if N_gt == 0 and N_pred == 0:
            continue
        if N_gt == 0:
            acc["fp"] += N_pred
            acc["ap_entries"].extend((s.item(), False) for s in sc_f)
            continue
        if N_pred == 0:
            acc["fn"] += N_gt
            continue

        iou_mat = _box_iou_xyxy(
            _cxcywh_to_xyxy(pb_f),
            _cxcywh_to_xyxy(gt_boxes),
        )  # [N_pred, N_gt]

        matched_gt   = set()
        matched_pred = set()

        for pi in sc_f.argsort(descending=True).tolist():
            best_iou = iou_thresh - 1e-9
            best_gi  = -1
            for gi in range(N_gt):
                if gi in matched_gt:
                    continue
                if iou_mat[pi, gi].item() > best_iou:
                    best_iou = iou_mat[pi, gi].item()
                    best_gi  = gi

            is_tp = best_gi >= 0
            acc["ap_entries"].append((sc_f[pi].item(), is_tp))

            if is_tp:
                matched_gt.add(best_gi)
                matched_pred.add(pi)

                pred_xy = pk_f[pi, :, :2]          # [K, 2]
                tgt_xy  = gt_kp[best_gi, :, :2]    # [K, 2]
                tgt_vis = gt_kp[best_gi, :, 2]     # [K]
                vis_mask = (tgt_vis > 0)

                if vis_mask.sum() > 0:
                    g = gt_boxes[best_gi]
                    # Clamp scale to scale_min so tiny far-away pedestrian boxes
                    # don't collapse the OKS denominator to near-zero.
                    # scale_min ≈ 0.05 means we never penalise predictions more
                    # harshly than for a box that is 5 % of the image side.
                    scale = (g[2] * g[3]).clamp(min=scale_min ** 2).sqrt()

                    d2 = ((pred_xy - tgt_xy) ** 2).sum(dim=-1)  # [K]

                    # ── OKS ───────────────────────────────────────────
                    oks_per_kp = torch.exp(-d2 / (2 * scale ** 2 * kp_sigma ** 2))
                    vis_f = vis_mask.float()
                    oks_val = (oks_per_kp * vis_f).sum() / vis_f.sum()
                    acc["oks_sum"] += oks_val.item()
                    acc["n_oks"]   += 1
                    for k in range(K):
                        if vis_mask[k]:
                            acc["oks_per_kp"][k]   += oks_per_kp[k].item()
                            acc["n_oks_per_kp"][k] += 1

                    # ── PCK@{pck_thresh} ──────────────────────────────
                    dist = d2.sqrt()                      # [K] normalised distance
                    threshold = pck_thresh * scale        # bbox-relative threshold
                    for k in range(K):
                        if vis_mask[k]:
                            acc["pck_total"]          += 1
                            acc["pck_total_per_kp"][k] += 1
                            if dist[k].item() < threshold.item():
                                acc["pck_correct"]          += 1
                                acc["pck_per_kp"][k]         += 1

        acc["tp"] += len(matched_pred)
        acc["fp"] += N_pred - len(matched_pred)
        acc["fn"] += N_gt  - len(matched_gt)

    return acc


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CocoKeypointDataset(Dataset):
    """
    Loads a COCO-format dataset that includes keypoint annotations.

    Expected annotation JSON structure:
      {
        "images": [...],
        "annotations": [
          {
            "id": ..., "image_id": ..., "category_id": ...,
            "bbox": [x, y, w, h],           ← absolute pixels
            "keypoints": [x1,y1,v1,...],    ← absolute pixels + visibility
            "num_keypoints": N,
            ...
          },
          ...
        ],
        "categories": [
          {"id": ..., "name": ..., "keypoints": [...], "skeleton": [...]}
        ]
      }
    """

    def __init__(
        self,
        image_dir: str,
        annotation_path: str,
        num_keypoints: int,
        resolution: int = 560,
        augment: bool = False,
    ):
        self.image_dir = image_dir
        self.resolution = resolution
        self.augment = augment
        self.num_keypoints = num_keypoints

        with open(annotation_path, "r") as f:
            data = json.load(f)

        # Build image-id → file map
        self.images = {img["id"]: img for img in data["images"]}

        # Build category-id → 0-indexed class id
        cats = sorted(data["categories"], key=lambda c: c["id"])
        self.cat_id_to_class = {c["id"]: i for i, c in enumerate(cats)}
        self.class_names = [c["name"] for c in cats]

        # Group annotations by image
        from collections import defaultdict
        ann_by_image = defaultdict(list)
        for ann in data["annotations"]:
            ann_by_image[ann["image_id"]].append(ann)

        self.samples = []
        for img_id, img_info in self.images.items():
            anns = ann_by_image.get(img_id, [])
            if not anns:
                continue
            self.samples.append((img_id, img_info, anns))

        # ImageNet normalisation (same as RF-DETR)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        logging.info(
            f"CocoKeypointDataset: {len(self.samples)} images, "
            f"{len(self.class_names)} classes, {num_keypoints} keypoints"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, img_info, anns = self.samples[idx]

        # Load image
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize to model resolution
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        sx = self.resolution / orig_w
        sy = self.resolution / orig_h

        # To tensor + normalise
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std

        # Build targets
        boxes, labels, keypoints = [], [], []
        for ann in anns:
            # Validate category before appending anything
            cat_id = ann["category_id"]
            if cat_id not in self.cat_id_to_class:
                logging.warning(f"Unknown category_id {cat_id} in annotation, skipping")
                continue
            cls_idx = self.cat_id_to_class[cat_id]
            if cls_idx >= len(self.class_names):
                logging.error(f"class idx {cls_idx} >= num_classes {len(self.class_names)}, cat_id={cat_id}")
                continue

            x, y, w, h = ann["bbox"]
            # Convert to normalised cx-cy-w-h
            cx = (x + w / 2) * sx / self.resolution
            cy = (y + h / 2) * sy / self.resolution
            nw = w * sx / self.resolution
            nh = h * sy / self.resolution
            boxes.append([cx, cy, nw, nh])
            labels.append(cls_idx)

            # Keypoints: [K*3] → [K, 3], normalise xy
            kp_flat = ann.get("keypoints", [0] * (self.num_keypoints * 3))
            # Pad / truncate to expected length
            if len(kp_flat) < self.num_keypoints * 3:
                kp_flat = kp_flat + [0] * (self.num_keypoints * 3 - len(kp_flat))
            kp_flat = kp_flat[: self.num_keypoints * 3]
            kp = np.array(kp_flat, dtype=np.float32).reshape(self.num_keypoints, 3)
            kp[:, 0] = kp[:, 0] * sx / self.resolution
            kp[:, 1] = kp[:, 1] * sy / self.resolution
            keypoints.append(kp)

        if not boxes:
            boxes = [[0.5, 0.5, 0.1, 0.1]]
            labels = [0]
            keypoints = [np.zeros((self.num_keypoints, 3), dtype=np.float32)]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "keypoints": torch.tensor(np.array(keypoints), dtype=torch.float32),
            "image_id": torch.tensor([img_id]),
        }

        return img_tensor, target


def kp_collate_fn(batch):
    """Collate function for CocoKeypointDataset."""
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Model builder (loads RF-DETR pretrain + adds keypoint head)
# ─────────────────────────────────────────────────────────────────────────────

def build_rfdetr_keypoint_model(
    pretrain_weights: Optional[str],
    num_classes: int,
    num_keypoints: int,
    keypoint_names: Optional[List[str]] = None,
    rfdetr_config_name: str = "rfdetr_base",
    device: str = "cuda",
) -> LWDETRWithKeypointHead:
    """
    Builds LWDETRWithKeypointHead, optionally loading RF-DETR pretrained weights.

    The backbone + transformer + bbox/class heads are loaded from pretrain_weights.
    keypoint_embed is always randomly initialised.
    """
    from rfdetr.main import Model, download_pretrain_weights
    from rfdetr.config import (
        RFDETRBaseConfig, RFDETRLargeConfig, RFDETRSmallConfig,
        RFDETRMediumConfig, RFDETRNanoConfig,
    )
    from rfdetr.platform.models import RFDETRXLargeConfig, RFDETR2XLargeConfig
    from rfdetr.models import build_model as rfdetr_build_model
    import argparse

    # Map config name → config class
    CONFIG_MAP = {
        "rfdetr_nano":    RFDETRNanoConfig,
        "rfdetr_small":   RFDETRSmallConfig,
        "rfdetr_medium":  RFDETRMediumConfig,
        "rfdetr_base":    RFDETRBaseConfig,
        "rfdetr_large":   RFDETRLargeConfig,
        "rfdetr_xlarge":  RFDETRXLargeConfig,
        "rfdetr_2xlarge": RFDETR2XLargeConfig,
    }
    config_cls = CONFIG_MAP.get(rfdetr_config_name.lower(), RFDETRBaseConfig)
    # Pydantic config only accepts 'cpu'/'cuda'/'mps' — strip rank suffix for DDP
    config_device = "cuda" if str(device).startswith("cuda") else str(device)
    model_cfg = config_cls(num_classes=num_classes, device=config_device)

    # Download pretrain weights if needed
    if pretrain_weights:
        download_pretrain_weights(pretrain_weights)

    # Build underlying args namespace (mirrors main.py populate_args)
    from rfdetr.main import populate_args
    kwargs = model_cfg.model_dump()
    kwargs["pretrain_weights"] = pretrain_weights
    args = populate_args(**kwargs)

    # Build backbone + transformer (same as rf-detr)
    from rfdetr.models.backbone import build_backbone
    from rfdetr.models.lwdetr import build_model as _build_lwdetr
    base_lwdetr = _build_lwdetr(args)

    # Now create the keypoint model, copying weights
    kp_model = LWDETRWithKeypointHead(
        backbone=base_lwdetr.backbone,
        transformer=base_lwdetr.transformer,
        segmentation_head=None,
        num_classes=num_classes + 1,   # +1 for no-object (RF-DETR convention)
        num_queries=args.num_queries,
        num_keypoints=num_keypoints,
        keypoint_names=keypoint_names,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
    )

    # Copy class_embed and bbox_embed weights from base model
    kp_model.class_embed.load_state_dict(base_lwdetr.class_embed.state_dict())
    kp_model.bbox_embed.load_state_dict(base_lwdetr.bbox_embed.state_dict())
    kp_model.refpoint_embed.load_state_dict(base_lwdetr.refpoint_embed.state_dict())
    kp_model.query_feat.load_state_dict(base_lwdetr.query_feat.state_dict())
    if base_lwdetr.two_stage:
        kp_model.transformer.enc_out_class_embed = base_lwdetr.transformer.enc_out_class_embed
        kp_model.transformer.enc_out_bbox_embed = base_lwdetr.transformer.enc_out_bbox_embed

    # Load full checkpoint if pretrain_weights is a .pt / .pth file path
    if pretrain_weights and os.path.isfile(pretrain_weights):
        logging.info(f"Loading RF-DETR checkpoint: {pretrain_weights}")
        ckpt = torch.load(pretrain_weights, map_location="cpu")
        state = ckpt.get("model", ckpt)

        # Filter to only keys that exist in the model AND have matching shapes.
        # Class-head weights (class_embed, enc_out_class_embed) will differ in
        # dim-0 when the checkpoint was trained on a different number of classes,
        # so we skip those and let them stay randomly initialised.
        model_state = kp_model.state_dict()
        compatible, skipped = {}, []
        for k, v in state.items():
            if k not in model_state:
                continue                          # key not in new model – skip
            if v.shape != model_state[k].shape:
                skipped.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_state[k].shape)}")
                continue
            compatible[k] = v

        if skipped:
            logging.info(
                f"Skipped {len(skipped)} shape-mismatched keys (class heads will be "
                f"randomly initialised for your dataset):\n  " + "\n  ".join(skipped)
            )

        missing, unexpected = kp_model.load_state_dict(compatible, strict=False)
        logging.info(
            f"Loaded checkpoint – transferred: {len(compatible)}, "
            f"missing: {len(missing)}, skipped (shape mismatch): {len(skipped)}"
        )
        non_kp_missing = [k for k in missing if "keypoint_embed" not in k and k not in {s.split(":")[0] for s in skipped}]
        if non_kp_missing:
            logging.warning(f"Unexpected missing keys: {non_kp_missing[:10]}")

    kp_model = kp_model.to(device)
    return kp_model


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DDP training worker (module-level so mp.spawn can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _ddp_train_worker(
    rank: int,
    world_size: int,
    dataset_dir: str,
    output_dir: str,
    run_config: dict,
    shared_config: dict,
    num_keypoints: int,
    keypoint_names: list,
    fo_prefetched: Optional[dict] = None,
):
    """Training loop for one GPU rank. Called by mp.spawn or directly."""
    import torch.optim as optim

    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size,
            timeout=__import__("datetime").timedelta(hours=4),
        )
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
        is_main = (rank == 0)
        logging.info(f"DDP rank {rank}/{world_size} on {device}")
    elif torch.cuda.is_available():
        device = "cuda"
        is_main = True
        logging.info("Single-GPU training on cuda")
    else:
        device = "cpu"
        is_main = True
        logging.warning("No GPU – training on CPU")

    model_name = run_config.get("config", "rfdetr_base").lower()
    os.makedirs(output_dir, exist_ok=True)

    # Derive num_classes — from COCO JSON (COCO-export mode) or run_config (FO-native mode)
    if fo_prefetched is not None:
        num_classes = run_config.get("num_classes", 1)
        class_names = run_config.get("class_names", [run_config.get("target_label", "pedestrian")])
        if is_main:
            logging.info(f"FO-native mode | classes: {class_names}")
    else:
        train_ann_path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
        try:
            with open(train_ann_path) as f:
                _coco = json.load(f)
            class_names = [c["name"] for c in sorted(_coco["categories"], key=lambda c: c["id"])]
            num_classes = len(class_names)
        except Exception:
            num_classes = run_config.get("num_classes", 1)
            class_names = None
        if is_main:
            logging.info(f"Classes (from COCO JSON): {num_classes}")

    # Pretrain weights
    pretrain_weights = run_config.get("pretrain_weights", None)
    if pretrain_weights is None:
        DEFAULT_WEIGHTS = {
            "rfdetr_nano":    "rf-detr-nano.pth",
            "rfdetr_small":   "rf-detr-small.pth",
            "rfdetr_medium":  "rf-detr-medium.pth",
            "rfdetr_base":    "rf-detr-base.pth",
            "rfdetr_large":   "rf-detr-large.pth",
            "rfdetr_xlarge":  "rf-detr-xlarge.pth",
            "rfdetr_2xlarge": "rf-detr-xxlarge.pth",
        }
        pretrain_weights = DEFAULT_WEIGHTS.get(model_name, "rf-detr-base.pth")

    if is_main:
        logging.info(f"Building {model_name} with {num_keypoints} keypoints …")
    model = build_rfdetr_keypoint_model(
        pretrain_weights=pretrain_weights,
        num_classes=num_classes,
        num_keypoints=num_keypoints,
        keypoint_names=keypoint_names,
        rfdetr_config_name=model_name,
        device=device,
    )

    # ── DataLoaders ───────────────────────────────────────────────────
    resolution = run_config.get("resolution", 560)
    batch_size = run_config.get("batch_size", 8)

    def _make_loader(split, shuffle):
        # ── FiftyOne-native mode (pre-fetched, no COCO export needed) ─
        if fo_prefetched is not None:
            from workflows.keypoint.fo_keypoint_dataset import (
                FiftyOneKeypointDataset, fo_kp_collate_fn,
            )
            split_key = "valid" if split == "valid" else split   # normalise key
            data = fo_prefetched.get(split_key)
            if not data:
                return None
            ds = FiftyOneKeypointDataset(
                prefetched=data,
                num_keypoints=num_keypoints,
                keypoint_names=keypoint_names,
                resolution=resolution,
                augment=shuffle,
            )
            collate = fo_kp_collate_fn
        else:
            # ── COCO-export mode (original path) ──────────────────────
            split_dir = os.path.join(dataset_dir, split)
            ann_path = os.path.join(split_dir, "_annotations.coco.json")
            if not os.path.exists(ann_path):
                return None
            ds = CocoKeypointDataset(
                image_dir=split_dir,
                annotation_path=ann_path,
                num_keypoints=num_keypoints,
                resolution=resolution,
                augment=shuffle,
            )
            collate = kp_collate_fn

        if is_distributed and shuffle:
            sampler = DistributedSampler(ds, rank=rank, num_replicas=world_size, shuffle=True)
            return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                              num_workers=2, collate_fn=collate, pin_memory=True)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=2, collate_fn=collate,
                          pin_memory=torch.cuda.is_available())

    train_loader = _make_loader("train", shuffle=True)
    val_loader   = _make_loader("valid", shuffle=False)
    if train_loader is None:
        raise FileNotFoundError("No training data found.")

    # ── Matcher + criteria ────────────────────────────────────────────
    from rfdetr.main import populate_args
    from rfdetr.config import (RFDETRBaseConfig, RFDETRNanoConfig,
                                RFDETRSmallConfig, RFDETRMediumConfig, RFDETRLargeConfig)
    from rfdetr.platform.models import RFDETRXLargeConfig, RFDETR2XLargeConfig
    from rfdetr.models import build_criterion_and_postprocessors

    CONFIG_MAP = {
        "rfdetr_nano":    RFDETRNanoConfig,
        "rfdetr_small":   RFDETRSmallConfig,
        "rfdetr_medium":  RFDETRMediumConfig,
        "rfdetr_base":    RFDETRBaseConfig,
        "rfdetr_large":   RFDETRLargeConfig,
        "rfdetr_xlarge":  RFDETRXLargeConfig,
        "rfdetr_2xlarge": RFDETR2XLargeConfig,
    }
    cfg_cls = CONFIG_MAP.get(model_name, RFDETRBaseConfig)
    config_device = "cuda" if str(device).startswith("cuda") else str(device)
    model_cfg = cfg_cls(num_classes=num_classes, device=config_device)
    rfdetr_args = populate_args(**model_cfg.model_dump())

    criterion, _ = build_criterion_and_postprocessors(rfdetr_args)
    criterion = criterion.to(device)

    kp_criterion = KeypointCriterion(
        matcher=build_matcher(rfdetr_args),
        num_keypoints=num_keypoints,
        kp_xy_coef=run_config.get("kp_xy_coef", 5.0),
        kp_vis_coef=run_config.get("kp_vis_coef", 1.0),
        group_detr=rfdetr_args.group_detr,
    )

    # ── Optimizer ────────────────────────────────────────────────────
    epochs   = shared_config.get("epochs", 50)
    base_lr  = shared_config.get("learning_rate", 1e-4)
    lr_enc   = run_config.get("lr_encoder") or (base_lr * 0.1)
    wd       = shared_config.get("weight_decay", 1e-4)

    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model

    backbone_params = list(raw_model.backbone.parameters())
    head_params = (
        list(raw_model.transformer.parameters())
        + list(raw_model.class_embed.parameters())
        + list(raw_model.bbox_embed.parameters())
        + list(raw_model.keypoint_embed.parameters())
        + list(raw_model.refpoint_embed.parameters())
        + list(raw_model.query_feat.parameters())
    )
    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": lr_enc},
         {"params": head_params,     "lr": base_lr}],
        weight_decay=wd,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    freeze_epochs = run_config.get("freeze_backbone_epochs", 5)
    best_val_loss = float("inf")
    patience  = shared_config.get("early_stop_patience", 10)
    no_improve = 0

    if is_main:
        logging.info("=" * 70)
        logging.info("RF-DETR KEYPOINT TRAINING")
        logging.info(f"  Model     : {model_name}")
        logging.info(f"  Epochs    : {epochs}")
        logging.info(f"  Batch size: {batch_size}  ×  {world_size} GPU(s)")
        logging.info(f"  Keypoints : {num_keypoints} {keypoint_names}")
        logging.info("=" * 70)

    for epoch in range(epochs):
        requires_grad = epoch >= freeze_epochs
        for p in raw_model.backbone.parameters():
            p.requires_grad_(requires_grad)
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}",
                                  disable=not is_main):
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            nested  = nested_tensor_from_tensor_list(imgs)
            outputs = model(nested, targets)

            det_loss_dict = criterion(outputs, targets)
            det_loss = sum(det_loss_dict[k] * criterion.weight_dict[k]
                           for k in det_loss_dict if k in criterion.weight_dict)
            kp_loss_dict = kp_criterion(outputs, targets)
            kp_loss = (kp_criterion.kp_xy_coef * kp_loss_dict.get("loss_kp_xy", 0.0)
                       + kp_criterion.kp_vis_coef * kp_loss_dict.get("loss_kp_vis", 0.0))

            total_loss = det_loss + kp_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            epoch_loss += total_loss.item()
            n_batches  += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation (ALL ranks) ────────────────────────────────────
        # All ranks process the full val set simultaneously so that
        # dist.all_reduce inside rfdetr's criterion stays in sync.
        avg_val_loss = avg_train_loss
        val_metrics: dict = {}
        if val_loader is not None:
            if is_main:
                logging.info(f"Epoch {epoch+1}: running validation on all ranks…")
            model.eval()
            val_loss, n_val = 0.0, 0
            K = num_keypoints
            agg = dict(
                tp=0, fp=0, fn=0, n_gt=0, ap_entries=[],
                oks_sum=0.0, n_oks=0,
                oks_per_kp=[0.0]*K, n_oks_per_kp=[0]*K,
                pck_correct=0, pck_total=0,
                pck_per_kp=[0]*K, pck_total_per_kp=[0]*K,
                # Diagnostics: track raw prediction counts to distinguish
                # "no predictions at all" from "predictions below conf_thresh"
                n_pred_total=0, max_pred_conf=0.0,
            )
            with torch.no_grad():
                for imgs, targets in tqdm(val_loader, desc=f"  Val {epoch+1}/{epochs}",
                                          disable=not is_main):
                    imgs    = [img.to(device) for img in imgs]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    nested  = nested_tensor_from_tensor_list(imgs)
                    outputs = raw_model(nested)
                    # Both criterion (rfdetr det loss) and kp_criterion are safe
                    # here because ALL ranks run this loop together, so any
                    # dist.all_reduce inside criterion completes without deadlock.
                    det_loss_dict = criterion(outputs, targets)
                    det_loss = sum(det_loss_dict[k] * criterion.weight_dict[k]
                                   for k in det_loss_dict if k in criterion.weight_dict)
                    kp_loss_dict = kp_criterion(outputs, targets)
                    kp_loss = (kp_criterion.kp_xy_coef * kp_loss_dict.get("loss_kp_xy", 0.0)
                               + kp_criterion.kp_vis_coef * kp_loss_dict.get("loss_kp_vis", 0.0))
                    batch_loss = det_loss + kp_loss
                    val_loss += batch_loss.item() if hasattr(batch_loss, "item") else float(batch_loss)
                    n_val += 1
                    if "pred_keypoints" in outputs and outputs["pred_keypoints"] is not None:
                        # Accumulate raw confidence stats for diagnostics (before
                        # conf_thresh filtering, so we can tell "no predictions" from
                        # "predictions present but all below 0.5 threshold").
                        scores = outputs["pred_logits"].detach().sigmoid().max(dim=-1).values
                        agg["n_pred_total"] += int(scores.numel())
                        batch_max = float(scores.max().item()) if scores.numel() > 0 else 0.0
                        if batch_max > agg["max_pred_conf"]:
                            agg["max_pred_conf"] = batch_max

                        batch_m = _val_metrics_batch(
                            outputs["pred_logits"].detach(),
                            outputs["pred_boxes"].detach(),
                            outputs["pred_keypoints"].detach(),
                            targets,
                        )
                        for key in ("tp","fp","fn","n_gt","oks_sum","n_oks","pck_correct","pck_total"):
                            agg[key] += batch_m[key]
                        agg["ap_entries"].extend(batch_m["ap_entries"])
                        for k in range(K):
                            agg["oks_per_kp"][k]       += batch_m["oks_per_kp"][k]
                            agg["n_oks_per_kp"][k]     += batch_m["n_oks_per_kp"][k]
                            agg["pck_per_kp"][k]       += batch_m["pck_per_kp"][k]
                            agg["pck_total_per_kp"][k] += batch_m["pck_total_per_kp"][k]

            avg_val_loss = val_loss / max(n_val, 1)
            # Only rank 0 computes final metrics (all ranks have identical values
            # since they all processed the full val set without a DistributedSampler)
            if is_main:
                tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
                prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
                rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
                f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
                mAP  = _compute_ap(agg["ap_entries"], agg["n_gt"])
                oks  = agg["oks_sum"]/agg["n_oks"] if agg["n_oks"]>0 else float("nan")
                pck  = agg["pck_correct"]/agg["pck_total"] if agg["pck_total"]>0 else float("nan")
                per_kp_oks = [agg["oks_per_kp"][k]/agg["n_oks_per_kp"][k]
                              if agg["n_oks_per_kp"][k]>0 else float("nan") for k in range(K)]
                per_kp_pck = [agg["pck_per_kp"][k]/agg["pck_total_per_kp"][k]
                              if agg["pck_total_per_kp"][k]>0 else float("nan") for k in range(K)]
                val_metrics = dict(prec=prec, rec=rec, f1=f1, mAP=mAP, oks=oks, pck=pck,
                                   per_kp_oks=per_kp_oks, per_kp_pck=per_kp_pck)

        # ── Logging & checkpoint (rank 0 only) ────────────────────────
        should_stop = 0
        if is_main:
            logging.info(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")
            if val_metrics:
                logging.info(f"  BBox  → Prec={val_metrics['prec']:.4f}  Rec={val_metrics['rec']:.4f}  "
                             f"F1={val_metrics['f1']:.4f}  mAP@0.5={val_metrics['mAP']:.4f}")
                def _fmt(v): return f"{v:.4f}" if v == v else "N/A (no annotated KP matched)"
                def _fmt3(v): return f"{v:.3f}" if v == v else "N/A"
                logging.info(f"  KP    → OKS={_fmt(val_metrics['oks'])}  "
                             f"PCK@0.2={_fmt(val_metrics['pck'])}")
                if val_metrics['oks'] != val_metrics['oks']:   # is nan
                    # Distinguish the two failure modes:
                    #   A) TPs=0 because model confidence is too low (below conf_thresh=0.5)
                    #   B) TPs exist but matched GT boxes all have vis=0 (keypoint data missing)
                    if agg["tp"] == 0:
                        if agg["max_pred_conf"] < 0.3:
                            diag = (
                                f"max_conf={agg['max_pred_conf']:.3f} < 0.3 thresh — "
                                "model producing very low-confidence predictions. "
                                "Verify pretrain_weights loaded (check log for 'Loaded checkpoint')."
                            )
                        else:
                            diag = (
                                f"max_conf={agg['max_pred_conf']:.3f} but no IoU>0.5 match — "
                                "predictions present but not overlapping GT boxes well enough."
                            )
                    else:
                        diag = "TPs exist but matched GT keypoints all have vis=0 — check 'pedestrian_points' field."
                    logging.warning(
                        f"  OKS=N/A: n_oks={agg['n_oks']}, TPs={agg['tp']}, "
                        f"n_gt={agg['n_gt']}, n_pred_total={agg['n_pred_total']}, "
                        f"max_pred_conf={agg['max_pred_conf']:.3f}. {diag}"
                    )
                oks_strs = "  ".join(f"{keypoint_names[k]}:{_fmt3(val_metrics['per_kp_oks'][k])}"
                                     for k in range(num_keypoints))
                pck_strs = "  ".join(f"{keypoint_names[k]}:{_fmt3(val_metrics['per_kp_pck'][k])}"
                                     for k in range(num_keypoints))
                logging.info(f"  OKS/kp  → {oks_strs}")
                logging.info(f"  PCK/kp  → {pck_strs}")

            ckpt = {
                "epoch": epoch, "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(), "val_loss": avg_val_loss,
                "num_keypoints": num_keypoints, "keypoint_names": keypoint_names,
                "num_classes": num_classes, "class_names": class_names,
                "model_name": model_name,
            }
            torch.save(ckpt, os.path.join(output_dir, "last.pt"))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                torch.save(ckpt, os.path.join(output_dir, "best.pt"))
                logging.info(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    should_stop = 1

        # Broadcast stop decision from rank 0 to all ranks.
        # This runs AFTER validation is fully complete on all ranks,
        # so there is no risk of deadlock with criterion's all_reduce.
        if is_distributed:
            stop_flag = torch.tensor([should_stop], device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break
        elif should_stop:
            break

    if is_distributed:
        dist.destroy_process_group()


# 7.  Main workflow class
# ─────────────────────────────────────────────────────────────────────────────

class RFDETRKeypointDetection:
    """
    Workflow for RF-DETR with bbox + keypoint heads.

    Mirrors the interface of CustomRFDETRObjectDetection and adds
    keypoint-specific data export, training, and inference.

    Configuration keys expected in run_config
    -----------------------------------------
    keypoint_field      : FiftyOne field name that holds fo.Keypoints  (default: "keypoints")
    keypoint_names      : list of keypoint names in order
    num_keypoints       : number of keypoints per instance
    batch_size          : default 8
    kp_xy_coef          : weight for keypoint xy loss  (default 5.0)
    kp_vis_coef         : weight for keypoint vis loss (default 1.0)
    freeze_backbone_epochs: epochs to keep backbone frozen (default 5)
    """

    def __init__(self, dataset, dataset_info, run_config):
        self.dataset = dataset
        self.dataset_name = dataset_info["name"]
        self.export_dir_root = run_config["export_dataset_root"]
        self.config_key = os.path.splitext(
            os.path.basename(run_config.get("config", "rfdetr_base"))
        )[0]
        self.hf_repo_name = f"{HF_ROOT}/{self.dataset_name}_{self.config_key}_kp"
        self.run_config = run_config

        self.keypoint_field = run_config.get("keypoint_field", "keypoints")
        self.keypoint_names = run_config.get("keypoint_names", [])
        self.num_keypoints = run_config.get("num_keypoints", len(self.keypoint_names))

        if self.num_keypoints == 0:
            raise ValueError(
                "run_config must specify 'num_keypoints' and/or 'keypoint_names'"
            )

    # ── Data conversion ───────────────────────────────────────────────

    def convert_data(self):
        """
        Export dataset to COCO keypoint format.

        Output structure:
          <export_dir_root>/<dataset_name>/rfdetr_kp/
              train/
                  _annotations.coco.json
                  *.jpg / *.png …
              valid/
              test/
        """
        if self.run_config.get("fo_native", False):
            logging.info("fo_native=True — skipping COCO export in convert_data().")
            return

        import fiftyone as fo  # lazy import — not needed by DDP workers
        export_dir = os.path.join(
            self.export_dir_root, self.dataset_name, "rfdetr_kp"
        )
        if os.path.exists(export_dir):
            # Only skip if the train split annotation file is actually present
            train_ann = os.path.join(export_dir, "train", "_annotations.coco.json")
            if os.path.exists(train_ann):
                # Check if keypoints are already injected; if not, re-inject
                # without re-exporting (saves time when export exists but
                # injection was skipped in a prior run).
                with open(train_ann) as _f:
                    _sample_data = json.load(_f)
                _has_kp = any(
                    a.get("num_keypoints", 0) > 0
                    for a in _sample_data.get("annotations", [])[:200]
                )
                if _has_kp:
                    logging.info(f"Export dir {export_dir} already complete, skipping.")
                    return
                logging.warning(
                    "Export exists but keypoints are missing — re-running injection "
                    "on existing annotation files (no re-export needed)."
                )
                for split_out in ("train", "valid", "test"):
                    ann_path = os.path.join(export_dir, split_out, "_annotations.coco.json")
                    if not os.path.exists(ann_path):
                        continue
                    # Pass an empty iterator as split_view — raw COCO file
                    # lookup doesn't need FiftyOne samples.
                    self._inject_keypoints(ann_path, [])
                    self._fix_annotation_indices(ann_path)
                logging.info("Keypoint injection complete.")
                return
            logging.warning(
                f"Export dir {export_dir} exists but is incomplete "
                f"(missing {train_ann}). Re-exporting."
            )

        os.makedirs(export_dir, exist_ok=True)

        # ── Build split views without modifying source dataset ────────
        # Priority: use existing tags; fall back to in-memory ID split so we
        # never write back to the source dataset (safe for zoo datasets).
        available_tags = self.dataset.distinct("tags")
        has_train = "train" in available_tags
        has_val   = "val"   in available_tags
        has_test  = "test"  in available_tags

        split_views = {}  # "train"/"val"/"test" → fo.DatasetView

        if has_train or has_val or has_test:
            # Dataset already has some split tags — use them, filling gaps in memory
            if has_train:
                split_views["train"] = self.dataset.match_tags("train")
            if has_val:
                split_views["val"] = self.dataset.match_tags("val")
            if has_test:
                split_views["test"] = self.dataset.match_tags("test")

            # Fill any missing splits by sub-sampling from an existing one
            if "train" in split_views and "val" not in split_views and "test" not in split_views:
                train_ids, val_ids = self._ids_split_fraction(
                    split_views["train"].values("id"), fraction=0.1
                )
                split_views["train"] = self.dataset.select(train_ids)
                split_views["val"]   = self.dataset.select(val_ids)
            elif "train" in split_views and "val" not in split_views:
                train_ids, val_ids = self._ids_split_fraction(
                    split_views["train"].values("id"), fraction=0.5
                )
                split_views["train"] = self.dataset.select(train_ids)
                split_views["val"]   = self.dataset.select(val_ids)
            elif "train" in split_views and "test" not in split_views:
                val_ids, test_ids = self._ids_split_fraction(
                    split_views["val"].values("id"), fraction=0.5
                )
                split_views["val"]  = self.dataset.select(val_ids)
                split_views["test"] = self.dataset.select(test_ids)
        else:
            # No tags at all (e.g. coco-2017-train zoo dataset) — split in memory
            logging.warning(
                "No split tags found. Splitting all samples: 90% train / 10% val."
            )
            all_ids = self.dataset.values("id")
            train_ids, val_ids = self._ids_split_fraction(all_ids, fraction=0.1)
            split_views["train"] = self.dataset.select(train_ids)
            split_views["val"]   = self.dataset.select(val_ids)
            logging.info(
                f"In-memory split: {len(train_ids)} train, {len(val_ids)} val"
            )

        # ── Detect detection field name ───────────────────────────────
        # COCODetectionDataset.export() needs the top-level fo.Detections field
        # name (e.g. "ground_truth" or "detections"), not the nested path.
        detection_field = self.run_config.get("detection_field", None)
        if detection_field is None:
            schema = self.dataset.get_field_schema()
            for candidate in ("ground_truth", "detections", "objects", "annotations"):
                if candidate in schema:
                    detection_field = candidate
                    break
            if detection_field is None:
                # Fall back to first Detections-type field found
                for fname, ftype in schema.items():
                    if "Detections" in type(ftype).__name__:
                        detection_field = fname
                        break
            if detection_field is None:
                raise ValueError(
                    f"Cannot find a Detections field in dataset schema: {list(schema.keys())}. "
                    "Set 'detection_field' in run_config explicitly."
                )
            logging.info(f"Using detection field: '{detection_field}'")

        # ── Export each split ─────────────────────────────────────────
        out_name = {"train": "train", "val": "valid", "test": "test"}
        for split_key, split_view in split_views.items():
            if len(split_view) == 0:
                continue

            split_dir = os.path.join(export_dir, out_name[split_key])
            os.makedirs(split_dir, exist_ok=True)

            ann_path = os.path.join(split_dir, "_annotations.coco.json")
            logging.info(f"Exporting {len(split_view)} samples → {out_name[split_key]}/")

            split_view.export(
                dataset_type=fo.types.COCODetectionDataset,
                data_path=split_dir,
                labels_path=ann_path,
                label_field=detection_field,
            )

            self._inject_keypoints(ann_path, split_view)
            self._fix_annotation_indices(ann_path)

        logging.info(f"Dataset exported to {export_dir}")

    def _build_coco_kp_lookup(self, coco_kp_file: str) -> dict:
        """
        Build a lookup: basename(file_name) → list of COCO annotations with keypoints.
        Reads from a raw COCO person_keypoints JSON (e.g. person_keypoints_train2017.json).
        """
        from collections import defaultdict
        with open(coco_kp_file) as f:
            kp_data = json.load(f)
        img_id_to_fn = {img["id"]: os.path.basename(img["file_name"])
                        for img in kp_data.get("images", [])}
        lookup = defaultdict(list)
        for ann in kp_data.get("annotations", []):
            if ann.get("num_keypoints", 0) > 0:
                fn = img_id_to_fn.get(ann["image_id"], "")
                lookup[fn].append(ann)
        logging.info(f"COCO KP lookup built: {len(lookup)} images with keypoints "
                     f"from {coco_kp_file}")
        return lookup

    def _inject_keypoints(self, ann_path: str, split_view):
        """
        Patch a COCO JSON file to add keypoint data.

        Priority:
          1. Raw COCO keypoints file (run_config["coco_keypoints_file"]) — most reliable
          2. FiftyOne sample field (run_config["keypoint_field"])          — if available
          3. Zeros                                                          — fallback
        """
        with open(ann_path, "r") as f:
            data = json.load(f)

        if not data.get("annotations"):
            logging.warning(
                f"No annotations found in {ann_path}. "
                "Check that the exported label_field matches the dataset's detection field. "
                "Common names: 'ground_truth', 'detections'. "
                "Skipping keypoint injection."
            )
            return

        # Enrich categories with keypoint metadata
        kp_names = self.keypoint_names or [f"kp_{i}" for i in range(self.num_keypoints)]
        for cat in data.get("categories", []):
            cat["keypoints"] = kp_names
            cat["skeleton"] = []

        # ── Option 1: raw COCO keypoints file ────────────────────────────
        coco_kp_file = self.run_config.get("coco_keypoints_file", None)
        coco_kp_lookup = None
        if coco_kp_file and os.path.exists(coco_kp_file):
            coco_kp_lookup = self._build_coco_kp_lookup(coco_kp_file)

        # Build filename → image_id map for this exported JSON
        fn_to_img_id = {img_info["file_name"]: img_info["id"]
                        for img_info in data.get("images", [])}
        img_id_to_fn = {v: k for k, v in fn_to_img_id.items()}

        # Build image_id → sample map (for FiftyOne field fallback)
        img_id_to_sample = {}
        for sample in split_view:
            fn = os.path.basename(sample.filepath)
            img_id = fn_to_img_id.get(fn)
            if img_id is not None:
                img_id_to_sample[img_id] = sample

        injected = 0
        # For each annotation, attach keypoints
        for ann in data["annotations"]:
            # ── Try raw COCO file first ───────────────────────────────────
            if coco_kp_lookup is not None:
                fn = os.path.basename(img_id_to_fn.get(ann["image_id"], ""))
                candidates = coco_kp_lookup.get(fn, [])
                best, best_dist = None, float("inf")
                ax, ay, aw, ah = ann["bbox"]
                for c in candidates:
                    cx, cy = c["bbox"][0], c["bbox"][1]
                    d = abs(cx - ax) + abs(cy - ay)
                    if d < best_dist:
                        best_dist, best = d, c
                if best and best_dist < max(aw, ah):  # sanity: close enough
                    kp = best["keypoints"]
                    if len(kp) < self.num_keypoints * 3:
                        kp = kp + [0] * (self.num_keypoints * 3 - len(kp))
                    ann["keypoints"] = kp[: self.num_keypoints * 3]
                    ann["num_keypoints"] = sum(
                        1 for i in range(2, len(ann["keypoints"]), 3)
                        if ann["keypoints"][i] > 0
                    )
                    injected += 1
                    continue

            # ── Try FiftyOne field ────────────────────────────────────────
            sample = img_id_to_sample.get(ann["image_id"])
            kp_field = getattr(sample, self.keypoint_field, None) if sample else None
            if kp_field is None:
                ann["keypoints"] = [0] * (self.num_keypoints * 3)
                ann["num_keypoints"] = 0
                continue

            # Get image dimensions for denormalisation
            img_w, img_h = Image.open(sample.filepath).size

            # Build keypoint list for this annotation
            kp_coco = self._extract_keypoints_for_annotation(
                ann, kp_field, img_w, img_h
            )
            ann["keypoints"] = kp_coco
            ann["num_keypoints"] = sum(1 for i in range(2, len(kp_coco), 3) if kp_coco[i] > 0)

        with open(ann_path, "w") as f:
            json.dump(data, f, indent=2)

        total = len(data["annotations"])
        logging.info(f"Injected keypoints into {ann_path} "
                     f"({injected}/{total} annotations have visible keypoints)")

    def _extract_keypoints_for_annotation(
        self,
        ann: dict,
        kp_field,
        img_w: int,
        img_h: int,
    ) -> List[float]:
        """
        Returns COCO keypoints list [x1,y1,v1, ...] for a single annotation.

        Matches fo.Keypoints by proximity of their bounding box to ann["bbox"].
        Falls back to zeros if no match found.
        """
        x, y, w, h = ann["bbox"]  # absolute pixel coords

        best_kp = None
        best_dist = float("inf")

        # kp_field is fo.Keypoints (a list of fo.Keypoint objects)
        kp_list = kp_field.keypoints if hasattr(kp_field, "keypoints") else []
        for kp_obj in kp_list:
            if not hasattr(kp_obj, "points") or kp_obj.points is None:
                continue
            # fo.Keypoint.points are normalised [(x,y), ...]
            pts = np.array(kp_obj.points)  # [N, 2] normalised
            cx_kp = pts[:, 0].mean() * img_w
            cy_kp = pts[:, 1].mean() * img_h
            cx_ann = x + w / 2
            cy_ann = y + h / 2
            dist = math.hypot(cx_kp - cx_ann, cy_kp - cy_ann)
            if dist < best_dist:
                best_dist = dist
                best_kp = kp_obj

        if best_kp is None:
            return [0] * (self.num_keypoints * 3)

        pts = best_kp.points or []
        result = []
        for i in range(self.num_keypoints):
            if i < len(pts):
                px, py = pts[i]
                # visibility: 2 = visible (FiftyOne doesn't encode occlusion by default)
                result.extend([px * img_w, py * img_h, 2])
            else:
                result.extend([0, 0, 0])
        return result

    def _ids_split_fraction(
        self, ids: list, fraction: float
    ) -> Tuple[list, list]:
        """
        Randomly split a list of sample IDs into two groups in memory.
        `fraction` of IDs go into the second group (e.g. val/test).
        The source dataset is never modified.
        """
        ids = list(ids)
        if len(ids) < 2:
            raise ValueError(f"Need at least 2 samples to split, got {len(ids)}.")
        random.seed(GLOBAL_SEED)
        random.shuffle(ids)
        n_second = max(1, int(len(ids) * fraction))
        return ids[n_second:], ids[:n_second]   # (keep, held-out)

    def _fix_annotation_indices(self, ann_path: str):
        """Convert 1-indexed COCO category IDs to 0-indexed (RF-DETR expectation)."""
        with open(ann_path, "r") as f:
            data = json.load(f)

        for cat in data["categories"]:
            if cat["id"] > 0:
                cat["id"] -= 1
        for ann in data["annotations"]:
            if ann["category_id"] > 0:
                ann["category_id"] -= 1

        with open(ann_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Training ──────────────────────────────────────────────────────

    def train(self, run_config: dict, shared_config: dict):
        """Train the dual-head RF-DETR model (multi-GPU via mp.spawn).

        Two modes:
          FO-native  (run_config["fo_native"] = True):
            Reads directly from the FiftyOne dataset — no COCO export.
            Pre-fetches all annotation data in the main process before
            spawning DDP workers (workers have no FiftyOne connection).

          COCO-export (default):
            Reads from the directory written by convert_data().
        """
        model_name = self.config_key.lower()
        output_dir = os.path.join(
            "output/models/rfdetr_kp", self.dataset_name, model_name
        )
        os.makedirs(output_dir, exist_ok=True)

        fo_prefetched = None

        if run_config.get("fo_native", False):
            # ── FiftyOne-native path ───────────────────────────────────
            import fiftyone as fo
            from workflows.keypoint.fo_keypoint_dataset import prefetch_fo_split

            detection_field = run_config.get("detection_field", "ground_truth")
            keypoint_field  = run_config.get("keypoint_field",  "pedestrian_points")
            target_label    = run_config.get("target_label",    "pedestrian")

            logging.info(f"FO-native mode: pre-fetching from '{self.dataset_name}' …")
            dataset = self.dataset   # fo.Dataset set on the workflow instance

            available_tags = dataset.distinct("tags")
            train_view = (dataset.match_tags("train")
                          if "train" in available_tags else dataset)
            val_view   = (dataset.match_tags("val")
                          if "val" in available_tags
                          else dataset.match_tags("validation")
                          if "validation" in available_tags else None)

            if val_view is None or len(val_view) == 0:
                # carve 10 % off train
                all_ids = train_view.values("id")
                n_val   = max(1, int(len(all_ids) * 0.1))
                val_ids   = all_ids[:n_val]
                train_ids = all_ids[n_val:]
                train_view = dataset.select(train_ids)
                val_view   = dataset.select(val_ids)
                logging.info(f"No 'val' tag — carved {n_val} val samples from train")

            kw = dict(
                detection_field=detection_field,
                keypoint_field=keypoint_field,
                target_label=target_label,
                num_keypoints=self.num_keypoints,
            )
            fo_prefetched = {
                "train": prefetch_fo_split(train_view, **kw),
                "valid": prefetch_fo_split(val_view,   **kw),
            }
            dataset_dir = ""   # not used in FO-native mode
        else:
            # ── COCO-export path ───────────────────────────────────────
            dataset_dir = os.path.join(
                self.export_dir_root, self.dataset_name, "rfdetr_kp"
            )
            if not os.path.exists(dataset_dir):
                raise FileNotFoundError(
                    f"Dataset directory not found: {dataset_dir}. "
                    "Run convert_data() first, or set fo_native=True in run_config."
                )

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        args = (dataset_dir, output_dir, run_config, shared_config,
                self.num_keypoints, self.keypoint_names, fo_prefetched)

        if n_gpus > 1:
            import torch.multiprocessing as mp
            mp.spawn(_ddp_train_worker, args=(n_gpus, *args), nprocs=n_gpus, join=True)
        else:
            _ddp_train_worker(0, max(n_gpus, 1), *args)

        self.model_path = os.path.join(output_dir, "best.pt")
        if HF_DO_UPLOAD:
            self._upload_to_hf()
        return True

    # ── Upload ────────────────────────────────────────────────────────

    def _upload_to_hf(self):
        if not hasattr(self, "model_path") or not os.path.exists(self.model_path):
            logging.warning("No model to upload.")
            return
        try:
            api = HfApi()
            api.create_repo(self.hf_repo_name, private=True, repo_type="model", exist_ok=True)
            api.upload_file(
                path_or_fileobj=self.model_path,
                path_in_repo="best.pt",
                repo_id=self.hf_repo_name,
                repo_type="model",
            )
            logging.info(f"Uploaded to {self.hf_repo_name}")
        except Exception as e:
            logging.error(f"HF upload failed: {e}")

    # ── Inference ─────────────────────────────────────────────────────

    def inference(
        self,
        inference_settings: dict,
        gt_field: str = "ground_truth",
    ):
        """
        Run dual-head inference: store bbox detections + keypoints in FiftyOne.

        Predictions are saved to:
          <pred_key>          : fo.Detections  (bounding boxes)
          <pred_key>_keypoints: fo.Keypoints   (associated keypoints per instance)
        """
        import fiftyone as fo  # lazy import — not needed by DDP workers
        logging.info(f"Running RF-DETR keypoint inference on {self.dataset_name}")

        model_name = self.config_key.lower()
        model_hf = inference_settings.get("model_hf", None)
        model_path_override = inference_settings.get("model_path", None)
        dataset_name = self.dataset_name

        # ── Locate model checkpoint ───────────────────────────────────
        if model_path_override is not None:
            if not os.path.exists(model_path_override):
                logging.error(f"model_path not found: {model_path_override}")
                return False
            model_path = model_path_override
            logging.info(f"Using model_path override: {model_path}")
        elif model_hf is not None:
            download_dir = os.path.join(
                "output/models/rfdetr_kp", dataset_name, model_name
            )
            os.makedirs(download_dir, exist_ok=True)
            try:
                model_path = hf_hub_download(
                    repo_id=model_hf,
                    filename="best.pt",
                    local_dir=download_dir,
                )
            except Exception as e:
                logging.error(f"HF download failed: {e}")
                return False
        else:
            possible = [
                os.path.join("output/models/rfdetr_kp", dataset_name, model_name, "best.pt"),
                os.path.join("output/models/rfdetr_kp", dataset_name, model_name, "last.pt"),
            ]
            model_path = next((p for p in possible if os.path.exists(p)), None)
            if model_path is None:
                logging.info(f"Local model not found. Trying {self.hf_repo_name} …")
                download_dir = os.path.join(
                    "output/models/rfdetr_kp", dataset_name, model_name
                )
                os.makedirs(download_dir, exist_ok=True)
                try:
                    model_path = hf_hub_download(
                        repo_id=self.hf_repo_name,
                        filename="best.pt",
                        local_dir=download_dir,
                    )
                except Exception as e:
                    logging.error(f"Cannot find or download model: {e}")
                    return False

        if not os.path.exists(model_path):
            logging.error(f"Model not found: {model_path}")
            return False

        # ── Load checkpoint ───────────────────────────────────────────
        logging.info(f"Loading model from {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")
        num_classes = ckpt.get("num_classes", 1)
        num_keypoints = ckpt.get("num_keypoints", self.num_keypoints)
        keypoint_names = ckpt.get("keypoint_names", self.keypoint_names)
        class_names = ckpt.get("class_names", None)
        ckpt_model_name = ckpt.get("model_name", model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = build_rfdetr_keypoint_model(
            pretrain_weights=None,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            keypoint_names=keypoint_names,
            rfdetr_config_name=ckpt_model_name,
            device=device,
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        logging.info("Model loaded")

        # ── Resolution + transforms ───────────────────────────────────
        resolution_map = {
            "rfdetr_nano": 560, "rfdetr_small": 560,
            "rfdetr_medium": 560, "rfdetr_base": 560, "rfdetr_large": 560,
        }
        resolution = inference_settings.get(
            "resolution", resolution_map.get(ckpt_model_name, 560)
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

        def preprocess(pil_img):
            pil_img = pil_img.resize((resolution, resolution), Image.BILINEAR)
            t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            return ((t.to(device) - mean) / std).unsqueeze(0)

        # ── Dataset view ──────────────────────────────────────────────
        threshold = inference_settings.get("detection_threshold", 0.2)

        if inference_settings.get("inference_on_test", True):
            dataset_view = self.dataset.match_tags(["test"])
            if len(dataset_view) == 0:
                # Tags were never persisted (in-memory split). Try to reconstruct
                # the test set from the exported COCO annotation file.
                export_dir = os.path.join(self.export_dir_root, dataset_name, "rfdetr_kp")
                test_ann = os.path.join(export_dir, "test", "_annotations.coco.json")
                if os.path.exists(test_ann):
                    with open(test_ann) as _f:
                        _test_data = json.load(_f)
                    test_filenames = {img["file_name"] for img in _test_data.get("images", [])}
                    all_fps = self.dataset.values("filepath")
                    all_ids = self.dataset.values("id")
                    test_ids = [sid for sid, fp in zip(all_ids, all_fps)
                                if os.path.basename(fp) in test_filenames]
                    dataset_view = self.dataset.select(test_ids)
                    logging.info(f"Reconstructed test view from COCO export: {len(dataset_view)} samples")
                if len(dataset_view) == 0:
                    logging.warning("No test samples found via tags or export; running inference on full dataset.")
                    dataset_view = self.dataset
        else:
            dataset_view = self.dataset

        pred_key = f"pred_kp_{model_name}-{dataset_name}"
        pred_kp_key = f"{pred_key}_keypoints"

        # ── Run inference ─────────────────────────────────────────────
        processed = 0
        try:
            for sample in tqdm(
                dataset_view.iter_samples(progress=True, autosave=True),
                total=len(dataset_view),
                desc="RF-DETR Keypoint Inference",
            ):
                try:
                    pil_img = Image.open(sample.filepath).convert("RGB")
                    img_w, img_h = pil_img.size

                    inp = preprocess(pil_img)

                    with torch.no_grad():
                        out = model(inp)

                    pred_logits = out["pred_logits"][0]   # [Q, C]
                    pred_boxes  = out["pred_boxes"][0]    # [Q, 4]  cx-cy-w-h norm
                    pred_kpts   = out["pred_keypoints"][0]  # [Q, K, 3]

                    # Score via max-class sigmoid
                    scores = pred_logits.sigmoid().max(-1).values  # [Q]
                    class_ids = pred_logits.sigmoid().argmax(-1)    # [Q]

                    keep = scores > threshold

                    fo_dets = []
                    fo_kps  = []

                    for i in torch.where(keep)[0]:
                        cx, cy, w, h = pred_boxes[i].cpu().tolist()
                        score = scores[i].item()
                        cid   = class_ids[i].item()

                        # Convert to fo: [x_top_left, y_top_left, w, h] relative
                        bx = cx - w / 2
                        by = cy - h / 2

                        label = (
                            class_names[cid]
                            if class_names and cid < len(class_names)
                            else f"class_{cid}"
                        )
                        fo_dets.append(
                            fo.Detection(
                                label=label,
                                bounding_box=[bx, by, w, h],
                                confidence=score,
                            )
                        )

                        # Keypoints for this query
                        kp = pred_kpts[i].cpu()  # [K, 3]
                        kp_xy   = kp[:, :2].tolist()   # [[x_norm, y_norm], ...]
                        kp_vis  = kp[:, 2].sigmoid().tolist()

                        fo_kps.append(
                            fo.Keypoint(
                                label=label,
                                points=kp_xy,
                                confidence=kp_vis,
                                keypoint_names=keypoint_names,
                            )
                        )

                    sample[pred_key]    = fo.Detections(detections=fo_dets)
                    sample[pred_kp_key] = fo.Keypoints(keypoints=fo_kps)
                    processed += 1

                except Exception as e:
                    logging.error(f"Error on sample {sample.id}: {e}")
                    continue

        except Exception as e:
            logging.error(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return False

        logging.info(
            f"Inference complete: {processed}/{len(dataset_view)} samples. "
            f"Detections → '{pred_key}', Keypoints → '{pred_kp_key}'"
        )

        # ── Evaluation (bbox only) ────────────────────────────────────
        if inference_settings.get("do_eval", True):
            eval_view = (
                self.dataset.match_tags(["test"])
                if inference_settings.get("inference_on_test", True)
                else self.dataset
            )
            eval_view = eval_view.exists(pred_key).exists(gt_field)
            if len(eval_view) > 0:
                eval_key = f"eval_kp_{model_name}_{dataset_name}".replace("-", "_")
                try:
                    results = eval_view.evaluate_detections(
                        pred_key,
                        gt_field=gt_field,
                        eval_key=eval_key,
                        compute_mAP=True,
                        iou=0.5,
                    )
                    logging.info("=" * 70)
                    logging.info("BBOX EVALUATION RESULTS")
                    results.print_report()
                    logging.info("=" * 70)
                except Exception as e:
                    logging.error(f"Evaluation failed: {e}")

        return True

