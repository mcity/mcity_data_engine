"""
Localization pipeline for FiftyOne datasets.

Workflow per sample:
  1. Read detections from detection_field.
  2. Convert to MSight DetectionResultBase (pixel bboxes, DetectedObject2D).
  3. For each detection compute the fisheye-corrected ground-contact pixel
     using the camera intrinsics (x0, y0) — see fisheye_ground_contact().
  4. Look up lat/lon at that pixel via the interpolator built from the
     sparse calibration control points.
  5. Drop detections where lat/lon is not finite.
  6. Write localized detections to msight_field (fo.Detections).
  7. Write a companion fo.Keypoints field (msight_field + "_keypoints") with
     one keypoint per detection at the fisheye ground-contact pixel.

Why fisheye-corrected bottom centre?
-------------------------------------
In a rectilinear image the ground-contact point of a bounding box is the
axis-aligned bottom-centre (cx, y2). In a fisheye image the projected "down"
direction is *radially away from the fisheye optical centre* (x0, y0). The
correct contact pixel is therefore the point on the bounding-box boundary
that lies along the outward radial ray from (x0, y0) through the box centre.
"""

import copy
import math
import time
from typing import Callable, Tuple

import fiftyone as fo

from msight_base.detection import DetectionResultBase

from MSight.utils.fiftyone_to_msight_det import fo_detections_to_msight, DetectedObject2D


def _is_valid(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def fisheye_ground_contact(
    bbox_xyxy: list,
    x0: float,
    y0: float,
) -> Tuple[float, float]:
    """Return the fisheye-corrected ground-contact pixel for a bounding box.

    Finds the intersection of the outward radial ray from (x0, y0) through
    the box centre with the bounding-box boundary (farthest edge from the
    optical centre).
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    dx, dy = cx - x0, cy - y0
    d = math.hypot(dx, dy)

    if d < 1e-6:
        return cx, y2

    ux, uy = dx / d, dy / d

    eps = 1e-9
    t_best = None

    if abs(ux) > eps:
        for xe in (x1, x2):
            t = (xe - x0) / ux
            if t > eps:
                py = y0 + t * uy
                if y1 - eps <= py <= y2 + eps:
                    if t_best is None or t < t_best:
                        t_best = t

    if abs(uy) > eps:
        for ye in (y1, y2):
            t = (ye - y0) / uy
            if t > eps:
                px = x0 + t * ux
                if x1 - eps <= px <= x2 + eps:
                    if t_best is None or t < t_best:
                        t_best = t

    if t_best is None:
        return cx, y2

    gx = max(x1, min(x2, x0 + t_best * ux))
    gy = max(y1, min(y2, y0 + t_best * uy))
    return gx, gy


def localize_detection_result(
    detection_result: DetectionResultBase,
    localizer: Callable[[float, float], Tuple[float, float]],
    x0: float,
    y0: float,
) -> DetectionResultBase:
    """Fill lat/lon on each DetectedObject2D using fisheye-corrected contact pixel."""
    for obj in copy.copy(detection_result.object_list):
        gx, gy = fisheye_ground_contact(obj.bbox_xyxy, x0, y0)
        obj.contact_px = (gx, gy)
        obj.lat, obj.lon = localizer(gx, gy)

    detection_result.object_list = [
        obj for obj in detection_result.object_list
        if _is_valid(obj.lat) and _is_valid(obj.lon)
    ]
    return detection_result


def build_fo_detections_from_msight(detection_result: DetectionResultBase) -> fo.Detections:
    """Convert a localized DetectionResultBase back to fo.Detections with lat/lon."""
    fo_dets = []
    for obj in detection_result.object_list:
        det = fo.Detection(
            label=obj.label,
            bounding_box=obj.bbox_norm,
            confidence=obj.confidence,
        )
        det["lat"] = obj.lat
        det["lon"] = obj.lon
        fo_dets.append(det)
    return fo.Detections(detections=fo_dets)


def build_fo_keypoints_from_msight(
    detection_result: DetectionResultBase,
    img_width: int,
    img_height: int,
) -> fo.Keypoints:
    """Build a fo.Keypoints field from localized MSight detections.

    Each keypoint is placed at the fisheye-corrected ground-contact pixel,
    normalized to [0, 1] by the image dimensions.
    """
    keypoints = []
    for obj in detection_result.object_list:
        if obj.contact_px is not None:
            gx, gy = obj.contact_px
        else:
            x_n, y_n, w_n, h_n = obj.bbox_norm
            gx = (x_n + w_n / 2.0) * img_width
            gy = (y_n + h_n) * img_height

        kp = fo.Keypoint(
            label=obj.label,
            points=[[gx / img_width, gy / img_height]],
            confidence=[obj.confidence] if obj.confidence is not None else None,
        )
        kp["lat"] = obj.lat
        kp["lon"] = obj.lon
        keypoints.append(kp)
    return fo.Keypoints(keypoints=keypoints)


def run_localization(
    dataset: fo.Dataset,
    detection_field: str,
    msight_field: str,
    localizer: Callable[[float, float], Tuple[float, float]],
    x0: float,
    y0: float,
    sensor_type: str = "fisheye",
) -> None:
    """Iterate over all samples, localize detections, and write two fields.

    Writes:
      msight_field            – fo.Detections with lat/lon attributes
      msight_field_keypoints  – fo.Keypoints at the fisheye ground-contact pixel
    """
    keypoints_field = f"{msight_field}_keypoints"
    total = len(dataset)
    print(
        f"Processing {total} samples: "
        f"'{detection_field}' -> '{msight_field}' + '{keypoints_field}'"
    )
    print(f"Fisheye optical centre: x0={x0}, y0={y0}")

    for sample in dataset.iter_samples(progress=True, autosave=True):
        fo_dets = sample.get_field(detection_field)

        if fo_dets is None or not fo_dets.detections:
            sample[msight_field] = fo.Detections(detections=[])
            sample[keypoints_field] = fo.Keypoints(keypoints=[])
            continue

        metadata = sample.metadata
        if metadata is None:
            sample.compute_metadata()
            metadata = sample.metadata

        img_w = metadata.width
        img_h = metadata.height
        timestamp = int(time.time() * 1000)
        sensor_id = sample.get_field("sensor") if "sensor" in sample.field_names else None

        detection_result = fo_detections_to_msight(
            fo_dets,
            img_width=img_w,
            img_height=img_h,
            timestamp=timestamp,
            sensor_id=str(sensor_id) if sensor_id is not None else None,
            sensor_type=sensor_type,
        )

        localize_detection_result(detection_result, localizer, x0, y0)

        sample[msight_field] = build_fo_detections_from_msight(detection_result)
        sample[keypoints_field] = build_fo_keypoints_from_msight(
            detection_result, img_w, img_h
        )

    print(f"Done. Detections -> '{msight_field}', keypoints -> '{keypoints_field}'.")
