import math
from typing import List, Optional

from msight_base.detection import DetectedObjectBase, DetectionResultBase


class DetectedObject2D(DetectedObjectBase):
    """MSight 2D detection built from a FiftyOne bounding box.

    Stores pixel-space bbox (x1, y1, x2, y2) and the original normalized bbox
    so the result can be written back to FiftyOne without re-scaling.
    lat/lon are filled in by the localization step.
    """

    def __init__(
        self,
        label: str,
        confidence: Optional[float],
        bbox_norm: List[float],
        bbox_xyxy: List[float],
        sensor_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
    ):
        super().__init__()
        self.label = label
        self.confidence = confidence
        self.bbox_norm = bbox_norm      # [x, y, w, h] in [0, 1]
        self.bbox_xyxy = bbox_xyxy      # [x1, y1, x2, y2] in pixels
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.lat: float = math.nan
        self.lon: float = math.nan
        self.contact_px: Optional[tuple] = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox_norm": self.bbox_norm,
            "bbox_xyxy": self.bbox_xyxy,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "lat": self.lat,
            "lon": self.lon,
        }

    @staticmethod
    def from_dict(data: dict) -> "DetectedObject2D":
        obj = DetectedObject2D(
            label=data["label"],
            confidence=data.get("confidence"),
            bbox_norm=data["bbox_norm"],
            bbox_xyxy=data["bbox_xyxy"],
            sensor_id=data.get("sensor_id"),
            sensor_type=data.get("sensor_type"),
        )
        obj.lat = data.get("lat", math.nan)
        obj.lon = data.get("lon", math.nan)
        return obj


def fo_detections_to_msight(
    fo_detections,
    img_width: int,
    img_height: int,
    timestamp: int,
    sensor_id: Optional[str] = None,
    sensor_type: str = "fisheye",
) -> DetectionResultBase:
    """Convert a FiftyOne Detections label to a MSight DetectionResultBase.

    FiftyOne bounding boxes are [x, y, w, h] in normalized [0, 1] coordinates.
    Each box is converted to pixel [x1, y1, x2, y2] for localization.
    """
    if fo_detections is None or not fo_detections.detections:
        return DetectionResultBase(
            object_list=[], timestamp=timestamp, sensor_type=sensor_type
        )

    object_list: List[DetectedObject2D] = []
    for det in fo_detections.detections:
        x_n, y_n, w_n, h_n = det.bounding_box
        x1 = x_n * img_width
        y1 = y_n * img_height
        x2 = (x_n + w_n) * img_width
        y2 = (y_n + h_n) * img_height

        obj = DetectedObject2D(
            label=det.label,
            confidence=det.confidence,
            bbox_norm=[x_n, y_n, w_n, h_n],
            bbox_xyxy=[x1, y1, x2, y2],
            sensor_id=sensor_id,
            sensor_type=sensor_type,
        )
        object_list.append(obj)

    return DetectionResultBase(
        object_list=object_list,
        timestamp=timestamp,
        sensor_type=sensor_type,
    )
