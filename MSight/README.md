# MSIGHT — Localization Module

Offline localization pipeline for camera-based object detection stored in [FiftyOne](https://docs.voxel51.com/) datasets.
Ported from the standalone `MSight-data-engine` repository and integrated into the `MSight-data-engine-rf-detr` workflow.

Given a FiftyOne dataset with 2-D bounding-box detections and an NPZ camera-calibration file, the pipeline:
1. Converts each detection to an MSight `DetectedObject2D`.
2. Estimates the fisheye ground-contact pixel using camera intrinsics.
3. Maps pixel coordinates to lat/lon via interpolation over calibration control points.
4. Writes localized detections and keypoints back to the FiftyOne dataset.

---

## Directory structure

```
MSIGHT/
├── data/
│   ├── ashley_huron_intrinsic.json          # Fisheye camera intrinsics (f, x0, y0)
│   └── calibration_results_ashley_huron.npz # Calibration file (lat_map, lon_map)
├── utils/
│   ├── __init__.py
│   ├── fiftyone_to_msight_det.py            # FiftyOne → MSight detection conversion
│   └── load_locamaps.py                     # NPZ loader, intrinsics loader, pixel localizer
├── localize_dataset.py                      # Core localization logic + FiftyOne write-back
├── install.sh                               # Install script for MSight dependencies
├── requirements.txt                         # Python package requirements
└── README.md
```

---

## Installation

Install MSight dependencies into the **active** Python environment before enabling localization:

```bash
# Activate your project virtual environment first (if applicable)
source venv/bin/activate

# Then run the install script
bash MSIGHT/install.sh
```

Alternatively install directly with pip:

```bash
pip install -r MSIGHT/requirements.txt
```

### Key dependencies

| Package | Purpose |
|---------|---------|
| `msight_base` | `DetectedObjectBase`, `DetectionResultBase` |
| `msight_core` | MSight messaging infrastructure |
| `scipy` | `LinearNDInterpolator` / `NearestNDInterpolator` for sparse-map localization |
| `numpy` | NPZ map loading and array operations |

---

## Configuration

Set `run_localization: True` in `config/config.py` to enable the pipeline after running workflows:

```python
MSIGHT_CONFIG = {
    "run_localization": True,
    "detection_field": "pred_od_rfdetr_2xlarge-gs_catherine_zina1",  # fo.Detections field to localize
    "loc_maps": "MSIGHT/data/calibration_results_ashley_huron.npz",  # path to calibration npz
    "intrinsics": "MSIGHT/data/ashley_huron_intrinsic.json",         # path to intrinsics json
}
```

| Key | Description |
|-----|-------------|
| `run_localization` | `True` to run localization after workflows complete |
| `detection_field` | FiftyOne field holding `fo.Detections` to localize |
| `loc_maps` | Project-relative or absolute path to the `.npz` calibration file |
| `intrinsics` | Project-relative or absolute path to the camera intrinsics JSON |

---

## Camera intrinsics format

```json
{ "f": 320, "x0": 645, "y0": 473 }
```

| Key | Description |
|-----|-------------|
| `f`  | Focal length in pixels |
| `x0` | Pixel column of the fisheye circle centre |
| `y0` | Pixel row of the fisheye circle centre |

---

## Calibration file format

Each `.npz` must contain two arrays of shape `(H, W)` matching the camera resolution:

| Array | dtype | Description |
|-------|-------|-------------|
| `lat_map` | float64 | Latitude at each pixel (non-calibrated pixels are `-inf`) |
| `lon_map` | float64 | Longitude at each pixel (non-calibrated pixels are `-inf`) |

The pipeline automatically handles sparse calibration maps via `LinearNDInterpolator`
with a `NearestNDInterpolator` fallback.

---

## Output fields

The pipeline writes **two** new fields per sample to the FiftyOne dataset:

| Field | Type | Description |
|-------|------|-------------|
| `msight_<detection_field>` | `fo.Detections` | Localized detections with `lat`/`lon` attributes |
| `msight_<detection_field>_keypoints` | `fo.Keypoints` | One keypoint per detection at the fisheye ground-contact pixel |

---

## Fisheye ground-contact logic

For a fisheye camera mounted overhead, "down" in the image is radially inward toward the
optical centre `(x0, y0)`. The ground-contact pixel is the bounding-box boundary point along
the outward radial ray from `(x0, y0)` through the box centre — implemented in
`localize_dataset.fisheye_ground_contact(bbox_xyxy, x0, y0)`.

---

## Module reference

### `utils/load_locamaps.py`

| Function | Description |
|----------|-------------|
| `load_intrinsics(path)` | Loads camera intrinsics JSON; returns `{'f', 'x0', 'y0'}`. |
| `load_locmaps(path)` | Loads NPZ calibration file; returns `(lat_map, lon_map)`. |
| `build_pixel_localizer(lat_map, lon_map)` | Returns a callable `localize(cx, cy) -> (lat, lon)`. |

### `utils/fiftyone_to_msight_det.py`

Converts `fo.Detections` to `DetectionResultBase` containing `DetectedObject2D` objects.

### `localize_dataset.py`

| Function | Description |
|----------|-------------|
| `fisheye_ground_contact(bbox_xyxy, x0, y0)` | Returns fisheye-corrected ground-contact pixel. |
| `localize_detection_result(...)` | Fills lat/lon on each detection via the localizer. |
| `build_fo_detections_from_msight(...)` | Converts localized `DetectionResultBase` to `fo.Detections`. |
| `build_fo_keypoints_from_msight(...)` | Builds `fo.Keypoints` from contact pixels. |
| `run_localization(...)` | Iterates every sample; writes detections and keypoints fields. |
