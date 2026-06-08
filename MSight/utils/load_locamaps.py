import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def _resolve_path(configured: str) -> Path:
    """Return a resolved Path, falling back to project-relative if absolute path missing."""
    path = Path(configured)
    if not path.exists() and configured.startswith("/"):
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / configured.lstrip("/")
    return path


def load_intrinsics(intrinsics_path: str) -> Dict[str, float]:
    """Load fisheye camera intrinsics from a JSON file.

    Expected keys:
        f  – focal length in pixels
        x0 – fisheye centre x (pixel column of the circle centre)
        y0 – fisheye centre y (pixel row of the circle centre)
    """
    path = _resolve_path(intrinsics_path)
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {intrinsics_path}")
    with open(path) as fh:
        data = json.load(fh)
    return {k: float(v) for k, v in data.items()}


def load_locmaps(loc_maps_path: str):
    """Load lat/lon localization maps from an npz calibration file.

    The file must contain 'lat_map' and 'lon_map' arrays of shape (H, W).
    Returns (lat_map, lon_map) as float64 numpy arrays.
    """
    path = _resolve_path(loc_maps_path)
    if not path.exists():
        raise FileNotFoundError(f"Localization map not found: {loc_maps_path}")

    data = np.load(path)
    return data["lat_map"], data["lon_map"]


def build_pixel_localizer(
    lat_map: np.ndarray,
    lon_map: np.ndarray,
) -> Callable[[float, float], Tuple[float, float]]:
    """Build a (cx, cy) -> (lat, lon) callable from a sparse calibration map.

    Extracts finite control points and builds a LinearNDInterpolator with a
    NearestNDInterpolator fallback so every detection receives a finite lat/lon.
    """
    valid = np.isfinite(lat_map) & np.isfinite(lon_map)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("Calibration map contains no finite lat/lon values.")

    ys, xs = np.where(valid)
    points = np.column_stack([xs.astype(float), ys.astype(float)])
    lat_vals = lat_map[valid]
    lon_vals = lon_map[valid]

    print(
        f"  Calibration map: {n_valid} valid control points "
        f"(x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}) "
        f"out of {lat_map.size} total pixels."
    )

    lat_linear = LinearNDInterpolator(points, lat_vals)
    lon_linear = LinearNDInterpolator(points, lon_vals)
    lat_nearest = NearestNDInterpolator(points, lat_vals)
    lon_nearest = NearestNDInterpolator(points, lon_vals)

    def localize(cx: float, cy: float) -> Tuple[float, float]:
        p = np.array([[cx, cy]])
        lat = float(lat_linear(p)[0])
        lon = float(lon_linear(p)[0])
        if not (np.isfinite(lat) and np.isfinite(lon)):
            lat = float(lat_nearest(p)[0])
            lon = float(lon_nearest(p)[0])
        return lat, lon

    return localize
