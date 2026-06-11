import cv2
import fiftyone as fo

from config.config import SELECTED_DATASET,MSIGHT_CONFIG

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Dataset and field names are derived from config.py.
DATASET_NAME    = SELECTED_DATASET["name"]
DETECTION_FIELD = MSIGHT_CONFIG["detection_field"]

# MSight localized fields written by main.py / localize_dataset.py
LABEL_FIELD      = f"msight_{DETECTION_FIELD}"           # fo.Detections with lat/lon
KEYPOINTS_FIELD  = f"msight_{DETECTION_FIELD}_keypoints" # fo.Keypoints (fisheye contact pixel)

# Only render these classes.  Add/remove labels as needed.
CLASSES_TO_RENDER: list[str] = ["pedestrian"]

OUTPUT_VIDEO = f"{DATASET_NAME}-msight-output.mp4"
FPS          = 10

# Drawing style
BOX_COLOR      = (0, 255, 0)    # BGR green  – bounding box
KP_COLOR       = (0, 0, 255)    # BGR red    – keypoint dot
TEXT_COLOR     = (0, 255, 0)
BOX_THICKNESS  = 2
KP_RADIUS      = 6
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.6
FONT_THICKNESS = 2

# ─── LOAD DATASET ─────────────────────────────────────────────────────────────

print(f"Loading dataset '{DATASET_NAME}' ...")
dataset = fo.load_dataset(DATASET_NAME)

print(f"  Detections field : {LABEL_FIELD}")
print(f"  Keypoints field  : {KEYPOINTS_FIELD}")
print(f"  Classes to render: {CLASSES_TO_RENDER}")

# ─── COLLECT FRAMES ──────────────────────────────────────────────────────────

frames = []   # list of (filepath, [(det, kp_point), ...])

for sample in dataset:
    dets_doc = sample.get_field(LABEL_FIELD)
    kps_doc  = sample.get_field(KEYPOINTS_FIELD)

    if dets_doc is None or not dets_doc.detections:
        continue

    detections = dets_doc.detections
    keypoints  = kps_doc.keypoints if (kps_doc and kps_doc.keypoints) else []

    # Pair each detection with its keypoint (same index, same order).
    # Filter to requested classes.
    pairs = []
    for i, det in enumerate(detections):
        if det.label not in CLASSES_TO_RENDER:
            continue
        kp_point = keypoints[i].points[0] if i < len(keypoints) else None
        pairs.append((det, kp_point))

    if pairs:
        frames.append((sample.filepath, pairs))

if not frames:
    raise ValueError(
        f"No samples found with class(es) {CLASSES_TO_RENDER} in '{LABEL_FIELD}'."
    )

print(f"Rendering {len(frames)} frame(s) ...")

# ─── DRAW AND ENCODE ──────────────────────────────────────────────────────────

drawn: list = []

for filepath, pairs in frames:
    image = cv2.imread(filepath)
    if image is None:
        print(f"[WARNING] Cannot read {filepath}. Skipping.")
        continue

    ih, iw = image.shape[:2]

    for det, kp_point in pairs:
        x_n, y_n, w_n, h_n = det.bounding_box
        x1 = int(x_n * iw)
        y1 = int(y_n * ih)
        x2 = int((x_n + w_n) * iw)
        y2 = int((y_n + h_n) * ih)

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Label + confidence
        conf = getattr(det, "confidence", None)
        text = f"{det.label} ({conf:.2f})" if conf is not None else det.label
        cv2.putText(image, text, (x1, max(y1 - 8, 0)),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

        # Fisheye ground-contact keypoint from the MSight keypoints field.
        # kp_point is [bx_norm, by_norm] computed by localize_dataset.py
        # using fisheye_ground_contact() with camera intrinsics.
        if kp_point is not None:
            kpx = int(kp_point[0] * iw)
            kpy = int(kp_point[1] * ih)
            cv2.circle(image, (kpx, kpy), KP_RADIUS, KP_COLOR, -1)

    drawn.append(image)

if not drawn:
    raise RuntimeError("No frames were rendered.")

# ─── WRITE VIDEO ──────────────────────────────────────────────────────────────

h, w = drawn[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

for img in drawn:
    writer.write(img)
writer.release()

print(f"Done. Video saved to: {OUTPUT_VIDEO}")
