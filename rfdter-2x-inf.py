import fiftyone as fo
import cv2
import os

# === CONFIGURATION ===
DATASET_NAME = "gs_catherine_glen1-sample-1"  # Replace with your actual dataset name
LABEL_FIELD = "pred_od_rfdetr_2xlarge-"+DATASET_NAME  # Change if your detection field has a different name
OUTPUT_VIDEO_PATH = "output_video.mp4"
FPS = 10  # Change as needed

# === LOAD DATASET ===
print(f"Loading FiftyOne dataset '{DATASET_NAME}'...")
print(f"Loading FiftyOne dataset '{DATASET_NAME}'...")
dataset = fo.load_dataset(DATASET_NAME)

filepaths = []
detections_sets = []

for sample in dataset:
    detections_doc = getattr(sample, LABEL_FIELD, None)
    # Check if the field exists and has detections
    if detections_doc is not None and hasattr(detections_doc, "detections"):
        filepaths.append(sample.filepath)
        detections_sets.append(detections_doc.detections)
    else:
        print(f"[WARNING] Sample {sample.id} has no '{LABEL_FIELD}' field or is empty. Skipping.")

if not filepaths:
    raise ValueError("No images with detection information found!")

# ===== DRAW DETECTIONS =====
drawn_images = []

for img_path, detections in zip(filepaths, detections_sets):
    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARNING] Failed to read {img_path}. Skipping.")
        continue

    ih, iw = image.shape[:2]

    for det in detections:
        # det.bounding_box: [x, y, w, h] (normalized)
        x_norm, y_norm, w_norm, h_norm = det.bounding_box
        x = int(x_norm * iw)
        y = int(y_norm * ih)
        w = int(w_norm * iw)
        h = int(h_norm * ih)
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        # Optionally include confidence
        label = det.label
        conf = getattr(det, "confidence", None)
        if conf is not None:
            label_text = f"{label} ({conf:.2f})"
        else:
            label_text = f"{label}"
        cv2.putText(image, label_text, (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    drawn_images.append(image)

if not drawn_images:
    raise RuntimeError("No drawn images available for video creation.")

# ===== MAKE VIDEO =====
print(f"Writing video to {OUTPUT_VIDEO_PATH}...")
height, width = drawn_images[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (width, height))

for img in drawn_images:
    out.write(img)
out.release()
print(f"DONE! Video created: {OUTPUT_VIDEO_PATH}")

