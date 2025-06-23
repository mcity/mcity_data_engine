import cv2

import os
'''
image_dir = "yolo_pose/images/train"
label_dir = "yolo_pose/labels/train"

# Normalize file lists (ignore hidden files)
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(".txt")]

# Extract base filenames (without extension)
image_basenames = set(os.path.splitext(f)[0] for f in image_files)
label_basenames = set(os.path.splitext(f)[0] for f in label_files)

# Images without matching labels
unmatched = image_basenames - label_basenames

print(f"🗑 Removing {len(unmatched)} images without labels...")

removed = 0
for base_name in unmatched:
    img_path = os.path.join(image_dir, base_name + ".jpg")
    if os.path.exists(img_path):
        try:
            os.remove(img_path)
            removed += 1
        except Exception as e:
            print(f"⚠️ Could not remove {img_path}: {e}")

print(f"✅ Removed {removed} unmatched images.")


# --- CONFIGURATION ---
image_path = "yolo_pose/images/train/gs_Main_stadium2_2023-09-23 12-39-04-824124.jpg"          # change this
label_path = "yolo_pose/labels/train/gs_Main_stadium2_2023-09-23 12-39-04-824124.txt"          # corresponding label
class_names = ["bus", "car", "motorbike/cycler", "pedestrian", "pickup", "trailer", "truck", "van"]

# --- LOAD IMAGE ---
img = cv2.imread(image_path)
h, w = img.shape[:2]

# --- READ LABELS ---
with open(label_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    cls_id = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:5])
    kp_x, kp_y, vis = map(float, parts[5:8])

    # Convert bbox from normalized to pixel
    x_center, y_center = int(cx * w), int(cy * h)
    box_width, box_height = int(bw * w), int(bh * h)
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = class_names[cls_id]
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw keypoint if visible
    if int(vis) > 0:
        kp_px, kp_py = int(kp_x * w), int(kp_y * h)
        cv2.circle(img, (kp_px, kp_py), 2, (0, 0, 255), -1)
        cv2.putText(img, f"KP", (kp_px + 5, kp_py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# --- SHOW ---
cv2.imshow("YOLOv8-Pose Visualization", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import os

def remove_visibility_from_labels(label_dir):
    fixed = 0
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_dir, filename)
            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 8:
                    # Remove the last element (visibility)
                    new_line = " ".join(parts[:7])
                    new_lines.append(new_line)
                    fixed += 1
                else:
                    new_lines.append(line.strip())

            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

    return f"✅ Removed visibility from {fixed} label lines in '{label_dir}'"

# Run the function on your label directory
remove_visibility_from_labels("yolo_pose/labels/val")
'''

'''
from ultralytics import YOLO

# Load the trained pose model
model = YOLO("/home/dataengine/Downloads/mcity_data_engine-1/runs/pose/pose_keypoint_train5/weights/best.pt")

# Path to test images
test_images = "yolo_pose/images/val"

# Step 1: Run predict once to initialize the predictor
results = model.predict(source=test_images, stream=True)

# Step 2: Set visualization parameters
model.predictor.args.line_width = 1
model.predictor.args.kp_line_thickness = 1
model.predictor.args.kp_radius = 1
model.predictor.args.font_size = 0.4

# Step 3: Run actual prediction with save=True
model.predict(
    source=test_images,
    save=True,
    save_txt=False,
    save_conf=True,
    imgsz=640,
    conf=0.25,
    show=False
)

print("✅ Inference complete. Visualizations saved to:")
print(model.predictor.save_dir)
'''

'''
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path

# Load models
detector = YOLO("/home/dataengine/Downloads/mcity_data_engine-1/output/models/ultralytics/mcity_2844_clean_crowd_updated/yolo12x/weights/best.pt")  # your stronger object detector
pose_model = YOLO("/home/dataengine/Downloads/mcity_data_engine-1/runs/pose/pose_keypoint_train5/weights/best.pt")
# Paths
test_images = Path("yolo_pose/images/val")
output_dir = Path("combined_output")
output_dir.mkdir(parents=True, exist_ok=True)

# Colors
box_color = (255, 0, 0)
kp_color = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Process each image
for img_path in test_images.glob("*.jpg"):
    image = cv2.imread(str(img_path))
    orig_img = image.copy()
    h, w = image.shape[:2]

    # Step 1: Object detection
    det_result = detector.predict(source=image, conf=0.2, imgsz=640, verbose=False)[0]
    boxes = det_result.boxes
    if boxes is None or boxes.shape[0] == 0:
        continue

    for i, box in enumerate(boxes):
        # Extract box coordinates and class
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        # Crop the region for pose detection
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (640, 640))

        # Pose prediction on cropped region
        pose_result = pose_model.predict(source=crop_resized, imgsz=640, conf=0.3, verbose=False)[0]
        if pose_result.keypoints is None or len(pose_result.keypoints.xy) == 0:
            continue

        # Get keypoints and scale to original image
        keypoints = pose_result.keypoints.xy[0].cpu().numpy()  # (num_kpts, 2)

        scale_x = (x2 - x1) / 640
        scale_y = (y2 - y1) / 640

        for x_kp, y_kp in keypoints:
            kp_x = int(x1 + x_kp * scale_x)
            kp_y = int(y1 + y_kp * scale_y)
            cv2.circle(orig_img, (kp_x, kp_y), radius=3, color=kp_color, thickness=-1)

        # Draw bounding box
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), box_color, 2)

        # Optional: label class ID and confidence
        label = f"{detector.model.names[cls_id]} {conf:.2f}"
        cv2.putText(orig_img, label, (x1, y1 - 10), font, 0.5, box_color, 1, cv2.LINE_AA)

    # Save image
    save_path = output_dir / img_path.name
    cv2.imwrite(str(save_path), orig_img)

print(f"✅ Visualizations saved to: {output_dir}")
'''
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import pandas as pd
import os
from torchvision import models
import torch.nn as nn

# ---------------- Load CSV ---------------- #
df = pd.read_csv("resnet_keypoints.csv")
class_to_idx = {cls: idx for idx, cls in enumerate(sorted(df['class'].unique()))}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ---------------- Load Model ---------------- #
def get_model(num_classes=8):
    resnet = models.resnet18(weights=None)  # pretrained=False is deprecated
    resnet.conv1 = nn.Conv2d(3 + num_classes, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    return resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=8)
model.load_state_dict(torch.load("best_resnet_keypoint.pth", map_location=device))
model.to(device)
model.eval()

# ---------------- Transform ---------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- Visualize Predictions ---------------- #
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms

def visualize_prediction(image_path, class_label, true_x, true_y, model, class_to_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Load image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)

    # Get class one-hot
    class_id = class_to_idx[class_label]
    class_onehot = torch.nn.functional.one_hot(torch.tensor(class_id), num_classes=len(class_to_idx)).float()
    class_onehot_expanded = class_onehot.view(-1, 1, 1).expand(-1, image_tensor.shape[1], image_tensor.shape[2])

    # Combine image + class info
    input_tensor = torch.cat([image_tensor, class_onehot_expanded], dim=0).unsqueeze(0).to(device)

    # Predict keypoint
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()[0]

    # Plot
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    h, w = image.size
    plt.scatter(pred[0]*w, pred[1]*h, c='red', label='Predicted')
    plt.scatter(true_x*w, true_y*h, c='blue', label='Ground Truth')
    plt.title(f"Class: {class_label}")
    plt.legend()
    plt.axis('off')
    plt.show()


# ---------------- Example ---------------- #
model = get_model().to(device)
model.load_state_dict(torch.load("best_resnet_keypoint.pth"))

# Prepare class-to-index map
class_to_idx = {cls: idx for idx, cls in enumerate(sorted(df['class'].unique()))}

# Pick a sample pedestrian
sample = df[(df['split'] == 'val') & (df['class'] == 'truck')].sample(1).iloc[0]
visualize_prediction(
    sample['image_path'],
    sample['class'],
    sample['x'],
    sample['y'],
    model,
    class_to_idx
)
