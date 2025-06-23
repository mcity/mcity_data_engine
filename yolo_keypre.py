import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import pandas as pd

# Define the root path
root_dir = Path("resampled_balanced_split")
splits = ["train", "val"]
classes = ["bus", "car", "motorbike", "pedestrian", "pickup", "trailer", "truck", "van"]
class_to_id = {cls: i for i, cls in enumerate(classes)}

# Output structure
output_root = Path("yolo_pose_from_mask")
for split in splits:
    (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

def extract_keypoint_from_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    cx = np.mean(xs) / 224
    cy = np.mean(ys) / 224
    return cx, cy

for split in splits:
    for cls in classes:
        image_dir = root_dir / split / "images" / cls
        mask_dir = root_dir / split / "masks" / cls
        out_image_dir = output_root / "images" / split
        out_label_dir = output_root / "labels" / split

        for image_file in tqdm(list(image_dir.glob("*.jpg")), desc=f"{split}/{cls}"):
            mask_file = mask_dir / (image_file.stem + ".png")
            if not mask_file.exists():
                continue

            kp = extract_keypoint_from_mask(mask_file)
            if kp is None:
                continue

            # Copy image
            out_img_path = out_image_dir / f"{cls}_{image_file.name}"
            out_label_path = out_label_dir / f"{cls}_{image_file.stem}.txt"
            cv2.imwrite(str(out_img_path), cv2.imread(str(image_file)))

            # YOLO format: cls_id x y w h kpx kpy visibility
            label = f"{class_to_id[cls]} 0.5 0.5 1.0 1.0 {kp[0]:.6f} {kp[1]:.6f}\n"
            with open(out_label_path, "w") as f:
                f.write(label)

# Create YAML file
yaml_content = {
    "path": str(output_root.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": len(classes),
    "names": classes,
    "kpt_shape": [1, 2]  # 1 keypoint with (x, y, visibility)
}
with open(output_root / "pose_data.yaml", "w") as f:
    yaml.dump(yaml_content, f)
