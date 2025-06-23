import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

def extract_keypoint_from_mask(mask):
    """Find the center of the white region (ROI) in the binary mask."""
    mask_array = np.array(mask)
    ys, xs = np.where(mask_array > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None  # No ROI found
    x_center = np.mean(xs) / mask_array.shape[1]
    y_center = np.mean(ys) / mask_array.shape[0]
    return x_center, y_center

def prepare_keypoint_dataset(root_dir):
    data = []

    for split in ['train', 'val']:
        split_image_root = os.path.join(root_dir, split, 'images')
        split_mask_root = os.path.join(root_dir, split, 'masks')

        for cls in os.listdir(split_image_root):
            image_dir = os.path.join(split_image_root, cls)
            mask_dir = os.path.join(split_mask_root, cls)

            for img_file in os.listdir(image_dir):
                if not img_file.endswith('.jpg'):
                    continue
                img_path = os.path.join(image_dir, img_file)
                mask_path = os.path.join(mask_dir, img_file.replace('.jpg', '.png'))

                if not os.path.exists(mask_path):
                    continue

                try:
                    mask = Image.open(mask_path).convert('L')
                    keypoint = extract_keypoint_from_mask(mask)
                    if keypoint is None:
                        continue

                    data.append({
                        'split': split,
                        'class': cls,
                        'image_path': img_path,
                        'x': keypoint[0],
                        'y': keypoint[1]
                    })
                except Exception as e:
                    continue

    return pd.DataFrame(data)

df_keypoints = prepare_keypoint_dataset("resampled_balanced_split")
df_keypoints.to_csv("resnet_keypoints.csv", index=False)
print("✅ Saved dataset with", len(df_keypoints), "entries to resnet_keypoints.csv")