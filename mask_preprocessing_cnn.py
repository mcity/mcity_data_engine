import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Configuration
input_root = "f1"
target_root = "f2"
output_root = "cnn_heatmap_dataset"
output_input_dir = os.path.join(output_root, "inputs")
output_mask_dir = os.path.join(output_root, "masks")
# Image size to which full images will be resized
target_size = (224, 224)

# Create output directories
os.makedirs(output_input_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Loop over all class folders in f1
all_classes = os.listdir(input_root)

for class_name in tqdm(all_classes, desc="Processing classes"):
    input_class_dir = os.path.join(input_root, class_name)
    target_class_dir = os.path.join(target_root, class_name)

    for file in os.listdir(input_class_dir):
        if not file.endswith("_full.jpg"):
            continue

        base_name = file.replace("_full.jpg", "")
        input_path = os.path.join(input_class_dir, file)
        target_path = os.path.join(target_class_dir, f"{base_name}_part.jpg")

        if not os.path.exists(target_path):
            print(f"❌ Missing part image for {file}")
            continue

        try:
            # Load full and part images
            full_img_raw = Image.open(input_path).convert("RGB")
            part_img_raw = Image.open(target_path).convert("RGB")

            orig_full_w, orig_full_h = full_img_raw.size
            part_w, part_h = part_img_raw.size

            # Resize full image to target size
            full_img_resized = full_img_raw.resize(target_size)
            full_np = np.array(full_img_resized)

            # Compute scale factor and resize part image proportionally
            scale_x = target_size[0] / orig_full_w
            scale_y = target_size[1] / orig_full_h
            new_part_size = (int(part_w * scale_x), int(part_h * scale_y))

            if new_part_size[0] < 1 or new_part_size[1] < 1:
                print(f"⚠️ Very small part image for {file}, upscaling minimally.")
                new_part_size = (1, 1)

            part_img_resized = part_img_raw.resize(new_part_size)
            part_np = np.array(part_img_resized)

            # Template match (grayscale)
            full_gray = cv2.cvtColor(full_np, cv2.COLOR_RGB2GRAY)
            part_gray = cv2.cvtColor(part_np, cv2.COLOR_RGB2GRAY)
            match = cv2.matchTemplate(full_gray, part_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

            top_left = max_loc
            bottom_right = (top_left[0] + new_part_size[0], top_left[1] + new_part_size[1])

            # Generate binary mask
            mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
            cv2.rectangle(mask, top_left, bottom_right, color=255, thickness=-1)

            # Save files
            input_save_path = os.path.join(output_input_dir, f"{class_name}_{base_name}.jpg")
            mask_save_path = os.path.join(output_mask_dir, f"{class_name}_{base_name}.png")
            full_img_resized.save(input_save_path)
            cv2.imwrite(mask_save_path, mask)

        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")
