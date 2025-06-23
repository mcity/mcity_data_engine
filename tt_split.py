import os
import random
import shutil
from glob import glob
from uuid import uuid4

# Parameters
BASE_DIR = "split_by_class_80_20"
OUT_DIR = "resampled_balanced_split"
TRAIN_TARGET = 500
VAL_TARGET = 125

TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "train", "images")
CLASSES = sorted([
    d for d in os.listdir(TRAIN_IMAGE_DIR)
    if os.path.isdir(os.path.join(TRAIN_IMAGE_DIR, d))
])

def sample_and_copy(split, target_count):
    for cls in CLASSES:
        img_src_dir = os.path.join(BASE_DIR, split, "images", cls)
        msk_src_dir = os.path.join(BASE_DIR, split, "masks", cls)

        out_img_dir = os.path.join(OUT_DIR, split, "images", cls)
        out_msk_dir = os.path.join(OUT_DIR, split, "masks", cls)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_msk_dir, exist_ok=True)

        image_paths = sorted(glob(os.path.join(img_src_dir, "*.jpg")))
        mask_paths = sorted(glob(os.path.join(msk_src_dir, "*.png")))
        paired = list(zip(image_paths, mask_paths))

        if len(paired) == 0:
            print(f"⚠️ No data for class {cls} in split {split}, skipping...")
            continue

        # Sample with or without replacement
        if len(paired) >= target_count:
            sampled = random.sample(paired, target_count)
        else:
            sampled = random.choices(paired, k=target_count)

        for i, (img_path, msk_path) in enumerate(sampled):
            suffix = uuid4().hex[:6]
            img_name = f"{cls}_{i}_{suffix}.jpg"
            msk_name = f"{cls}_{i}_{suffix}.png"

            shutil.copy(img_path, os.path.join(out_img_dir, img_name))
            shutil.copy(msk_path, os.path.join(out_msk_dir, msk_name))

        print(f"✅ {cls}: {len(sampled)} images in {split}")

# Run for both splits
sample_and_copy("train", TRAIN_TARGET)
sample_and_copy("val", VAL_TARGET)
