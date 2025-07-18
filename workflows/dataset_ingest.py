import os
import fiftyone as fo
import fiftyone.types as fot
import random
import shutil
import fiftyone.utils.yolo as fouy
import fiftyone.utils.video as fouv
from config.config import WORKFLOWS


def detect_format(dataset_dir):
    files = os.listdir(dataset_dir)
    files_lower = [f.lower() for f in files]

    # Flatten all files (including nested) up to 2 levels
    all_files = []
    for root, dirs, fs in os.walk(dataset_dir):
        for f in fs:
            all_files.append(os.path.join(root, f).lower())

    # Format detection
    if any("annotations.xml" in f for f in all_files):
        return "cvat"
    elif any(f.endswith(".json") and ("instances" in f or "coco" in f) for f in all_files):
        return "coco"
    elif any(f.endswith(".xml") for f in all_files):
        return "voc"
    elif any("labels" in f for f in all_files) or any(f.endswith(".txt") for f in all_files):
        return "yolo"
    elif any(f.endswith((".mp4", ".avi", ".mov")) for f in all_files):
        return "video"
    elif any(f.endswith((".jpg", ".jpeg", ".png")) for f in all_files):
        return "image_only"
    else:
        raise ValueError("Unable to auto-detect dataset format.")


def get_dataset_type(fmt):
    """Map format string to FiftyOne dataset type or string indicators."""
    if fmt == "coco":
        return fot.COCODetectionDataset
    elif fmt == "voc":
        return fot.VOCDetectionDataset
    elif fmt == "yolo":
        return fot.YOLOv5Dataset
    elif fmt == "video":
        return "video"
    elif fmt == "image_only":
        return "image_only"
    elif fmt == "cvat":
        return fot.CVATImageDataset
    else:
        raise ValueError(f"Unsupported annotation_format: {fmt}")



def run_dataset_ingest():
    config = WORKFLOWS["dataset_ingest"]
    dataset_name = config["dataset_name"]
    dataset_dir = config["dataset_dir"]
    split = config.get("split_percentages", [0.7, 0.15, 0.15])
    fmt = config["annotation_format"]

    if fmt == "auto":
        fmt = detect_format(dataset_dir)

    dataset_type = get_dataset_type(fmt)

    print(f"Ingesting dataset: {dataset_name}")
    print(f"Detected format: {fmt}")
    print(f"Loading from: {dataset_dir}")

    if dataset_type == "video":
        print(f"🎞️ Converting videos in {dataset_dir} to frames at 5 FPS...")

        # Create temp dataset from videos
        video_dataset = fo.Dataset.from_videos_dir(dataset_dir, name=f"{dataset_name}_video_temp")

        # Define frame sampling output directory
        frames_dir = os.path.join(dataset_dir, "extracted_frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Sample at 5 FPS using FFmpeg
        fouv.sample_videos(
            video_dataset,
            fps=5,
            output_dir=frames_dir,
            original_frame_numbers=False,
            force_sample=True,
            verbose=False,
            progress=True,
        )

        print(f"📸 Sampled frames stored at {frames_dir}")

        # Now load as image-only dataset
        dataset = fo.Dataset.from_images_dir(frames_dir, name=dataset_name)

    elif dataset_type == "image_only":
        dataset = fo.Dataset.from_images_dir(dataset_dir, name = dataset_name)
    elif fmt == "coco":
        # ✅ Fast path: Single JSON + single image folder (any name)
        flat_jsons = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]
        image_dirs = [
            os.path.join(dataset_dir, d)
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d)) and d.lower() != "annotations"
        ]

        if len(flat_jsons) == 1 and len(image_dirs) == 1:
            image_dir = image_dirs[0]
            print(f"ℹ️ Using {image_dir} for all splits (defaulted to 'train')")

            dataset = fo.Dataset.from_dir(
                dataset_type=fot.COCODetectionDataset,
                data_path=image_dir,
                labels_path=os.path.join(dataset_dir, flat_jsons[0]),
                name=dataset_name,
            )

            dataset.tag_samples("train")
            print(f"✅ Loaded {len(dataset)} samples from flat COCO directory")
            dataset.persistent = True

        else:
            annotations_dir = os.path.join(dataset_dir, "annotations")
            json_search_dir = annotations_dir if os.path.exists(annotations_dir) else dataset_dir

            # Find all COCO-style annotation files
            possible_jsons = [
                f for f in os.listdir(json_search_dir)
                if f.endswith(".json") or ("instances" in f or "annotation" in f)
            ]

            if not possible_jsons:
                raise ValueError(f"No COCO-style annotation JSON files found in '{json_search_dir}'")

            # Find available image directories with common split names
            available_dirs = {
                split: os.path.join(dataset_dir, d)
                for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
                for split in ["train", "val", "test"]
                if split in d.lower()
            }

            dataset = fo.Dataset(name=dataset_name)

            for json_file in possible_jsons:
                json_lower = json_file.lower()
                if "train" in json_lower:
                    split_name = "train"
                elif "val" in json_lower:
                    split_name = "val"
                elif "test" in json_lower:
                    split_name = "test"
                else:
                    split_name = "unlabeled"

                # Match to image folder
                data_path = available_dirs.get(split_name)
                if not data_path:
                    print(f"⚠️ Skipping '{split_name}' — No image folder found for '{json_file}'")
                    continue

                dataset_split = fo.Dataset.from_dir(
                    dataset_type=fot.COCODetectionDataset,
                    data_path=data_path,
                    labels_path=os.path.join(json_search_dir, json_file),
                    name=f"{dataset_name}_{split_name}",
                )

                sample_ids = dataset.add_samples(dataset_split)
                dataset.select(sample_ids).tag_samples(split_name)
                print(f"✅ Loaded {len(sample_ids)} samples for split '{split_name}'")

        print("Final split counts:", dataset.count_sample_tags())

    elif fmt == "cvat":
        dataset = fo.Dataset.from_dir(
            dataset_type=fot.CVATImageDataset,
            data_path=os.path.join(dataset_dir, "data"),
            labels_path=os.path.join(dataset_dir, "annotations.xml"),
            name=dataset_name,
        )
    else:
        if fmt == "yolo":
            # Check for dataset.yaml
            yaml_path = os.path.join(dataset_dir, "dataset.yaml")
            if os.path.exists(yaml_path):
                dataset = fo.Dataset(name=dataset_name)
                split_names = ["train", "val", "test"]
                loaded_splits = {}

                with open(yaml_path) as f:
                    yaml_content = f.read()

                for split_name in split_names:
                    if f"{split_name}:" not in yaml_content:
                        print(f"Skipping split '{split_name}' — not found in dataset.yaml")
                        continue

                    try:
                        importer = fouy.YOLOv5DatasetImporter(
                            dataset_dir=dataset_dir,
                            yaml_path=yaml_path,
                            split=split_name,
                        )

                        sample_ids = dataset.add_importer(importer, label_field="ground_truth")

                        if not sample_ids:
                            print(f"No samples loaded for split '{split_name}'")
                            continue

                        dataset.select(sample_ids).tag_samples(split_name)
                        loaded_splits[split_name] = sample_ids

                        print(f"Loaded {len(sample_ids)} samples for split '{split_name}'")
                    except Exception as e:
                        print(f"Warning: Failed to load split '{split_name}': {e}")



                # Fallback logic if val or test is missing
                if "val" not in loaded_splits and "test" in loaded_splits:
                    test_samples = loaded_splits["test"]
                    midpoint = len(test_samples) // 2

                    dataset.select(test_samples).untag_samples("test")

                    dataset.select(test_samples[:midpoint]).tag_samples("val")
                    dataset.select(test_samples[midpoint:]).tag_samples("test")
                    print("Split test → val/test 50-50")

                elif "test" not in loaded_splits and "val" in loaded_splits:
                    val_samples = loaded_splits["val"]
                    midpoint = len(val_samples) // 2

                    dataset.select(val_samples).untag_samples("val")

                    dataset.select(val_samples[:midpoint]).tag_samples("val")
                    dataset.select(val_samples[midpoint:]).tag_samples("test")
                    print("Split val → val/test 50-50")

                elif "val" not in loaded_splits and "test" not in loaded_splits and "train" in loaded_splits:
                    all_train_samples = loaded_splits["train"]
                    dataset.select(all_train_samples).untag_samples("train")# clear tag first
                    random.seed(51)
                    random.shuffle(all_train_samples)

                    n = len(all_train_samples)
                    s_train, s_val, s_test = split
                    n_train = int(s_train * n)
                    n_val = int(s_val * n)

                    # Get sample IDs
                    dataset.select(all_train_samples[:n_train]).tag_samples("train")
                    dataset.select(all_train_samples[ n_train : n_train + n_val]).tag_samples("val")
                    dataset.select(all_train_samples[ n_train + n_val:]).tag_samples("test")

                    print(f"Split train → train/val/test with {split} proportion")

                print("📊 Final split counts:", dataset.count_sample_tags())

        else:
            dataset = fo.Dataset.from_dir(
                dataset_dir=dataset_dir,
                dataset_type=dataset_type,
                name=dataset_name,
            )


    dataset.persistent = True

    if fmt == "yolo" and os.path.exists(os.path.join(dataset_dir, "dataset.yaml")):
        print("Detected YOLO dataset with splits defined in dataset.yaml — skipping split and shuffle.")
    else:
        # Clear all existing split tags before reassigning
        for tag in ["train", "val", "test"]:
            dataset.match_tags(tag).untag_samples(tag)

        dataset.shuffle(seed=51)

        n = len(dataset)
        n_train = int(split[0] * n)
        n_val = int(split[1] * n)
        n_test = n - n_train - n_val

        # Apply tag-based split
        dataset[:n_train].tag_samples("train")
        dataset[n_train:n_train + n_val].tag_samples("val")
        dataset[n_train + n_val:].tag_samples("test")

        print(f"✅ Split applied: train {n_train}, val {n_val}, test {n_test}")
    print(f"✅ Dataset '{dataset_name}' ingested with {len(dataset)} samples")