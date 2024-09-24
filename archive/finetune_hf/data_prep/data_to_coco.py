import fiftyone as fo
import os
from pathlib import Path
import shutil

DATA_FOLDER = "/home/dbogdoll/datasets/"
OUTPUT_Folder = "/home/dbogdoll/datasets/coco_experiments/"

NAME_COMPLETE = "midadvrb_2000_coco"
NAME_TRAIN = "midadvrb_train"
NAME_VAL = "midadvrb_val"


def convert_yolo_to_coco():

    dataset_dir = os.path.join(DATA_FOLDER, "midadvrb_2000")

    try:
        fo.delete_dataset(NAME_COMPLETE)
        fo.delete_dataset(NAME_TRAIN)
        fo.delete_dataset(NAME_VAL)
    except:
        print("No prior dataset active")

    # Convert and export train and val
    dataset_train = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        name=NAME_TRAIN,
        split="train",
    )

    export_dir_train = os.path.join(OUTPUT_Folder, NAME_TRAIN)
    dataset_train.export(
        export_dir=export_dir_train,
        dataset_type=fo.types.COCODetectionDataset,
    )

    dataset_val = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        name=NAME_VAL,
        split="val",
    )

    export_dir_val = os.path.join(OUTPUT_Folder, NAME_VAL)
    dataset_val.export(
        export_dir=export_dir_val,
        dataset_type=fo.types.COCODetectionDataset,
    )

    return export_dir_train, export_dir_val


def merge_coco(export_dir_train, export_dir_val):
    # Merge exportted separate train and val folders
    # All images into dir/images, and separate annotation files in annotations/train_labels.json and val_labels.json
    MERGE_FOLDER = os.path.join(OUTPUT_Folder, NAME_COMPLETE)
    Path(os.path.join(MERGE_FOLDER, "data")).mkdir(parents=True, exist_ok=True)

    # Move images
    for file in os.listdir(os.path.join(export_dir_train, "data")):
        file_path = os.path.join(export_dir_train, "data", file)
        shutil.move(file_path, os.path.join(MERGE_FOLDER, "data", file))

    for file in os.listdir(os.path.join(export_dir_val, "data")):
        file_path = os.path.join(export_dir_val, "data", file)
        shutil.move(file_path, os.path.join(MERGE_FOLDER, "data", file))

    # Move labels
    shutil.move(
        os.path.join(OUTPUT_Folder, NAME_TRAIN, "labels.json"),
        os.path.join(MERGE_FOLDER, "train_labels.json"),
    )

    shutil.move(
        os.path.join(OUTPUT_Folder, NAME_VAL, "labels.json"),
        os.path.join(MERGE_FOLDER, "val_labels.json"),
    )

    # Delete old folders
    try:
        shutil.rmtree(os.path.join(OUTPUT_Folder, NAME_TRAIN))
        shutil.rmtree(os.path.join(OUTPUT_Folder, NAME_VAL))
    except OSError as ex:
        print(ex)


def upload_coco():
    # Load COCO dataset and upload to HuggingFace
    splits = ["train", "val"]
    dataset = fo.Dataset(NAME_COMPLETE)
    dataset_dir = os.path.join(OUTPUT_Folder, NAME_COMPLETE)
    for split in splits:
        dataset = dataset.add_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.COCODetectionDataset,
            split=split,
            tags=split,
        )

    print(dataset)


def main():
    export_dir_train, export_dir_val = convert_yolo_to_coco()
    merge_coco(export_dir_train, export_dir_val)
    upload_coco()


if __name__ == "__main__":
    main()
