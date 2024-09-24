# Expected result: metadata.jsonl
# bboxes need to be in COCO format [x, y, width, height]
# {"file_name": "0001.png", "objects": {"bbox": [[302.0, 109.0, 73.0, 52.0]], "categories": [0]}}
# {"file_name": "0002.png", "objects": {"bbox": [[810.0, 100.0, 57.0, 28.0]], "categories": [1]}}
# {"file_name": "0003.png", "objects": {"bbox": [[160.0, 31.0, 248.0, 616.0], [741.0, 68.0, 202.0, 401.0]], "categories": [2, 2]}}

#  Yolov5 format
#  0 0.176372 0.359001 0.00411315 0.0149609
#  0 0.164114 0.373934 0.00330154 0.00769851
#  6 0.252547 0.403505 0.0078125 0.0104167

# https://huggingface.co/docs/datasets/v2.7.1/en/image_dataset


import os
import shutil
from pathlib import Path
import json


def yolo_to_jsonl(yolo_lines, image_width, image_height, file_name):
    bboxes = []
    categories = []

    for yolo_line in yolo_lines:
        # Split the YOLO line into components
        parts = yolo_line.strip().split()
        class_id = int(parts[0])
        center_x = float(parts[1]) * image_width
        center_y = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height

        # Convert center coordinates to top-left coordinates
        top_left_x = center_x - (width / 2)
        top_left_y = center_y - (height / 2)

        # Round the coordinates and dimensions
        top_left_x = round(top_left_x)
        top_left_y = round(top_left_y)
        width = round(width)
        height = round(height)

        # Append to lists
        bboxes.append([top_left_x, top_left_y, width, height])
        categories.append(class_id)

    # Create the JSON object
    json_obj = {
        "file_name": file_name,
        "objects": {
            "bbox": bboxes,
            "categories": categories,
        },
    }

    return json.dumps(json_obj)


def main():
    source_folder = "/home/dbogdoll/datasets/midadvrb_2000/"
    target_folder = "/home/dbogdoll/datasets/midadvrb_2000_hf/"

    IMG_WIDTH = 1280
    IMG_HEIGHT = 960

    # Iterate over files in directory
    splits = ["train", "val"]
    for split in splits:
        jsonl_labels = []
        Path(target_folder + split).mkdir(parents=True, exist_ok=True)
        img_directory = source_folder + "images/" + split + "/"
        for img_name in os.listdir(img_directory):
            # Open file
            with open(os.path.join(img_directory, img_name)) as f:
                print(f)

                # Copy images
                img_path = os.path.join(img_directory, img_name)
                img_target_directory = target_folder + split + "/" + img_name
                if not os.path.exists(img_target_directory):
                    shutil.copy2(img_path, img_target_directory)

                # Convert labels
                label_name = img_name.replace(".jpg", ".txt")
                label_path = source_folder + "labels/" + split + "/" + label_name
                if os.path.exists(label_path):
                    with open(label_path, "r") as yolo_labels:
                        yolo_lines = yolo_labels.readlines()
                        jsonl_label = yolo_to_jsonl(
                            yolo_lines, IMG_WIDTH, IMG_HEIGHT, img_name
                        )
                        jsonl_labels.append(jsonl_label)

        # Save JSONL labels to file
        with open(target_folder + split + "/metadata.jsonl", "w") as jsonl_file:
            for jsonl_label in jsonl_labels:
                jsonl_file.write(jsonl_label + "\n")


if __name__ == "__main__":
    main()
