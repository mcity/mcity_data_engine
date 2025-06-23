import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import Counter
CLASS_NAMES = ["bus", "car", "motorbike/cycler", "pedestrian", "pickup", "trailer", "truck", "van"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def parse_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes_by_image = {}
    for image in root.findall("image"):
        image_name = image.attrib["name"]
        boxes = []
        for box in image.findall("box"):
            label = box.attrib["label"]
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])
            boxes.append((label, (xtl, ytl, xbr, ybr)))
        boxes_by_image[image_name] = boxes
    return boxes_by_image

def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    return inter > 0

def intersection(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    return (xi1, yi1, xi2, yi2)

def center_of_box(box):
    xtl, ytl, xbr, ybr = box
    return (xtl + xbr) / 2, (ytl + ybr) / 2

def preprocess_for_yolov8_pose(full_xml, contact_xml, image_dir, output_image_dir, output_label_dir):
    full_boxes = parse_boxes(full_xml)
    contact_boxes = parse_boxes(contact_xml)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    matched = 0
    class_counter = Counter()  # <-- NEW

    for image_name in tqdm(full_boxes):
        if image_name not in contact_boxes:
            continue

        full_list = full_boxes[image_name]
        contact_list = contact_boxes[image_name]
        image_path = os.path.join(image_dir, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        new_image_name = os.path.splitext(image_name)[0] + ".jpg"
        cv2.imwrite(os.path.join(output_image_dir, new_image_name), img)
        label_lines = []

        for f_cls, f_box in full_list:
            for c_cls, c_box in contact_list:
                if f_cls != c_cls:
                    continue
                if not iou(f_box, c_box):
                    continue

                matched += 1

                cls_id = CLASS_TO_ID.get(f_cls, -1)
                if cls_id == -1:
                    print(f"⚠️ Unknown class '{f_cls}' in image {image_name}")
                    continue
                class_counter[f_cls] += 1

                xtl, ytl, xbr, ybr = f_box

                # Normalize full box
                cx = (xtl + xbr) / 2 / w
                cy = (ytl + ybr) / 2 / h
                bw = (xbr - xtl) / w
                bh = (ybr - ytl) / h

                # Use intersection box for keypoint
                intersection_box = intersection(f_box, c_box)
                kp_x, kp_y = center_of_box(intersection_box)
                kp_x /= w
                kp_y /= h
                visibility = 2

                line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kp_x:.6f} {kp_y:.6f} {visibility}"
                label_lines.append(line)
                break

        if label_lines:
            label_path = os.path.join(output_label_dir, os.path.splitext(image_name)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

    print(f"✅ YOLOv8-Pose dataset created with intersection logic. Total matched pairs: {matched}")
    print("\n📊 Class Distribution:")
    for cls, count in sorted(class_counter.items()):
        print(f"  {cls:15} → {count} samples")

# Paths setup (update if needed)
full_xml_path = "/home/dataengine/Downloads/entire_box/annotations.xml"
contact_xml_path = "/home/dataengine/Downloads/contact_box/annotations.xml"
image_directory = "/home/dataengine/Downloads/raw_sip_batch3_yolo (1)/raw_sip_batch3_yolo/images/train"
output_images = "yolo_pose/images/train"
output_labels = "yolo_pose/labels/train"

preprocess_for_yolov8_pose(full_xml_path, contact_xml_path, image_directory, output_images, output_labels)
