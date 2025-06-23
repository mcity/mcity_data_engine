import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

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

def safe_class_name(cls):
    return cls.replace("/", "_").replace("\\", "_").strip()

def crop_and_save(img_path, box, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    xtl, ytl, xbr, ybr = map(int, box)
    xtl, ytl = max(0, xtl), max(0, ytl)
    xbr, ybr = min(w, xbr), min(h, ybr)
    cropped = img[ytl:ybr, xtl:xbr]
    if cropped.size > 0:
        cv2.imwrite(out_path, cropped)

def preprocess_dataset(full_xml, contact_xml, image_dir, output_dir_f1, output_dir_f2):
    full_boxes = parse_boxes(full_xml)
    contact_boxes = parse_boxes(contact_xml)

    os.makedirs(output_dir_f1, exist_ok=True)
    os.makedirs(output_dir_f2, exist_ok=True)

    matched = 0

    for image_name in tqdm(full_boxes):
        if image_name not in contact_boxes:
            continue

        full_list = full_boxes[image_name]
        contact_list = contact_boxes[image_name]

        for f_cls, f_box in full_list:
            for c_cls, c_box in contact_list:
                if f_cls != c_cls:
                    continue
                if not iou(f_box, c_box):
                    continue

                matched += 1
                # Prepare directories
                f_cls_safe = safe_class_name(f_cls)
                out_dir_f1 = os.path.join(output_dir_f1, f_cls_safe)
                out_dir_f2 = os.path.join(output_dir_f2, f_cls_safe)
                os.makedirs(out_dir_f1, exist_ok=True)
                os.makedirs(out_dir_f2, exist_ok=True)

                base = os.path.splitext(os.path.basename(image_name))[0]
                f_out_path = os.path.join(out_dir_f1, f"{base}_{matched}_full.jpg")
                c_out_path = os.path.join(out_dir_f2, f"{base}_{matched}_part.jpg")

                crop_and_save(os.path.join(image_dir, image_name), f_box, f_out_path)
                crop_and_save(os.path.join(image_dir, image_name), intersection(f_box, c_box), c_out_path)
                break

    print(f"✅ Done. Total matched pairs: {matched}")

# Set paths here
full_xml_path = "/home/dataengine/Downloads/entire_box/annotations.xml"
contact_xml_path = "/home/dataengine/Downloads/contact_box/annotations.xml"
image_directory = "/home/dataengine/Downloads/raw_sip_batch3_yolo (1)/raw_sip_batch3_yolo/images/train"
output_f1 = "f1"
output_f2 = "f2"

preprocess_dataset(full_xml_path, contact_xml_path, image_directory, output_f1, output_f2)
