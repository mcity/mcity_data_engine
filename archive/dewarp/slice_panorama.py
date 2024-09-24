# Required image input is tensor: float32[batch_size,3,544,960]
# As defined by Nvidia TAO grounding_dino/specs/gen_trt_engine.yaml

import cv2
import numpy as np
from tqdm import tqdm
import os

from dewarp import read_labels
from dewarp import yolo_labels_to_points
from dewarp import points_to_yolo_labels
from dewarp import points_to_yolo_string


def main():
    splits = "train", "val"
    for split in splits:
        image_folder_source = (
            "/home/dbogdoll/datasets/midadvrb_2000_dewarp/images/" + split + "/"
        )
        image_folder_target = (
            "/home/dbogdoll/datasets/midadvrb_2000_dewarp_slices/images/" + split + "/"
        )

        labels_folder_source = (
            "/home/dbogdoll/datasets/midadvrb_2000_dewarp/labels/" + split + "/"
        )
        labels_folder_target = (
            "/home/dbogdoll/datasets/midadvrb_2000_dewarp_slices/labels/" + split + "/"
        )

        for filename_img in tqdm(os.listdir(image_folder_source)):
            filename_label = filename_img.replace(".jpg", ".txt")
            image_source = image_folder_source + filename_img
            label_source = labels_folder_source + filename_label

            image_target = image_folder_target + filename_img
            label_target = labels_folder_target + filename_label

            img = cv2.imread(image_source)
            img_height, img_width, img_channels = img.shape

            # Split image intwo two
            half = img_width // 2
            img_left = img[:, :half]
            img_right = img[:, half:]

            # Create new filenames for left and right slices
            base_name, ext = filename_img.rsplit(".", 1)
            filename_left = f"{base_name}_left.{ext}"
            filename_right = f"{base_name}_right.{ext}"
            cv2.imwrite(image_folder_target + filename_left, img_left)
            cv2.imwrite(image_folder_target + filename_right, img_right)

            # Divide labels into left or right
            labels = read_labels(label_source)

            left_min_x = 0
            left_max_x = half - 1
            right_min_x = half
            right_max_x = img_width - 1

            labels_left = []
            labels_right = []

            for classid, x_center, y_center, width_norm, height_norm in labels:
                top_left, top_right, bottom_left, bottom_right = yolo_labels_to_points(
                    x_center, y_center, width_norm, height_norm, img_width, img_height
                )

                # Check if label is fully in the left image
                if top_left[0] >= left_min_x and top_right[0] <= left_max_x:
                    labels_left.append(
                        points_to_yolo_string(
                            classid,
                            top_left,
                            top_right,
                            bottom_left,
                            bottom_right,
                            half,
                            img_height,
                        )
                    )

                elif top_left[0] >= right_min_x and top_right[0] <= right_max_x:
                    labels_right.append(
                        points_to_yolo_string(
                            classid,
                            np.subtract(top_left, (half, 0)),
                            np.subtract(top_right, (half, 0)),
                            np.subtract(bottom_left, (half, 0)),
                            np.subtract(bottom_right, (half, 0)),
                            half,
                            img_height,
                        )
                    )

                else:
                    left_top_right = (left_max_x, top_right[1])
                    left_bottom_right = (left_max_x, bottom_right[1])
                    labels_left.append(
                        points_to_yolo_string(
                            classid,
                            top_left,
                            left_top_right,
                            bottom_left,
                            left_bottom_right,
                            half,
                            img_height,
                        )
                    )

                    right_top_left = (right_min_x, top_left[1])
                    right_bottom_left = (right_min_x, bottom_left[1])
                    labels_right.append(
                        points_to_yolo_string(
                            classid,
                            np.subtract(right_top_left, (half, 0)),
                            np.subtract(top_right, (half, 0)),
                            np.subtract(right_bottom_left, (half, 0)),
                            np.subtract(bottom_right, (half, 0)),
                            half,
                            img_height,
                        )
                    )

            target = labels_folder_target + base_name + "_left.txt"
            with open(target, "w") as file:
                for line in labels_left:
                    file.write(line + "\n")

            target = labels_folder_target + base_name + "_right.txt"
            with open(target, "w") as file:
                for line in labels_right:
                    file.write(line + "\n")


if __name__ == "__main__":
    main()
