import os
from tqdm import tqdm
import cv2
from fisheyewarping import FisheyeWarping
import numpy as np

import matplotlib.pyplot as plt


DEBUG = False
MANUAL_REVIEW = False

# -----------------------FUNCTIONS-----------------------------------


def yolo_labels_to_points(x_center, y_center, width_norm, height_norm, width, height):
    x_center_pixel = x_center * width
    y_center_pixel = y_center * height
    width_pixel = width_norm * width
    height_pixel = height_norm * height

    # Calculate top-left and bottom-right coordinates
    top_left = (
        int(x_center_pixel - width_pixel / 2),
        int(y_center_pixel - height_pixel / 2),
    )
    top_right = (
        int(x_center_pixel + width_pixel / 2),
        int(y_center_pixel - height_pixel / 2),
    )
    bottom_left = (
        int(x_center_pixel - width_pixel / 2),
        int(y_center_pixel + height_pixel / 2),
    )
    bottom_right = (
        int(x_center_pixel + width_pixel / 2),
        int(y_center_pixel + height_pixel / 2),
    )

    return (top_left, top_right, bottom_left, bottom_right)


def points_to_yolo_labels(
    top_left, top_right, bottom_left, bottom_right, width, height
):
    # Extract x and y coordinates
    x_coords = [top_left[0], top_right[0], bottom_left[0], bottom_right[0]]
    y_coords = [top_left[1], top_right[1], bottom_left[1], bottom_right[1]]

    # Compute center coordinates
    x_center_pixel = (min(x_coords) + max(x_coords)) / 2
    y_center_pixel = (min(y_coords) + max(y_coords)) / 2

    # Compute width and height in pixels
    width_pixel = max(x_coords) - min(x_coords)
    height_pixel = max(y_coords) - min(y_coords)

    # Normalize width and height
    width_norm = width_pixel / width
    height_norm = height_pixel / height

    # Normalize center coordinates
    x_center = x_center_pixel / width
    y_center = y_center_pixel / height

    return x_center, y_center, width_norm, height_norm


def points_to_yolo_string(
    classid, top_left, top_right, bottom_left, bottom_right, img_width, img_height
):

    x_center, y_center, width_norm, height_norm = points_to_yolo_labels(
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        img_width,
        img_height,
    )

    line = (
        str(classid)
        + " "
        + str(x_center)
        + " "
        + str(y_center)
        + " "
        + str(width_norm)
        + " "
        + str(height_norm)
    )

    return line


def read_labels(source):
    labels = []
    with open(source, "r") as file:
        for line in file:
            parts = line.strip().split()
            classid = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])
            labels.append((classid, x_center, y_center, width_norm, height_norm))

    return labels


def dewarp_fisheye(source, target, labels_debug):

    # Read file
    fisheye_img = cv2.imread(source)
    height, width, _ = fisheye_img.shape

    # Dewarp
    frd = FisheyeWarping(fisheye_img, use_multiprocessing=True)
    # frd.build_dewarp_mesh(save_path="./dewarp-mesh.pkl")
    panorama_shape, _, _ = frd.load_dewarp_mesh("./dewarp-mesh.pkl")

    if DEBUG:  # Add labels onto image prior to warp to assess quality of warped labels
        color = (0, 255, 0)
        thickness = 5

        for classid, x_center, y_center, width_norm, height_norm in labels_debug:
            top_left, top_right, bottom_left, bottom_right = yolo_labels_to_points(
                x_center, y_center, width_norm, height_norm, width, height
            )
            cv2.rectangle(fisheye_img, top_left, bottom_right, color, thickness)

    frd.run_dewarp(target)

    return fisheye_img, frd, height, width, panorama_shape


def compute_divergence(original_coords, new_coords):
    """
    Compute the divergence of the warping vectors between original and new coordinates.

    Parameters:
    original_coords (list of tuples): List of original coordinates [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    new_coords (list of tuples): List of new coordinates [(x1', y1'), (x2', y2'), (x3', y3'), (x4', y4')]

    Returns:
    float: The divergence of the warping vectors.
    """
    # Compute the vectors
    vectors = [
        (new[0] - orig[0], new[1] - orig[1])
        for orig, new in zip(original_coords, new_coords)
    ]

    # Convert to numpy arrays for easier manipulation
    vectors = np.array(vectors)
    original_coords = np.array(original_coords)

    # Compute finite differences for partial derivatives
    dx = np.gradient(vectors[:, 0], original_coords[:, 0])
    dy = np.gradient(vectors[:, 1], original_coords[:, 1])

    # Compute divergence
    divergence = np.sum(dx) + np.sum(dy)

    return divergence


def dewarp_labels(labels, target, frd, height, width, panorama_shape, image_target):

    # Read the coordinates from the text file
    rectangles_dewarped_points = []
    rectangles_dewarped = []

    debug_ratios = []

    for classid, x_center, y_center, width_norm, height_norm in labels:
        top_left, top_right, bottom_left, bottom_right = yolo_labels_to_points(
            x_center, y_center, width_norm, height_norm, width, height
        )

        # Dewarp coordinates
        top_left_new = frd.dewarp_coordinates(top_left[0], top_left[1])
        top_right_new = frd.dewarp_coordinates(top_right[0], top_right[1])
        bottom_right_new = frd.dewarp_coordinates(bottom_right[0], bottom_right[1])
        bottom_left_new = frd.dewarp_coordinates(bottom_left[0], bottom_left[1])

        # Recompute rectangles to fully cover objects in the new panoramic view
        x_coords = [
            top_left_new[0],
            top_right_new[0],
            bottom_left_new[0],
            bottom_right_new[0],
        ]
        y_coords = [
            top_left_new[1],
            top_right_new[1],
            bottom_left_new[1],
            bottom_right_new[1],
        ]
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # Define the corners of the proper rectangle
        top_left_new = (min_x, min_y)
        top_right_new = (max_x, min_y)
        bottom_left_new = (min_x, max_y)
        bottom_right_new = (max_x, max_y)

        # Get width and height of panorama image
        pan_width, pan_height = panorama_shape

        # Detect cases where object is split to the left and right of the panorama

        bbox_pano_width = top_right_new[0] - top_left_new[0]
        if bbox_pano_width > 0:
            bbox_ratio_pano = bbox_pano_width / pan_width

        threshold = 0.8
        if bbox_ratio_pano > threshold:

            # First bounding box on the left
            top_left_new_one = (0, top_left_new[1])
            bottom_left_new_one = (0, bottom_left_new[1])
            top_right_new_one = top_left_new
            bottom_right_new_one = bottom_left_new
            rectangles_dewarped_points.append(
                (
                    top_left_new_one,
                    top_right_new_one,
                    bottom_right_new_one,
                    bottom_left_new_one,
                )
            )
            rectangles_dewarped.append(
                points_to_yolo_string(
                    classid,
                    top_left_new_one,
                    top_right_new_one,
                    bottom_right_new_one,
                    bottom_left_new_one,
                    pan_width,
                    pan_height,
                )
            )

            # Second bounding box on the right
            top_right_new_two = (pan_width - 1, top_right_new[1])
            bottom_right_new_two = (pan_width - 1, bottom_right_new[1])
            top_left_new_two = top_right_new
            bottom_left_new_two = bottom_right_new
            rectangles_dewarped_points.append(
                (
                    top_left_new_two,
                    top_right_new_two,
                    bottom_right_new_two,
                    bottom_left_new_two,
                )
            )

            rectangles_dewarped.append(
                points_to_yolo_string(
                    classid,
                    top_left_new_two,
                    top_right_new_two,
                    bottom_right_new_two,
                    bottom_left_new_two,
                    pan_width,
                    pan_height,
                )
            )

        else:
            rectangles_dewarped_points.append(
                (top_left_new, top_right_new, bottom_right_new, bottom_left_new)
            )

            rectangles_dewarped.append(
                points_to_yolo_string(
                    classid,
                    top_left_new,
                    top_right_new,
                    bottom_right_new,
                    bottom_left_new,
                    pan_width,
                    pan_height,
                )
            )

    # Open the file in write mode
    with open(target, "w") as file:
        # Write each line to the file
        for line in rectangles_dewarped:
            file.write(line + "\n")  # Add a newline character at the end of each line

    cv2.destroyAllWindows()

    return rectangles_dewarped_points, rectangles_dewarped


# ------------------------BODY----------------------------------


def main():
    splits = "train", "val"
    for split in splits:
        image_folder_source = (
            "/home/dbogdoll/datasets/midadvrb_2000/images/" + split + "/"
        )
        image_folder_target = (
            "/home/dbogdoll/datasets/midadvrb_2000_dewarp/images/" + split + "/"
        )

        labels_folder_source = (
            "/home/dbogdoll/datasets/midadvrb_2000/labels/" + split + "/"
        )
        labels_folder_target = (
            "/home/dbogdoll/datasets/midadvrb_2000_dewarp/labels/" + split + "/"
        )

        for filename_img in tqdm(os.listdir(image_folder_source)):
            filename_label = filename_img.replace(".jpg", ".txt")
            image_source = image_folder_source + filename_img
            label_source = labels_folder_source + filename_label

            image_target = image_folder_target + filename_img
            label_target = labels_folder_target + filename_label

            labels = read_labels(label_source)

            fisheye_img, frd, height, width, panorama_shape = dewarp_fisheye(
                image_source,
                image_target,
                labels,
            )

            rectangles_dewarped_points, rectangles_dewarped = dewarp_labels(
                labels, label_target, frd, height, width, panorama_shape, image_target
            )

            if DEBUG:
                panorama_img = cv2.imread(image_target)
                color = (255, 0, 0)
                thickness = 5
                for (
                    top_left,
                    top_right,
                    bottom_right,
                    bottom_left,
                ) in rectangles_dewarped_points:
                    cv2.rectangle(
                        panorama_img, top_left, bottom_right, color, thickness
                    )
                cv2.imwrite(image_target, panorama_img)


if __name__ == "__main__":
    main()
