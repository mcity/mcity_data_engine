import logging

import fiftyone.utils.random as four
from fiftyone import ViewField as F

from config.config import GLOBAL_SEED

#### FUNCTION MUST EXPECT A DATASET AND RETURN A VIEW ####


def keep_val_no_crowd_no_added(dataset, include_added=False):
    view_pot_train = dataset.match_tags(["val", "crowd"], bool=False)
    view_pot_train.untag_samples(["train"])

    if include_added == False:
        view_train = view_pot_train.match_tags(["added"], bool=False)
    else:
        view_train = view_pot_train
    view_train.tag_samples(["train"])

    view_train_val = view_train.match_tags(["train", "val"])
    logging.info(
        f"Reduced number of samples from {len(dataset)} to {len(view_train_val)}"
    )
    return view_train_val


def subset_splits(
    dataset,
    n_iteration,
    target_train_perc=0.8,
    target_test_perc=0,
    target_val_perc=0.2,
):

    dataset_perct_target = (n_iteration + 1) / 10
    logging.info(f"Trying to select {dataset_perct_target}% of the dataset.")
    dataset_perct = min(dataset_perct_target, 1)

    # All samples that are not reserved for eval
    view_pot_train = dataset.match_tags(["val", "crowd"], bool=False)
    view_pot_train.untag_samples(["train"])  # keep fixed val set (two cameras)

    n_samples_total = len(view_pot_train)
    n_samples_target = int(n_samples_total * dataset_perct)

    # Take percentage for train
    view_train = view_pot_train.take(n_samples_target, seed=GLOBAL_SEED)
    view_train.tag_samples(["train"])

    # gs_Huron_Plymouth
    # Main_stadium

    # view.untag_samples(["train", "test", "val"])
    # four.random_split(
    #    view_train,
    #    {
    #        "train": target_train_perc,
    #        "test": target_test_perc,
    #        "val": target_val_perc,
    #    },
    # )

    view_train_val = dataset.match_tags(["train", "val"])
    logging.info(
        f"Reduced number of samples from {len(dataset)} to {len(view_train_val)}"
    )
    logging.info(f"Split Distribution: {view_train_val.count_sample_tags()}")

    return view_train_val


def max_detections(
    dataset,
    target_train_perc=0.6,
    target_test_perc=0.2,
    target_val_perc=0.2,
    max_detections=7,
    gt_field="ground_truth",
):

    # Only consider frames with a max. number of detections
    num_objects = F(f"{gt_field}.detections").length()
    max_detections_view = dataset.match(num_objects <= max_detections)

    logging.info(
        f"Reduced number of samples from {len(dataset)} to {len(max_detections_view)}"
    )

    # Generate new splits
    max_detections_view.untag_samples(["train", "test", "val"])
    four.random_split(
        max_detections_view,
        {
            "train": target_train_perc,
            "test": target_test_perc,
            "val": target_val_perc,
        },
    )

    return max_detections_view


def vru_mcity_fisheye(
    dataset,
    target_train_perc=0.7,
    target_test_perc=0.2,
    target_val_perc=0.1,
    create_splits=False,
    random_split=True,
):

    # Only select labels of VRU classes
    vru_view = dataset.filter_labels(
        "ground_truth",
        (F("label") == "pedestrian") | (F("label") == "motorbike/cycler"),
    )

    n_samples = len(vru_view)
    logging.info(f"Reduced number of samples from {len(dataset)} to {n_samples}")
    logging.info(f"Original split distribution: {vru_view.count_sample_tags()}")

    if create_splits and random_split:
        vru_view.untag_samples(["train", "test", "val"])
        four.random_split(
            vru_view,
            {
                "train": target_train_perc,
                "test": target_test_perc,
                "val": target_val_perc,
            },
        )
    elif create_splits:
        # Get target number of samples per split
        target_n_train = int(n_samples * target_train_perc)
        target_n_test = int(n_samples * target_test_perc)
        target_n_val = int(n_samples * target_val_perc)

        # Get current number of samples for 'train' split and rest
        train_view = vru_view.match_tags(["train"])
        test_val_view = vru_view.match_tags(["train"], bool=False)
        n_samples_train = len(train_view)

        if target_n_test + target_n_val > len(test_val_view):
            logging.error(
                f"Target test/val count of {target_n_test + target_n_val} exceeds the number of available samples {test_val_view}."
            )
            return dataset

        test_val_view.untag_samples(["test", "val"])
        needed_train_from_test_val = target_n_train - n_samples_train
        if needed_train_from_test_val < 0:
            logging.error(
                f"Already {n_samples_train} samples labeled 'train', but requested {target_n_train} samples for 'train' split"
            )
            return dataset

        to_be_train = test_val_view.take(needed_train_from_test_val, seed=GLOBAL_SEED)
        to_be_train.tag_samples("train")

        test_val_view = test_val_view.match_tags("train", bool=False)
        to_be_test = test_val_view.take(target_n_test, seed=GLOBAL_SEED)
        to_be_test.tag_samples("test")

        test_val_view = test_val_view.match_tags(["train", "test"], bool=False)
        to_be_val = test_val_view.take(target_n_val, seed=GLOBAL_SEED)
        to_be_val.tag_samples("val")

    logging.info(f"New split distribution: {vru_view.count_sample_tags()}")
    return vru_view
