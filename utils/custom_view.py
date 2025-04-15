import logging

from fiftyone import ViewField as F

from config.config import GLOBAL_SEED

#### FUNCTION MUST EXPECT A DATASET AND RETURN A VIEW ####


def view_vru_mcity_fisheye(
    dataset, target_train_perc=0.7, target_test_perc=0.2, target_val_perc=0.1
):

    # Only select labels of VRU classes
    vru_view = dataset.filter_labels(
        "ground_truth",
        (F("label") == "pedestrian") | (F("label") == "motorbike/cycler"),
    )
    n_samples = len(vru_view)
    logging.info(f"Reduced number of samples from {len(dataset)} to {n_samples}")

    # Generate new splits
    # Get target number of samples per split
    logging.info(f"Original split distribution: {vru_view.count_sample_tags()}")
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
