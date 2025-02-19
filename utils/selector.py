import logging

from fiftyone import ViewField as F

from config.config import GLOBAL_SEED


def select_random(dataset, n_samples):
    """
    Select a random subset of samples from the dataset.

    Parameters:
    dataset (Dataset): The dataset from which to select samples.
    n_samples (int): The number of samples to select.
    seed (int, optional): The seed for the random number generator. Default is 0.

    Returns:
    Dataset: A dataset containing the randomly selected samples.
    """

    random_view = dataset.take(n_samples, seed=GLOBAL_SEED)
    return random_view


def select_by_class(dataset, classes_in=[], classes_out=[]):
    """
    Filters a dataset based on inclusion and exclusion of specified classes.

    Args:
        dataset: The dataset to be filtered.
        classes_in (str or list of str): Classes to include in the filtered dataset.
        classes_out (str or list of str): Classes to exclude from the filtered dataset.

    Returns:
        A filtered view of the dataset based on the specified inclusion and exclusion conditions.
    """
    incl_conditions = None
    excl_conditions = None

    if classes_in:
        if isinstance(classes_in, str):
            classes_in = [classes_in]
        for class_in in classes_in:
            condition = F("ground_truth.detections.label").contains(class_in)
            incl_conditions = (
                condition if incl_conditions is None else incl_conditions | condition
            )

    if classes_out:
        if isinstance(classes_out, str):
            classes_out = [classes_out]
        for class_out in classes_out:
            condition = ~F("ground_truth.detections.label").contains(class_out)
            excl_conditions = (
                condition if excl_conditions is None else excl_conditions & condition
            )

    if incl_conditions is not None and excl_conditions is not None:
        conditions = incl_conditions & excl_conditions
    elif incl_conditions is not None:
        conditions = incl_conditions
    elif excl_conditions is not None:
        conditions = excl_conditions
    else:
        conditions = None

    view = dataset.match(conditions) if conditions is not None else dataset
    return view


def generate_view_embedding_selection(
    dataset,
    configuration,
    embedding_selection_field="embedding_selection",
    embedding_count_field="embedding_selection_count",
):
    n_samples_in = len(dataset)
    min_selection_count = configuration["min_selection_count"]
    view_selection_count = dataset.match(
        F(embedding_count_field) >= min_selection_count
    )

    n_samples_out = len(view_selection_count)
    logging.info(
        f"Sample Reduction: {n_samples_in} -> {n_samples_out}. Workflow 'Embedding Selection'"
    )
    return view_selection_count


def generate_view_anomaly_detection_selection(
    dataset, configuration, field_anomaly_score_root="pred_anomaly_score_"
):
    n_samples_in = len(dataset)
    model_name = configuration["model"]
    min_anomaly_score = configuration["min_anomaly_score"]
    field_name_anomaly_score = field_anomaly_score_root + model_name
    view_anomaly_score = dataset.match(F(field_name_anomaly_score) >= min_anomaly_score)

    n_samples_out = len(view_anomaly_score)
    logging.info(
        f"Sample Reduction: {n_samples_in} -> {n_samples_out}. Workflow 'Anomaly Detection'"
    )
    return view_anomaly_score


def generate_view_ensemble_selection(
    dataset,
    configuration,
    ensemble_selection_field="n_unique_ensemble_selection",
    ensemble_selection_tag="detections_overlap",
):
    n_samples_in = len(dataset)
    min_n_unique_selection = configuration["min_n_unique_selection"]
    view_n_unique_exploration = dataset.match(
        F(ensemble_selection_field) >= min_n_unique_selection
    )
    view_tagged_labels = view_n_unique_exploration.select_labels(
        tags=ensemble_selection_tag
    )

    n_samples_out = len(view_tagged_labels)
    logging.info(
        f"Sample Reduction: {n_samples_in} -> {n_samples_out}. Workflow 'Ensemble Selection'"
    )

    return view_tagged_labels
