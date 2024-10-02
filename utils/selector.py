from fiftyone import ViewField as F


def select_random(dataset, n_samples, seed=0):
    # Pick random number of samples

    random_view = dataset.take(n_samples, seed=seed)
    return random_view


def select_by_class(dataset, classes_in, classes_out):
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
