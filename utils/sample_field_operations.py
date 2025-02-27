import logging

import fiftyone as fo


def add_sample_field(
    v51_dataset_or_view, v51_field, v51_type, v51_embedded_doc_type=None
):
    """Wrapper for V51 function 'add_sample_field()' to also work for dataset views"""
    type_dataset = fo.core.dataset.Dataset
    type_dataset_view = fo.core.view.DatasetView
    try:
        if type(v51_dataset_or_view) == type_dataset:
            dataset_operation = v51_dataset_or_view
        elif type(v51_dataset_or_view) == type_dataset_view:
            dataset_operation = fo.load_dataset(v51_dataset_or_view.dataset_name)
        else:
            logging.error(
                f"Type {type(v51_dataset_or_view)} not supported for variable 'v51_dataset_or_view'"
            )

        dataset_operation.add_sample_field(v51_field, v51_type, v51_embedded_doc_type)
    except Exception as e:
        logging.error(f"Adding field {v51_field} of type {v51_type} failed: {e}")


def rename_sample_field(v51_dataset_or_view, v51_field_old, v51_field_new):
    """Wrapper for V51 function 'rename_sample_field()' to also work for dataset views"""
    type_dataset = fo.core.dataset.Dataset
    type_dataset_view = fo.core.view.DatasetView
    try:
        if type(v51_dataset_or_view) == type_dataset:
            dataset_operation = v51_dataset_or_view
        elif type(v51_dataset_or_view) == type_dataset_view:
            dataset_operation = fo.load_dataset(v51_dataset_or_view.dataset_name)
        else:
            logging.error(
                f"Type {type(v51_dataset_or_view)} not supported for variable 'v51_dataset_or_view'"
            )

        dataset_operation.rename_sample_field(v51_field_old, v51_field_new)
    except Exception as e:
        logging.error(f"Renaming field {v51_field_old} to {v51_field_new} failed: {e}")
