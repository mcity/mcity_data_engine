import logging

import fiftyone as fo


def arrange_fields_in_groups(dataset):
    """Arranges dataset fields into groups based on predefined task categories and updates the dataset's sidebar configuration."""
    task_field_mapping = {  # Prefixes end with a "_"
        "Object Detection": ["pred_od_"],
        "Zero-Shot Object Detection": ["pred_zsod_"],
        "Semantic Segmentation": ["pred_ss_"],
        "Depth Estimation": ["pred_de_"],
        "Anomaly Detection": ["pred_anomaly_"],
        "Embedding Selection": [
            "embedding_selection",
            "embedding_selection_model",
            "embedding_selection_count",
        ],
        "Embedding Computation": ["representativeness_", "uniqueness_"],
        "Evaluation": ["eval_"],
        "Data Recording": ["location", "timestamp", "name", "time_of_day"],
    }

    dataset_fields = dataset.get_field_schema()

    # Create a new dict for each task which has a list of all fields that match the prefix or fieldname in dataset_fields
    task_groups = {}
    for task, prefixes in task_field_mapping.items():
        task_groups[task] = []
        for field_name in dataset_fields:
            # Check if field matches any prefix or exact field name
            if (
                any(field_name.startswith(prefix) for prefix in prefixes)
                or field_name in prefixes
            ):
                task_groups[task].append(field_name)
                logging.info(f"Added field {field_name} to group {task}")

    sidebar_groups = fo.DatasetAppConfig.default_sidebar_groups(dataset)

    # Create a mapping of fields to their original groups for later removal
    field_to_original_group = {}
    for group in sidebar_groups:
        for path in group.paths:
            field_to_original_group[path] = group.name

    # Create a new group for every task that has at least one field
    n_added_groups = 0
    moved_fields = set()
    for task, fields in task_groups.items():
        if len(fields) > 0:
            group = fo.SidebarGroupDocument(name=task)
            group.paths = []
            for field in fields:
                group.paths.append(field)
                moved_fields.add(field)
            sidebar_groups.append(group)
            n_added_groups += 1

    if n_added_groups > 0:
        for group in sidebar_groups:
            # Collapse default groups
            if group.name in ["metadata", "other", "primitives"]:
                group.expanded = False

            # Remove moved fields from their original groups
            group.paths = [
                path
                for path in group.paths
                if not (
                    path in moved_fields and field_to_original_group[path] == group.name
                )
            ]

        dataset.app_config.sidebar_groups = sidebar_groups
        dataset.save()
