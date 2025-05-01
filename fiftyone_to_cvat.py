import fiftyone as fo


# Load your existing dataset
dataset = fo.load_dataset("mcity-data-engine/mcity-fisheye-vru-2844")

# Create a view containing samples to edit (use .select() for specific samples)
view = dataset  # Edit entire dataset, or filter with .match(), .take(), etc.

# Launch CVAT annotation editor for existing bounding boxes
anno_key = "cvat_existing_field_edit3"  # Unique identifier for this annotation session
view.annotate(anno_key, label_field="ground_truth")
