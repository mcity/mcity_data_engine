import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
# Load your existing dataset
#dataset = fo.load_dataset("codetr_mini_inference_1")
#dataset = fo.load_dataset("Abeyankar/mtl_ds_mini_fin")
dataset = load_from_hub("Abeyankar/mtl_ds_mini_fin")

# Create a view containing samples to edit
view = dataset  # or use .match(), .take(), etc.

# Unique identifier for this annotation session
anno_key = "cvat_existing_field_edit75"

# Upload in smaller chunks (avoid crashing CVAT server)
view.annotate(
    anno_key,
    label_field="ground_truth"
)