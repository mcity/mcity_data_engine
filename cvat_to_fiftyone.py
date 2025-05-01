import fiftyone as fo

name = "mcity_clean_ds_2844"
data_path = "/home/dataengine/Downloads/mcity_new_ds/data"
labels_path = "/home/dataengine/Downloads/mcity_new_ds/annotations.xml"

# Import dataset by explicitly providing paths to the source media and labels
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.CVATImageDataset,
    data_path=data_path,
    labels_path=labels_path,
    name=name,
)

session = fo.launch_app(dataset)
session.wait(-1)  # (-1) forever