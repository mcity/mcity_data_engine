import fiftyone as fo
from fiftyone.utils.huggingface import push_to_hub,load_from_hub

# Load original datasets
ds1 = load_from_hub("Abeyankar/mtl_ds_mini_fin")  # 795 images total
ds2 = load_from_hub("Abeyankar/mcity_clean_2844_crowd_updated")  # 2844 images total

# Create new dataset
new_ds = fo.Dataset(name="train_2844_mcityclean_val_7642_7k")

# Add all samples from ds2 as "train"
for sample in ds2.iter_samples(progress=True):
    sample.tags = ["train"]
    new_ds.add_sample(sample.copy())

# Add all samples from ds1 as "val"
for sample in ds1.iter_samples(progress=True):
    sample.tags = ["val"]
    new_ds.add_sample(sample.copy())

dataset = fo.load_dataset("train_2844_mcityclean_val_7642_7k")
push_to_hub(
    dataset,                      # dataset object
    "mtl_795_inference_ds",                  # repo name only (not full repo_id)
    private=False,
    commit_message="Hand annotated dataset MTL 7k",
    overwrite=True,
)
'''
# Launch the app
session = fo.launch_app(new_ds)
session.wait(-1)
'''