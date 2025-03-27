import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub, push_to_hub

# Load the original dataset
dataset = load_from_hub("Abeyankar/Visdrone_fisheye-v51-complete")

# Create a new dataset with only the 6000th sample
new_dataset = fo.Dataset()
sample = dataset.skip(5999).first()
new_dataset.add_sample(sample)

# Upload the new dataset to Hugging Face Hub
# Replace 'your-username' and 'your-dataset-name' with appropriate values
push_to_hub(new_dataset, "visheye_dummy", public=True)

print("New dataset uploaded successfully!")
