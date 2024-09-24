# https://docs.voxel51.com/integrations/huggingface.html#loading-datasets-from-the-hub

from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub(
    "ai4ce/MARS",
    config_file="./mars.yaml",
)

# ValueError: Could not find fiftyone metadata for ai4ce/MARS (without the config_file)
# any repo on the Hugging Face Hub that contains a fiftyone.yml or fiftyone.yaml file
# can be loaded using the load_from_hub() function by passing the repo_id of the dataset, without needing to specify any additional arguments.
# --> Not the case here

# If you know how the dataset is structured, you can load the dataset by passing the path to a local yaml config file
# that describes the dataset via the config_file keyword argument.
# https://docs.voxel51.com/integrations/huggingface.html#huggingface-hub-load-dataset-config-kwargs
