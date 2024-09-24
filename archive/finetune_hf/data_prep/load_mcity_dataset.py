from datasets import load_dataset, Image, Sequence, Features, Value, ClassLabel

DATA_PATH = "/home/dbogdoll/datasets/midadvrb_2000_hf"

class_names = [
    "car",
    "truck",
    "bus",
    "trailer",
    "motorbike/cycler",
    "pedestrian",
    "van",
    "pickup",
]

features = Features(
    {
        "image": Image(),
        "objects": {
            "bbox": Sequence(Sequence(Value("float32"), length=4)),
            "categories": Sequence(ClassLabel(names=class_names)),
        },
    }
)

dataset = load_dataset("imagefolder", data_dir=DATA_PATH, features=features)
print(dataset["train"][0])

dataset.push_to_hub("dbogdollumich/mcity_fisheye", private=True)
