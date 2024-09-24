import fiftyone as fo

def main():
    name = "midadvrb_2000_dewarpi"
    dataset_dir = "/home/dbogdoll/datasets/midadvrb_2000_dewarp/"
    dataset_type = fo.types.YOLOv5Dataset
    splits = ['train','val']

    dataset = fo.Dataset(name)
    for split in splits:
        dataset.add_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            split = split,
            tags=split
        )

    print(dataset)
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()
