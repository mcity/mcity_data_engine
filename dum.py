import fiftyone as fo

print(fo.list_datasets())
fo.delete_dataset("custom_dataset")
print(fo.list_datasets())