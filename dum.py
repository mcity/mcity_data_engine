import fiftyone as fo

print(fo.list_datasets())
fo.delete_dataset('visdrone_fisheye-v51-complete')
print(fo.list_datasets())