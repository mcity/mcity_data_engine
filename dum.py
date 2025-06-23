import fiftyone as fo

from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc

dataset = load_from_hub("Abeyankar/mtl_ds_mini_fin",persistent=True)

# Launch the App
session = fo.launch_app(dataset)
session.wait(-1)
'''

print(fo.list_datasets())

fo.delete_dataset('codetr_inference_new_final2')
fo.delete_dataset('codetr_inference_new_final3')
fo.delete_dataset('codetr_inference_new_final4')
fo.delete_dataset('codetr_pt1')
fo.delete_dataset('codetr_pt2')
fo.delete_dataset('codetr_pt3')
fo.delete_dataset('codetr_chunk1')
fo.delete_dataset('codetr_chunk2')
fo.delete_dataset('codetr_chunk3')
print(fo.list_datasets())
'''