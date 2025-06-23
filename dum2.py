import fiftyone as fo

from fiftyone.utils.huggingface import load_from_hub


print(fo.list_datasets())

fo.delete_dataset('Abeyankar/mtl_ds_mini_fin')
#fo.delete_dataset('Abeyankar/mcity_clean_2844_crowd_updated')
print(fo.list_datasets())