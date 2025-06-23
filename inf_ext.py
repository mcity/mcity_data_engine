import fiftyone as fo

# Load the full inference dataset
dataset = fo.load_dataset("mtl_mini_inference")
PRED_FIELD = "pred_od_co_deformable_detr_r50_1x_coco-visdrone_fisheye_mcity_2844_clean"

# Create new dataset
new_dataset = fo.Dataset(name="codetr_mini_inference_1", persistent=True)

# Skip first 2844 samples
for i, sample in enumerate(dataset.iter_samples(progress=True)):
    if i < 2844:
        continue

    # Only include samples with predictions
    if sample.has_field(PRED_FIELD) and sample[PRED_FIELD] is not None:
        preds = sample[PRED_FIELD].detections

        # Copy sample and predictions
        new_sample = fo.Sample(filepath=sample.filepath)
        gt_dets = [
            fo.Detection(label=det.label, bounding_box=det.bounding_box)
            for det in preds
        ]
        new_sample["ground_truth"] = fo.Detections(detections=gt_dets)
        new_dataset.add_sample(new_sample)

session = fo.launch_app(new_dataset)
session.wait(-1)
# Launch FiftyOne App
'''
#import fiftyone as fo

# Load the dataset
dataset = fo.load_dataset("codetr_inference_final")

# Define chunk size
chunk_size = 2547

# Get deterministic chunks using slicing
chunk1 = dataset[:chunk_size]
chunk2 = dataset[chunk_size:2*chunk_size]
chunk3 = dataset[2*chunk_size:]

# Save each chunk as a new dataset
fo.Dataset("codetr_pt1", persistent=True).add_samples(chunk1)
fo.Dataset("codetr_pt2", persistent=True).add_samples(chunk2)
fo.Dataset("codetr_pt3", persistent=True).add_samples(chunk3)

# Launch session on one chunk
#session = fo.launch_app(fo.load_dataset("codetr_chunk3"))
#session.wait(-1)
'''