#!/bin/bash

# Ensure the dist_train.sh script is executable
chmod +x tools/dist_train.sh
chmod +x tools/slurm_train.sh
chmod +x tools/dist_test.sh

# Check if the first argument is "train", "slurm_train", or "test"
if [ "$1" == "train" ]; then
    # Train with 8 GPUs
    sh tools/dist_train.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py 1 path_to_exp
elif [ "$1" == "slurm_train" ]; then
    # Train using slurm
    sh tools/slurm_train.sh partition job_name projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_exp
elif [ "$1" == "test" ]; then
    # Test with 8 GPUs and evaluate
    sh tools/dist_test.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint 1 --eval bbox
else
    echo "Invalid argument. Use 'train', 'slurm_train', or 'test'."
    exit 1
fi