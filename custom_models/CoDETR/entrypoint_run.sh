#!/bin/bash

# Ensure the dist_train.sh script is executable
chmod +x tools/dist_train.sh
chmod +x tools/slurm_train.sh
chmod +x tools/dist_test.sh

# Install the package in editable mode
pip install -e .

# Check if the first argument is "train", "slurm_train", or "test"
if [ "$1" == "train" ]; then
    # Train with n GPUs
    ./tools/dist_train.sh projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py 1 output
elif [ "$1" == "slurm_train" ]; then
    # Train using slurm
    ./tools/slurm_train.sh partition job_name projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py output
elif [ "$1" == "test" ]; then
    # Test with n GPUs and evaluate
    ./tools/dist_test.sh projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py output/latest.pth 1 --eval bbox --out output/test.pkl --cfg-options test_evaluator.classwise=True --eval-options classwise=True
elif [ "$1" == "test-output" ]; then
    # Test with n GPUs
    ./tools/dist_test.sh projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py output/latest.pth 1 --format-only --options "jsonfile_prefix=./output/co_detr_test_results"
elif [ "$1" == "interactive" ]; then
    # Start an interactive shell
    /bin/bash
else
    echo "Invalid argument. Use 'train', 'slurm_train', or 'test'."
    exit 1
fi