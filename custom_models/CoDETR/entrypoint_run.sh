#!/bin/bash

# Ensure the dist_train.sh script is executable
chmod +x tools/dist_train.sh
chmod +x tools/slurm_train.sh
chmod +x tools/dist_test.sh

# Install the package in editable mode
pip install -e .

# Default config and number of GPUs
CONFIG_FILE=${2:-"projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py"}
NUM_GPUS=${3:-1}
DATASET_NAME=${4:-"no_name"}
INFERENCE_DATASET_FOLDER=${5:-""}
INFERENCE_MODEL_CHECKPOINT=${6:-""}

# Check if the first argument is "train", "slurm_train", or "test"
if [ "$1" == "train" ]; then
    # Train with n GPUs
    ./tools/dist_train.sh $CONFIG_FILE $NUM_GPUS output
elif [ "$1" == "slurm_train" ]; then
    # Train using slurm
    ./tools/slurm_train.sh partition job_name $CONFIG_FILE output
elif [ "$1" == "test" ]; then
    # Test with n GPUs and evaluate
    ./tools/dist_test.sh $CONFIG_FILE output/latest.pth $NUM_GPUS --eval bbox --out output/test.pkl --cfg-options test_evaluator.classwise=True --eval-options classwise=True
elif [ "$1" == "test-output" ]; then
    # Test with n GPUs
    ./tools/dist_test.sh $CONFIG_FILE output/latest.pth $NUM_GPUS --format-only --options "jsonfile_prefix=./output/co_detr_test_results"
elif [ "$1" == "inference" ]; then
    # Run inference
    python mcity_data_engine/run_inference.py --config_file $CONFIG_FILE --data_folder_root $INFERENCE_DATASET_FOLDER --model_checkpoint $INFERENCE_MODEL_CHECKPOINT
elif [ "$1" == "interactive" ]; then
    # Start an interactive shell
    /bin/bash
elif [ "$1" == "clear-output" ]; then
    python mcity_data_engine/clear_output.py --config_file $CONFIG_FILE --dataset_name $DATASET_NAME
else
    echo "Invalid argument. Use 'train', 'slurm_train', 'test', 'test-output', 'inference', 'clear-output', or 'interactive'."
    exit 1
fi