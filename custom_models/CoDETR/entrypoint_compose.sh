#!/bin/bash

# Ensure the dist_train.sh script is executable
chmod +x tools/dist_train.sh
chmod +x tools/slurm_train.sh
chmod +x tools/dist_test.sh
exec "$@"
