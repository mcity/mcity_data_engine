# Mcity Data Engine

![mcity_dataengine](https://github.com/user-attachments/assets/4b80c882-7522-4a06-8b15-c4e294b95b56)

<p align="center">
  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
  <img alt="Test Results" src="https://github.com/daniel-bogdoll/mcity_data_engine/actions/workflows/python-app.yml/badge.svg?branch=main"/>
  <img alt="Ubuntu Version" src="https://img.shields.io/badge/Ubuntu-24.04-blue"/>
  <img alt="Python Version" src="https://img.shields.io/badge/Python-3.12-blue"/>
  <img alt="CUDA Version" src="https://img.shields.io/badge/CUDA-12.4-blue"/>
</p>


The Mcity Data Engine is an essential tool in the Mcity makerspace for transportation innovators making AI algorithms and seeking actionable data insights through machine learning.

## Instructions

Download, install requirements, and run:
```
git clone --recurse-submodules git@github.com:daniel-bogdoll/mcity_data_engine.git
cd mcity_data_engine
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

To exclude the output of jupyter notebooks from git tracking, add the following lines to your ```.git/config```:

```
[filter "strip-notebook-output"]
clean = jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR
```

In case there are issues with MongoDB, use ```ps aux | grep mongod``` and ```kill``` the fiftyone process.

Development System Specifications:
- Ubuntu 24.04.1 LTS
- Python 3.12.3
- CUDA Version: 12.4 (RTX 4090)

## Structure

.   
    ├── ano_dec.py              # Anolib: Pixel-wise anomaly detection  
    ├── brain.py                # Voxel51: Frame-wise analysis of data   
    ├── config.py               # General config    
    ├── data_loader/            # Handles dataset loading   
    │   └── nuscenes-devkit     # Modified devkit for python 3.12   
    ├── datasets/               # Stores datasets and computed  embeddings  
    │   └── datasets.yaml       # Dataset related parameters    
    │   └── embeddings/         # Default storage for computed  embeddings  
    ├── docs/                   # pdoc Documentation    
    │   └── index.html          # Documentation entry point     
    ├── logs/                   # Default storage for logs  
    ├── main.py                 # Core of the framework     
    ├── requirements.txt        # pip install -r requirements.txt   
    ├── scripts/                # Experiments and one-time  operations  
    ├── tests/                  # Pytest cases  
    └── utils/                  # Utility functions 

## Documentation

Open the [docs/index.html](./docs/index.html) file locally with your browser to see the API documentation. The documentation is updated automatically with Github Actions, generating pull requests.

## Training

Training runs are logged with [Weights and Biases (WandB)](https://wandb.ai/mcity/mcity-data-engine). To fill queues with your local machine, you need to setup an [agent](https://docs.wandb.ai/guides/launch/setup-launch-docker):

```
wandb.login()
wandb launch-agent -q <queue-name> --max-jobs <n>
```

In order to change the standard WandB directory, add the following line to the bottom of your ```~/.profile``` file and then run ```source ~/.profile```:

```
export WANDB_DIR="/home/dbogdoll/mcity_data_engine/logs/wandb"
```

In order to execute jobs, the [following tools](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) need to be installed:

- [Docker](https://docs.docker.com/engine/install/ubuntu/) incl. [post-installation](https://docs.docker.com/engine/install/linux-postinstall/)
- [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)(https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Datasets

### [MARS](https://ai4ce.github.io/MARS/)
Make sure you have entered your [SSH key at huggingface](https://huggingface.co/settings/keys). Download the dataset with

```
sudo apt-get install git-lfs
git lfs install
git clone git@hf.co:datasets/ai4ce/MARS
```

If not installed yet, install the nuscenes-devkit with

```
cd data_loader/nuscenes-devkit
pip install .
```

### Mcity Fisheye 2000 (labeled)

Download the data with
```
cd datasets
scp -r <uniqname>@lighthouse.arc-ts.umich.edu:/nfs/turbo/coe-mcity/tinghanw/midadvrb_2000 .
```

If ```images/val``` contains json files, delete them.

### Mcity Fisheye 3-Months (unlabeled)
```
cd datasets
scp -r <uniqname>@lighthouse.arc-ts.umich.edu:/nfs/turbo/coe-mcity/tinghanw/midadv_swinl_label_veh_0p3_ped_0p3_2023q4_yolov8_format_v2 .
```

### Fisheye 8k


