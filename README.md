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

> [!NOTE]
> In case there are issues with MongoDB, use ```ps aux | grep mongod``` and ```kill``` the fiftyone process.

## Repository Structure

.   
├── main.py                   # Core of the framework     
├── ano_dec.py                # WORKFLOW: Pixel-wise anomaly detection with anomalib  
├── brain.py                  # WORKFLOW: Frame-wise analysis of data with Voxel51  
├── teacher.py                # WORKFLOW: Training teacher model with labeled data   
├── config/                   # Local configuration files  
├── utils/                    # Utility functions  
├── scripts/                  # Experiments and one-time operations  
├── logs/                     # Default storage for logs  
├── datasets/                 # Default storage for datasets  
├── output/                   # Default storage for models, embeddings etc.  
├── docs/                     # pdoc Documentation    
├── tests/                    # Pytest cases  
├── wandb_runs/               # Entrypoints and configs for WandB experiments  
├── Dockerfile.wand           # Dockerfile for WandB experiments  
├── .github/workflows         # Github Action Workflows  
├── .gitignore                # Ignored files for Git tracking  
├── .gitattributes            # Used to clean Notebooks prior to commits  
├── .gitmodules               # Managing Git submodules  
├── .secret                   # Secret TOKENS (untracked)  
└── requirements.txt          # pip install -r requirements.txt   


## Documentation

Open the [docs/index.html](./docs/index.html) file locally with your browser to see the API documentation. The documentation is updated automatically by pdoc with Github Actions, generating pull requests.

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
- [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [GitHub Authentication Caching](https://docs.github.com/en/get-started/getting-started-with-git/caching-your-github-credentials-in-git) with SSH

To launch a job on your machine with Docker, an [active agent](https://wandb.ai/mcity/launch/UnVuUXVldWU6NDQ0OTE4MA==/agents) needs to be running in a terminal. Adapt the [config](https://wandb.ai/mcity/launch/UnVuUXVldWU6NDQ0OTE4MA==/config) as necessary. Then, run

```
wandb launch --uri "git@github.com:daniel-bogdoll/mcity_data_engine.git" --job-name <name-run> --project mcity-data-engine --entry-point "python main.py" --dockerfile Dockerfile.wandb --queue data-engine
```

Locally, you will need to clean up old docker images once in a while. Run ```docker image prune --all --filter until=48h``` to delete docker images older than 48 hours.

## Datasets

### Huggingface Integration

To upload datasets to Huggingface with Voxel51, load private datasets from Huggingface, and run the tests successfully, you need to set a [Huggingface token](https://huggingface.co/docs/hub/en/security-tokens):

```.secrets``` file in the root folder with the following content:
```
HF_TOKEN=<YOUR_HF_TOKEN>
```

[Secret in Github Action](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions) with the same name and content as above.




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

### Mcity Fisheye Anomalies: Pedestrians
This dataset is based on the ```Mcity Fisheye 2000 (labeled)``` dataset. It computes the splits newly with a ```train``` split that does not contain any pedestrians, and a ```val``` split which has pedestrians on every frame. This way, pedestrians can be treated as anomalies. It was first designed to evaluate the Anomalib library.


### Fisheye 8k
Follow the instructions from the [MoyoG/FishEye8K repository](https://github.com/MoyoG/FishEye8K) to download the dataset. Afterwards, delete intermediate folders, such that the ```test``` and ```train``` folders are at the root level of the dataset folder. Scripts are designed for the dataset version ```Fisheye8K_all_including_train&test_update_2024Jan Update.zip```.


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


