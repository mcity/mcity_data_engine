# Mcity Data Engine
[![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-blue)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.4-blue)]()

[![Basic validation](https://github.com/daniel-bogdoll/mcity_data_engine/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/daniel-bogdoll/mcity_data_engine/blob/main/.github/workflows/python-app.yml)


Download, install requirements, and run:
```
git clone --recurse-submodules git@github.com:daniel-bogdoll/mcity_data_engine.git
cd mcity_data_engine
pip install -e ./data_loader/nuscenes-devkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
fiftyone plugins download https://github.com/jacobmarks/clustering-plugin
python main.py
```

In case there are issues with MongoDB, use ```ps aux | grep mongod``` and ```kill``` the fiftyone process.

Development System Specifications:
- Ubuntu 24.04.1 LTS
- Python 3.12.3
- CUDA Version: 12.4 (RTX 4090)

## Structure

    .
    ├── datasets                # Stores datasets and computed embeddings
        ├── datasets.yaml       # Dataset related parameters
    ├── data_loader             # Handles dataset loading
    ├── main.py                 # Core of the framework
    ├── brain.py                # V51 Analysis of data
    └── config.yaml             # Main parameters

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


