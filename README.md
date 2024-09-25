# Mcity Data Engine

Developed under Ubuntu 24.04 and Python 3.12.

Download and install requirements:
```
git clone --recurse-submodules git@github.com:daniel-bogdoll/mcity_data_engine.git
pip install -r "requirements.txt"
```

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


