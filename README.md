# Mcity Data Engine

![mcity_dataengine](https://github.com/user-attachments/assets/4b80c882-7522-4a06-8b15-c4e294b95b56)

<p align="center">
  <img alt="Test Results" src="https://github.com/mcity/mcity_data_engine/actions/workflows/tests_documentation.yml/badge.svg"/>
  <img alt="Test Results for UofM Cluster" src="https://github.com/mcity/mcity_data_engine/actions/workflows/lighthouse_build.yml/badge.svg"/>
  <img alt="Ubuntu Version" src="https://img.shields.io/badge/Ubuntu-24.04-blue"/>
  <img alt="Python Version" src="https://img.shields.io/badge/Python-3.12-blue"/>
  <img alt="PyTorch Version" src="https://img.shields.io/badge/PyTorch-2.5-blue"/>
  <img alt="CUDA Version" src="https://img.shields.io/badge/CUDA-12.4-blue"/>
  <img alt="Visitors" src="https://visitor-badge.laobi.icu/badge?page_id=mcity.mcity_data_engine"/>
</p>

The Mcity Data Engine is an essential tool in the Mcity makerspace for transportation innovators making AI algorithms and seeking actionable data insights through machine learning. Details on the Data Engine can be found in the [**Wiki**](https://github.com/mcity/mcity_data_engine/wiki).

## Instructions

At least one GPU is required for the Data Engine. Check the hardware setups we have tested in the [**Wiki**](https://github.com/mcity/mcity_data_engine/wiki/Environments). To download the repository and install requirements run:
```
git clone --recurse-submodules git@github.com:mcity/mcity_data_engine.git
cd mcity_data_engine
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Login with your [Weights and Biases](https://wandb.ai/) and [Hugging Face](https://huggingface.co/) accounts:
```
wandb.login()
huggingface-cli login
```

Launch a **Voxel51** session in one terminal:
```python session_v51.py```

Configure your run in the [config/config.py](https://github.com/mcity/mcity_data_engine/blob/main/config/config.py) and launch the **Mcity Data Engine** in a second terminal:
```python main.py```

Open Voxel51 in your browser:
```http://localhost:5151/```

In case there are issues with MongoDB, the underlying database Voxel51 uses, run ```ps aux | grep mongod``` and ```kill``` the fiftyone process.

### Notebooks and Submodules

To exclude the output of jupyter notebooks from git tracking, add the following lines to your ```.git/config``` :

```
[filter "strip-notebook-output-engine"]
    clean = <your_path>/mcity_data_engine/.venv/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout
    smudge = cat
    required = true
```

and those to ```.git/modules/mcity_data_engine_scripts/config```

```
[filter "strip-notebook-output-scripts"]
    clean = <your_path>/mcity_data_engine/.venv/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --to=notebook --stdin --stdout
    smudge = cat
    required = true
```

In order to keep the submodules updated, add the following lines to the top of your ```.git/hooks/pre-commit```:

```
git submodule update --recursive --remote
git add .gitmodules $(git submodule foreach --quiet 'echo $name')
```

## Repository Structure
```
.
├── main.py                   # Entry point of the framework → Terminal 1
├── session_v51.py              # Script to launch Voxel51 session → Terminal 2
├── workflows/                  # Workflows for the Mcity Data Engine
├── config/                     # Local configuration files
├── wandb_runs/                 # Entrypoints and configs for WandB experiments
├── utils/                      # General-purpose utility functions
├── cloud/                      # Scripts run in the cloud to pre-process data
├── datasets/                   # Default storage directory for datasets
├── output/                     # Default storage for models, embeddings, etc.
├── logs/                       # Default storage for log files
├── docs/                       # Documentation generated with `pdoc`
├── tests/                      # Tests using Pytest
├── custom_models/              # External models with containerized environments
├── mcity_data_engine_scripts/  # Experiment scripts and one-time operations
├── Dockerfile.wandb            # Dockerfile for WandB experiments
├── .vscode                     # Settings for VS Code IDE
├── .github/workflows/          # GitHub Action workflows
├── .gitignore                  # Files and directories to be ignored by Git
├── .gitattributes              # Rules for handling files like Notebooks during commits
├── .gitmodules                 # Configuration for managing Git submodules
├── .secret                     # Secret tokens (not tracked by Git)
└── requirements.txt            # Python dependencies (pip install -r requirements.txt)
```

## Documentation

Open the [docs/index.html](./docs/index.html) file locally with your browser to see the API documentation. The documentation is updated automatically by pdoc with Github Actions, generating pull requests.

## Training

Training runs are logged with [Weights and Biases (WandB)](https://wandb.ai/mcity/mcity-data-engine). 

In order to change the standard WandB directory, add the following line to the bottom of your ```~/.profile``` file and then run ```source ~/.profile```:

```
export WANDB_DIR="<your_path>/mcity_data_engine/logs"
```

## Contributors

Special thanks to these amazing people for contributing to the Mcity Data Engine! 🙌

<a href="https://github.com/mcity/mcity_data_engine/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mcity/mcity_data_engine" />
</a>

## Citation

If you use the Mcity Data Engine in your research, feel free to cite the project:

```bibtex
@article{bogdoll2024mcitydataengine,
  title={Mcity Data Engine},
  author={Bogdoll, Daniel and Stevens, Gregory},
  journal={GitHub. Note: https://github.com/mcity/mcity_data_engine},
  year={2024}
}
```

