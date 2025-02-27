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

<p align='center'>

<a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/mcity/mcity_data_engine/blob/main/fish_eye_8k_colab.ipynb">
<picture>
  <source srcset="https://github.com/user-attachments/assets/26c12ccc-327a-4702-a49d-76bfeb83bc62" width="15%">
  <img alt="Mcity Data Engine Google Colab Demo" src="">
</picture>
</a>

<a target="_blank" rel="noopener noreferrer" href="https://github.com/mcity/mcity_data_engine/wiki">
<picture>
  <source srcset="https://github.com/user-attachments/assets/e3e1cd10-5195-4db7-9147-11b75e078662" width="15%">
  <img alt="Mcity Data Engine Wiki" src="">
</picture>
</a>

<a target="_blank" rel="noopener noreferrer" href="https://mcity.github.io/mcity_data_engine/">
<picture>
  <source srcset="https://github.com/user-attachments/assets/b93f0c88-172d-4eed-8dac-3fdb82436f71"
width="15%">
  <img alt="Mcity Data Engine Docs" src="">
</picture>
</a>

<a target="_blank" rel="noopener noreferrer" href="https://wandb.ai/mcity">
<picture>
  <source srcset="https://github.com/user-attachments/assets/2e54c0ba-26b7-42cf-b33f-903ddfd55ae9" width="15%">
  <img alt="Mcity Data Engine Logs" src="">
</picture>
</a>

<a target="_blank" rel="noopener noreferrer" href="https://huggingface.co/mcity-data-engine">
<picture>
  <source srcset="https://github.com/user-attachments/assets/5b925a76-d0a2-46ad-8d95-b9296d6a5b46" width="15%">
  <img alt="Mcity Data Engine Models" src="">
</picture>
</a>
</p>

On February 24, 2025, Daniel Bogdoll, a research scholar at Mcity, gave a presentation on the first release of the Mcity Data Engine in Ann Arbor, Michigan. The recording provides insight into the general architecture, it's features and ecosystem integrations, and demonstrates successful data curation and model training for improved Vulnerable Road User (VRU) detection:

<div align="center">
      <a href="https://www.youtube.com/watch?v=ciT8YwQCHwo">
         <img src="https://github.com/user-attachments/assets/dcd2cd42-9cc0-4cf0-abab-a4d4ebd14198" style="width:60%;">
      </a>
</div>

## Online Demo: Data Selection with Embeddings

To get a first feel for the Mcity Data Engine, we provide an online demo in a [Google Colab](https://colab.research.google.com/github/mcity/mcity_data_engine/blob/main/fish_eye_8k_colab.ipynb) environment. We will load the [Fisheye8K dataset](https://huggingface.co/datasets/Voxel51/fisheye8k) and demonstrate the Mcity Data Engine workflow [Embedding Selection](https://github.com/mcity/mcity_data_engine/wiki/Workflows#embedding-selection). This workflow leverages a set of models to compute image embeddings which are used to determine both representative and rare samples. The dataset is then visualized in the Voxel51 UI, highlighting how often a sample was picked by the workflow.

Note that most of the Mcity Data Engine workflows require a more powerful GPU, so the possibilities within the Colab environment are limited. Other workflows may not work.

Online demo on Google Colab: [Mcity Data Engine Web Demo](https://colab.research.google.com/github/mcity/mcity_data_engine/blob/main/fish_eye_8k_colab.ipynb
)

## Local Execution

At least one GPU is required for many of the Mcity Data Engine workflows. Check the hardware setups we have tested in the [**Wiki**](https://github.com/mcity/mcity_data_engine/wiki/Environments). To download the repository and install the requirements run:
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
├── main.py                     # Entry point of the framework → Terminal 1
├── session_v51.py              # Script to launch Voxel51 session → Terminal 2
├── workflows/                  # Workflows for the Mcity Data Engine
├── config/                     # Local configuration files
├── utils/                      # General-purpose utility functions
├── cloud/                      # Scripts run in the cloud to pre-process data
├── docs/                       # Documentation generated with `pdoc`
├── tests/                      # Tests using Pytest
├── custom_models/              # External models with containerized environments
├── mcity_data_engine_scripts/  # Experiment scripts and one-time operations (Mcity internal)
├── .vscode                     # Settings for VS Code IDE
├── .github/workflows/          # GitHub Action workflows
├── .gitignore                  # Files and directories to be ignored by Git
├── .gitattributes              # Rules for handling files like Notebooks during commits
├── .gitmodules                 # Configuration for managing Git submodules
├── .secret                     # Secret tokens (not tracked by Git)
└── requirements.txt            # Python dependencies (pip install -r requirements.txt)
```

## Training

Training runs are logged with [Weights and Biases (WandB)](https://wandb.ai/mcity/mcity-data-engine).

In order to change the standard WandB directory, run

```
echo 'export WANDB_DIR="<your_path>/mcity_data_engine/logs"' >> ~/.profile
source ~/.profile
```

## Contribution

Contributions are very welcome! The Mcity Data Engine is a blueprint for data curation and model training and will not support every use case out of the box. Please find instructions on how to contribute here:

- [Contribute new workflow](https://github.com/mcity/mcity_data_engine/wiki/Workflows#how-to-add-a-new-workflow)
- [Contribute new dataset](https://github.com/mcity/mcity_data_engine/wiki/Datasets#how-to-add-a-new-dataset)
- [Contribute new model](https://github.com/mcity/mcity_data_engine/wiki/Models)

Special thanks to these amazing people for contributing to the Mcity Data Engine! 🙌

<a href="https://github.com/mcity/mcity_data_engine/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mcity/mcity_data_engine" />
</a>

## Citation

If you use the Mcity Data Engine in your research, feel free to cite the project:

```bibtex
@article{bogdoll2025mcitydataengine,
  title={Mcity Data Engine},
  author={Bogdoll, Daniel and Anata, Rajanikant Patnaik and Stevens, Gregory},
  journal={GitHub. Note: https://github.com/mcity/mcity_data_engine},
  year={2025}
}
```
