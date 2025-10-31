## Acknowledgements
Mcity would like to thank Amazon Web Services (AWS) for their pivotal role in providing the cloud infrastructure on which the Data Engine depends. We couldn’t have done it without their tremendous support!

# Agentic Mcity Data Engine

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

The Agentic MCity Data Engine introduces a conversational AI layer that sits seamlessly on top of the core data engine, enabling natural language interaction with complex computer vision workflows. Built using the Model Context Protocol (MCP), the agent acts as an intelligent orchestrator that guides users through workflow configuration and execution without requiring deep technical knowledge.

<div align="center">
  <picture>
    <source srcset="https://github.com/user-attachments/assets/19f326be-6588-457a-92d4-b7ec08f7491b" width="75%">
    <img alt="Agentic Mcity Data Engine Architecture" src="">
  </picture>
  <p><em>Figure 1. The Agentic Mcity Data Engine bridges programmatic and natural-language workflows through an LLM-agnostic MCP layer.</em></p>
</div>


On February 24, 2025, Daniel Bogdoll, a research scholar at Mcity, gave a presentation on the first release of the Mcity Data Engine in Ann Arbor, Michigan. The recording provides insight into the general architecture, its features and ecosystem integrations, and demonstrates successful data curation and model training for improved Vulnerable Road User (VRU) detection:
<div align="center">
      <a href="https://www.youtube.com/watch?v=ciT8YwQCHwo">
         <img src="https://github.com/user-attachments/assets/dcd2cd42-9cc0-4cf0-abab-a4d4ebd14198" style="width:60%;">
      </a>
</div>

## Key Features of the Agentic Implementation:
The Agentic Mcity Data Engine extends the Mcity Data Engine with an LLM-agnostic orchestration layer powered by the Model Context Protocol (MCP).
This layer transforms each workflow—such as auto-labeling, class mapping, or embedding selection—into structured, callable tools that can be accessed either through natural-language interaction or programmatic APIs.

<div align="center"> <picture> <source srcset="https://github.com/user-attachments/assets/3a5c751d-a386-4170-bd44-a29783fc92d6" width="80%"> <img alt="Agentic Mcity Data Engine Detailed Architecture" src=""> </picture> <p><em>Figure 2. Agentic Mcity Data Engine architecture – detailed interaction between user, LLM, chat server, MCP tool server.</em></p> </div>

Natural Language Configuration: Configure complex workflows through conversational commands instead of manually editing Python config files. The agent translates natural language requests into correct configuration settings, validates parameters, maintains context across conversation turns, and guides users through multi-step workflow setup with intelligent prompts and error prevention.

### Core Components:

**- User Interface :** A unified entry point for interaction—users can chat via a natural-language web UI or send direct HTTP API requests from the terminal.

**- Chat Server:**  A FastAPI service (port 8001) acting as the bridge between the user, LLM, and backend MCP services.
It maintains multi-turn chat history, handles tool invocations, streams Server-Sent Event (SSE) logs, and supports both web-UI and programmatic clients.

**- LLM Layer (Model-Agnostic):** Connects to OpenAI GPT-4o, Google Gemini, or Groq Llama models.
The LLM interprets user instructions, determines the appropriate workflow tool call, and sends structured requests back to the chat server for execution.

**- MCP Server:** A FastAPI-based backend (port 8000) exposing 40 + tools that represent the core Mcity Data Engine workflows.

**- Data Ingestion Server:** A dedicated service (port 8002) for uploading and preprocessing datasets.
It supports drag-and-drop ingestion of images, videos, and annotations in COCO, YOLO, or CVAT-XML formats, automatically converting them into FiftyOne-compatible datasets.
This server streams conversion logs and progress via SSE and updates datasets.yaml dynamically to register new datasets for use across workflows.

**- Data Engine Core:** The underlying Mcity Data Engine handling data selection, labeling, training, validation, and visualization.
The agentic layer orchestrates these modules programmatically via MCP instead of relying on static configuration editing.




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
pip install -r agent_requirements.txt
```

Login with your [Weights and Biases](https://wandb.ai/) and [Hugging Face](https://huggingface.co/) accounts:
```
wandb login
huggingface-cli login
```

### Agentic Implementation Guide:

**Configuration**: Create a .env file with your LLM API key.
```
# Choose your LLM provider (openai, gemini, or groq)
LLM_PROVIDER=openai

# Add your API key
OPENAI_API_KEY=sk-...
# OR
GEMINI_API_KEY=...
# OR
GROQ_API_KEY=...
```

**Launch the Agentic Interface:**
Start all four components one after the other in separate terminals:
```
# Terminal 1: MCP Tool Server (port 8000)
python mcp_layer/mcp_server.py

# Terminal 2: Chat Server (port 8001)
uvicorn mcp_layer.chat_server:app --port 8001 --reload

# Terminal 3: Data Ingestion Server (port 8002)
uvicorn mcp_layer.ingest_server:app --host 0.0.0.0 --port 8002 --reload

# Terminal 4: Launch Web UI (port 5225)
python mcp_layer/client_chat.py --ui
```

The web interface will open automatically at http://localhost:5225.

**Talk to Task: Example Run**
```
User: Hi, I would like to explore the data engine for processing my data.

Agent: Great! We have several workflows you can choose from to process your data:
        1. Auto Labeling
        2. Class Mapping
        3. Anomaly Detection
        4. Embedding Selection
        5. Zero-Shot Auto Labeling
        6. Ensemble Selection

        Which workflow would you like to start with?

User: I would like to generate labels for my dataset. It’s completely unlabeled right now.

Agent: Since your dataset has no labels, the best fit is the Zero-Shot Auto Labeling workflow.
       This workflow lets you generate labels automatically without any prior annotations.
       You can choose from the following state-of-the-art zero-shot models:
          1. omlab/omdet-turbo-swin-tiny-hf
          2. IDEA-Research/grounding-dino-tiny
          3. google/owlvit-large-patch14
          4. google/owlv2-base-patch16-finetuned
          5. google/owlv2-large-patch14-ensemble

User: I would like to use the first 3 models.

Agent: The models have been successfully updated.
       Would you like to modify the detection threshold value (default is 0.2) for the zero-shot models?

User: Can you explain what the detection threshold means?

Agent: The detection threshold is a confidence score that decides whether a detected object is valid.
       If the model’s confidence is above this value, the detection is accepted; if below, it’s rejected.
       Lower thresholds yield more detections but may include false positives, while higher thresholds are more selective.
       The default is 0.2, meaning detections above 20% confidence are accepted.

User: Set threshold to 0.3.

Agent: Threshold set. Next, provide the object classes to detect from the data.

User: Car, bike, pedestrian, and truck.

Agent: Classes set to car, bike, pedestrian, truck. The workflow is ready to run.
       Let me know when to start.

User: We can begin execution of the workflow.

[Data Engine Server]: Executing Zero-Shot Auto Labeling… live logs are streamed to Weights and Biases.

Agent: Zero-Shot Auto Labeling completed successfully.
       You can now use Ensemble Selection to find detections where models agree.
       I can also assist you with visualizing detections using Voxel51.
```

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
├── mcp_layer/  # Experiment scripts and one-time operations (Mcity internal)
│  ├── mcp_server.py           # MCP tool registry (port 8000)
│  ├── chat_server.py          # FastAPI chat endpoint (port 8001)
│  ├── ingest_server.py        # File upload & processing (port 8002)
│  ├── client_chat.py          # Web/terminal client (port 5225)
│  ├── mcptools/               # Tool implementations
│  │   ├── __init__.py
│  │   ├── workflow_selector.py
│  │   ├── auto_labeling.py
│  │   ├── class_mapping.py
│  │   ├── anomaly_detection.py
│  │   ├── embedding_selection.py
│  │   ├── zsal.py             # Zero-shot auto-labeling
│  │   ├── ensemble_selection.py
│  │   ├── data_ingest.py
│  │   └── v51.py              # Voxel51 integration
│  ├── llm_clients.py          # Multi-LLM support
│  ├── tool_schema.py          # OpenAI tool definitions
│  └── ui/                     # Web interface assets
│      └── index.html
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
