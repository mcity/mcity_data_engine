import importlib.util
import subprocess
import re
import json
import os
from groq import Groq

CONFIG_PATH = "/home/dataengine/Desktop/mcity_data_engine/config/config.py"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def load_config():
    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def build_system_prompt():
    with open(CONFIG_PATH, "r") as f:
        config_text = f.read()

    return f"""
You are the MCity Data Engine Agent. Your job is to help users explore and configure the MCity Data Engine using only the workflows, models, and parameters defined in the `config.py` file shown below.

❗ Important rules:
- DO NOT invent models or workflows.
- Only refer to what is explicitly listed in the WORKFLOWS dictionary inside config.py.
- Do not assume any default ML models like 'random_forest' or 'gradient_boosting' unless they are present in the config.
- When listing workflows, use the exact keys from WORKFLOWS (e.g., auto_labeling, anomaly_detection, mask_teacher, etc.).

Here is the full content of config.py:


```python
{config_text}

```

You must respond with a single, syntactically correct Python code block that contains only the updated parts of `config.py`.

❗ Rules:
- DO NOT include duplicate keys (e.g., no two 'epochs' entries).
- DO NOT add explanatory comments inside the Python code block.
- DO NOT include '...' or partial ellipsis — return full valid Python syntax.
- Always return a valid `dict` structure without unmatched brackets.

Example (GOOD):
```python
SELECTED_WORKFLOW = ["auto_labeling"]
SELECTED_DATASET = {"name": "fisheye8k", "n_samples": None}
WORKFLOWS = {
    "auto_labeling": {
        "mode": ["train"],
        "epochs": 7,
        "model_source": ["ultralytics"],
        "ultralytics": {
            "export_dataset_root": "output/datasets/ultralytics_data/",
            "models": {
                "yolo12n": {"batch_size": 16, "img_size": 960}
            }
        }
    }
}

"""

def ask_groq(system_prompt, user_input):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content.strip()

def extract_code_block(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None

def apply_config_patch(patch_code):
    with open(CONFIG_PATH, "r") as f:
        original = f.read()

    # Replace SELECTED_WORKFLOW
    patch_workflow = re.search(r'SELECTED_WORKFLOW\s*=\s*\[.*?\]', patch_code)
    if patch_workflow:
        original = re.sub(r'SELECTED_WORKFLOW\s*=\s*\[.*?\]', patch_workflow.group(0), original)

    # Replace SELECTED_DATASET
    patch_dataset = re.search(r'SELECTED_DATASET\s*=\s*\{.*?\}', patch_code, re.DOTALL)
    if patch_dataset:
        original = re.sub(r'SELECTED_DATASET\s*=\s*\{.*?\}', patch_dataset.group(0), original, flags=re.DOTALL)

    # Replace specific WORKFLOWS[...] block
    patch_workflow_block = re.search(r'"(.*?)"\s*:\s*\{.*?\}', patch_code, re.DOTALL)
    if patch_workflow_block:
        key = patch_workflow_block.group(1)
        pattern = rf'"{key}"\s*:\s*\{{.*?\}}(,?)'
        original = re.sub(
            pattern,
            patch_workflow_block.group(0) + r'\1',
            original,
            flags=re.DOTALL
        )
        with open(CONFIG_PATH, "w") as f:
            f.write(original)

def main():
    system_prompt = build_system_prompt()

    print("\n👋 Welcome to the MCity AI Agent (Groq Mode)")
    while True:
        user_input = input("You: ")
        llm_output = ask_groq(system_prompt, user_input)
        print(f"\nAgent: {llm_output}\n")

        patch = extract_code_block(llm_output)
        if patch:
            apply_config_patch(patch)
            print("\n✅ Config updated based on agent's response. Launching main.py...\n")
            subprocess.run(["python", "main.py"])
            break

if __name__ == "__main__":
    main()
