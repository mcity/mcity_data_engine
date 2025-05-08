# agent_tools.py
import subprocess
from pathlib import Path
import re

CONFIG_FILE = Path("config/config.py")


def update_config(workflow: str, dataset: str = None, model: str = None) -> str:
    lines = CONFIG_FILE.read_text().splitlines()

    # Update SELECTED_WORKFLOW
    for i, line in enumerate(lines):
        if line.strip().startswith("SELECTED_WORKFLOW"):
            lines[i] = f'SELECTED_WORKFLOW = ["{workflow}"]'
            break

    # Update SELECTED_DATASET
    if dataset:
        in_dataset_block = False
        for i, line in enumerate(lines):
            if 'SELECTED_DATASET = {' in line:
                in_dataset_block = True
            if in_dataset_block and '"name":' in line:
                lines[i] = f'    "name": "{dataset}",'
                break

    # Enable only ultralytics model if specified
    if model:
        in_model_source_block = False
        for i, line in enumerate(lines):
            if '"model_source": [' in line:
                in_model_source_block = True
            elif in_model_source_block and '],' in line:
                in_model_source_block = False
            elif in_model_source_block:
                if '"ultralytics"' in line:
                    lines[i] = '            "ultralytics",'
                else:
                    lines[i] = f'            #"{line.strip().strip(",# ")}"'

        in_ultralytics_block = False
        for i, line in enumerate(lines):
            if '"ultralytics": {' in line:
                in_ultralytics_block = True
            if in_ultralytics_block and '"models": {' in line:
                start = i + 1
                while not lines[start].strip().startswith("}"):
                    model_line = lines[start].strip().lstrip("#").strip()
                    if model in model_line:
                        lines[start] = f'                "{model}": {{"batch_size": 16, "img_size": 960}},'
                    else:
                        lines[start] = f'                #{model_line}'
                    start += 1
                break

    CONFIG_FILE.write_text("\n".join(lines))
    return f"✅ Updated config.py to use workflow '{workflow}' with dataset '{dataset}' and model '{model}'"


def run_main_py() -> str:
    process = subprocess.Popen(["python", "main.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = []
    for line in iter(process.stdout.readline, b''):
        decoded = line.decode("utf-8")
        print(decoded, end="")
        output.append(decoded)
    process.stdout.close()
    process.wait()
    return "".join(output)


def list_available_workflows() -> list:
    lines = CONFIG_FILE.read_text().splitlines()
    workflows = []
    in_workflows = False
    for line in lines:
        if line.strip().startswith("WORKFLOWS ="):
            in_workflows = True
            continue
        if in_workflows:
            if line.strip().startswith("}"):
                break
            match = re.match(r'\s+"(.*?)": {', line)
            if match:
                workflows.append(match.group(1))
    return workflows


def run_workflow_via_llm(client, user_prompt: str):
    workflows = list_available_workflows()
    prompt = (
        "The MCity Data Engine supports the following workflows: "
        f"{', '.join(workflows)}.\n"
        "Given the user request below, identify:\n"
        "- the most appropriate workflow\n"
        "- the dataset name (if any)\n"
        "- the model name (if any)\n\n"
        f"User: \"{user_prompt}\"\n"
        "Respond strictly in this format:\n"
        "workflow=<workflow>; dataset=<dataset>; model=<model>\n"
        "Use 'None' if dataset or model are not mentioned."
    )

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a config assistant for MCity's Data Engine."},
            {"role": "user", "content": prompt}
        ]
    )
    content = response.choices[0].message.content.strip()
    match = re.match(r"workflow=(.*?); dataset=(.*?); model=(.*)", content)
    if not match:
        return f"❌ Failed to parse LLM response:\n{content}"

    workflow, dataset, model = match.groups()
    dataset = None if dataset == "None" else dataset
    model = None if model == "None" else model

    update_msg = update_config(workflow.strip(), dataset, model)
    run_output = run_main_py()
    return f"{update_msg}\n\n🛠️ main.py output:\n{run_output}"
