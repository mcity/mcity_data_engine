from mcptools import mcp  #shared instance from __init__.py
import re
import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "config" / "config.py"
MAIN_PATH = ROOT_DIR / "main.py"



@mcp.tool()
def configure_auto_labeling(selected_source: str, selected_model: str) -> str:
    """
    Enable the selected model source and model inside config.py for the auto_labeling workflow.
    """
    selected_source = selected_source.lower().strip()
    selected_model = selected_model.strip()
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []

    in_model_source = False
    in_selected_source = False
    in_model_dict = False
    in_config_list = False
    in_rf_config_list = False

    source_key_pattern = re.compile(rf'^\s*"{selected_source}"\s*:\s*{{')
    model_line_pattern = re.compile(r'^\s*"([^"]+)"\s*:')
    config_line_pattern = re.compile(r'^\s*"([^"]+\.py)"')

    for line in lines:
        stripped = line.strip()

        if '"model_source": [' in line:
            in_model_source = True
            modified.append(line)
            continue
        if in_model_source:
            if ']' in stripped:
                in_model_source = False
                modified.append(line)
                continue
            match = re.search(r'"([^"]+)"', stripped)
            if match:
                source = match.group(1)
                if source == selected_source:
                    modified.append(f'        "{source}",')
                else:
                    modified.append(f'        # "{source}",')
                continue

        if source_key_pattern.match(line):
            in_selected_source = True
            modified.append(line)
            continue

        if in_selected_source:
            if stripped.startswith("}"):
                in_selected_source = False
                modified.append(line)
                continue

            if '"models": {' in line and selected_source == "ultralytics":
                in_model_dict = True
                modified.append(line)
                continue

            if in_model_dict:
                if stripped.startswith("}"):
                    in_model_dict = False
                    modified.append(line)
                    continue
                match = model_line_pattern.match(line)
                if match:
                    model_name = match.group(1)
                    if model_name == selected_model:
                        modified.append(line.lstrip('#').strip())
                    else:
                        modified.append("#" + line if not line.strip().startswith("#") else line)
                    continue

            if selected_source == "hf_models_objectdetection":
                match = model_line_pattern.match(line)
                if match:
                    model_name = match.group(1)
                    if model_name == selected_model:
                        modified.append(line.lstrip('#').strip())
                    else:
                        modified.append("#" + line if not line.strip().startswith("#") else line)
                    continue

            if '"configs": [' in line and selected_source == "custom_codetr":
                in_config_list = True
                modified.append(line)
                continue

            if in_config_list:
                if ']' in stripped:
                    in_config_list = False
                    modified.append(line)
                    continue
                match = config_line_pattern.match(line)
                if match:
                    config_path = match.group(1)
                    if selected_model in config_path:
                        modified.append(line.lstrip('#').strip())
                    else:
                        modified.append("#" + line if not line.strip().startswith("#") else line)
                    continue

            if '"configs": [' in line and selected_source == "roboflow":
                in_rf_config_list = True
                modified.append(line)
                continue

            if in_rf_config_list:
                if ']' in stripped:
                    in_rf_config_list = False
                    modified.append(line)
                    continue

                match = re.search(r'^\s*#?\s*"([^"]+)"', stripped)
                if match:
                    config_name = match.group(1)
                    if config_name == selected_model:
                        indent = len(line) - len(line.lstrip())
                        uncommented = line.lstrip().lstrip('#').lstrip()
                        modified.append(' ' * indent + uncommented)
                    else:
                        if not stripped.startswith("#"):
                            indent = len(line) - len(line.lstrip())
                            modified.append(' ' * indent + '# ' + stripped)
                        else:
                            modified.append(line)
                    continue

        modified.append(line)

    # Validate against list_model_sources_and_models()
    valid = list_model_sources_and_models()
    if selected_model not in valid.get(selected_source, []):
        return (
            f"Invalid model '{selected_model}' for source '{selected_source}'. "
            f"Available: {valid.get(selected_source, [])}"
        )
    CONFIG_PATH.write_text('\n'.join(modified).rstrip('\n') + '\n')
    return f"Config updated to use `{selected_model}` from `{selected_source}`."


@mcp.tool()
def set_auto_labeling_hyperparams(
    mode: list = None,
    epochs: int = None,
    early_stop_patience: int = None,
    early_stop_threshold: float = None,
    learning_rate: float = None,
    weight_decay: float = None,
    max_grad_norm: float = None,
) -> str:
    """
    Modify only the specified hyperparameters for the auto_labeling workflow.
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    in_auto_labeling = False

    updated_keys = {}
    if mode is not None:
        updated_keys["\"mode\""] = str(mode)
    if epochs is not None:
        updated_keys["\"epochs\""] = str(epochs)
    if early_stop_patience is not None:
        updated_keys["\"early_stop_patience\""] = str(early_stop_patience)
    if early_stop_threshold is not None:
        updated_keys["\"early_stop_threshold\""] = str(early_stop_threshold)
    if learning_rate is not None:
        updated_keys["\"learning_rate\""] = str(learning_rate)
    if weight_decay is not None:
        updated_keys["\"weight_decay\""] = str(weight_decay)
    if max_grad_norm is not None:
        updated_keys["\"max_grad_norm\""] = str(max_grad_norm)

    for line in lines:
        if '"auto_labeling": {' in line:
            in_auto_labeling = True
            modified.append(line)
            continue

        if in_auto_labeling:
            stripped = line.strip()
            key = stripped.split(":")[0]
            if key in updated_keys:
                line = f"        {key}: {updated_keys[key]},"
            elif stripped.startswith("}"):
                in_auto_labeling = False

        modified.append(line)

    CONFIG_PATH.write_text("\n".join(modified).rstrip("\n") + "\n")
    return "Hyperparameters updated successfully."


@mcp.tool()
def list_model_sources_and_models() -> dict:
    """
    Lists valid model sources and models for auto_labeling.
    """
    return {
        "ultralytics": ["yolo11n", "yolo11x", "yolo12n", "yolo12x"],
        "hf_models_objectdetection": [
            "microsoft/conditional-detr-resnet-50",
            "Omnifact/conditional-detr-resnet-101-dc5",
            "facebook/detr-resnet-50",
            "facebook/detr-resnet-50-dc5",
            "facebook/detr-resnet-101",
            "facebook/detr-resnet-101-dc5",
            "facebook/deformable-detr-detic",
            "facebook/deformable-detr-box-supervised",
            "SenseTime/deformable-detr",
            "SenseTime/deformable-detr-with-box-refine",
            "jozhang97/deta-swin-large",
            "jozhang97/deta-swin-large-o365",
            "hustvl/yolos-base"
        ],
        "custom_codetr": [
            "co_deformable_detr_r50_1x_coco.py",
            "co_dino_5scale_vit_large_coco.py"
        ],
        "roboflow": ["rfdetr_nano", "rfdetr_small", "rfdetr_medium", "rfdetr_large", "rfdetr_xlarge", "rfdetr_2xlarge"]
    }


@mcp.tool()
async def run_auto_labeling() -> str:
    """Run auto_labeling workflow and return training or inference summary."""
    import os as _os

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(MAIN_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(MAIN_PATH.parent),
        )

        stdout_data, stderr_data = await process.communicate()
        output = stdout_data.decode("utf-8", errors="ignore") if stdout_data else ""
        error_output = stderr_data.decode("utf-8", errors="ignore") if stderr_data else ""

        combined_output = output + "\n" + error_output

        if "Evaluating detections..." in combined_output:
            res_lines = []
            capture = False

            for line in output.splitlines():
                if "              precision    recall  f1-score   support" in line:
                    capture = True
                    res_lines.append(line)
                    continue
                elif capture and line.startswith("You have launched a remote App on port 5151"):
                    break
                elif capture:
                    res_lines.append(line)

            report = "\n".join(res_lines).strip() if res_lines else "No inference results found."
        else:
            report = "Training completed successfully.\nThe model is ready to be tested using inference on the validation set."

        log_path = "output/logs/last_auto_labeling_log.txt"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(output)
            f.write("\n\n=== STDERR ===\n")
            f.write(error_output)
            f.write(f"\n\n=== EXIT CODE ===\n{process.returncode}")

        if process.returncode == 0:
            return (
                f"Auto-labeling workflow completed.\n\n"
                f"**Result Summary:**\n```\n{report}\n```\n"
                f"Full logs saved to `{log_path}`"
            )
        else:
            return (
                f"Auto-labeling failed with exit code {process.returncode}.\n"
                f"Error details:\n```\n{error_output[-3000:]}\n```\n"
                f"Full logs saved to `{log_path}`"
            )

    except Exception as e:
        return f"Error executing auto_labeling: {str(e)}"