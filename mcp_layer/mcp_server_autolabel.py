from fastmcp import FastMCP
import subprocess
import re
import asyncio
from pathlib import Path

mcp = FastMCP("AutoLabeling Agent")

CONFIG_PATH = Path("/home/dataengine/Downloads/mcity_data_engine/config/config.py")
MAIN_PATH = Path("/home/dataengine/Downloads/mcity_data_engine/main.py")

@mcp.tool()
def configure_auto_labeling(selected_source: str, selected_model: str) -> str:
    """
    Enable the selected model source and model inside config.py for the auto_labeling workflow.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []

    in_model_source = False
    in_selected_source = False
    in_model_dict = False
    in_config_list = False

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

        modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified))
    return f"✅ Config updated to use `{selected_model}` from `{selected_source}`."

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

    # Only include keys that are not None
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

    CONFIG_PATH.write_text("\n".join(modified))
    return "✅ Hyperparameters updated successfully."


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
        ]
    }


@mcp.tool()
async def run_auto_labeling() -> str:
    """Run main.py, save full logs, and return only the evaluation summary."""
    process = await asyncio.create_subprocess_exec(
        "python", str(MAIN_PATH),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    output = ""
    eval_summary = ""
    capture = False

    while True:
        line = await process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="ignore")
        print(decoded, end="")  # Stream to terminal
        output += decoded

        # Start capturing after Voxel51 session launch
        if "Launching Voxel51 session for dataset" in decoded:
            capture = True
            eval_summary = ""
            continue

        if capture:
            # Stop capturing when the remote app launch message appears
            if "You have launched a remote App on port 5151" in decoded:
                capture = False
                continue
            eval_summary += decoded

    await process.wait()

    # Save full logs
    log_path = "output/logs/last_run_log.txt"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)

    if process.returncode == 0:
        return (
            f"✅ Auto-labeling finished successfully.\n\n"
            f"📊 **Final Evaluation Summary:**\n```\n{eval_summary}\n```\n"
            f"📄 Full logs saved to `{log_path}`"
        )
    else:
        return (
            f"❌ Auto-labeling failed with exit code {process.returncode}.\n"
            f"📄 Full logs saved to `{log_path}`\n"
            f"🛠 Please check the log file or terminal output for errors."
        )


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
