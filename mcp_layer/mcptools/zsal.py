from mcptools import mcp
import re
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Define config and dataset paths relative to the project root
CONFIG_PATH = ROOT_DIR / "config" / "config.py"
MAIN_PATH = ROOT_DIR / "main.py"

@mcp.tool()
def list_zsal() -> list:
    """
    Lists all available Hugging Face zero-shot object detection models
    under auto_labeling_zero_shot > hf_models_zeroshot_objectdetection.
    """
    return [
        "omlab/omdet-turbo-swin-tiny-hf",
        "IDEA-Research/grounding-dino-tiny",
        "google/owlvit-large-patch14",
        "google/owlv2-base-patch16-finetuned",
        "google/owlv2-large-patch14-ensemble"
    ]


@mcp.tool()
def configure_auto_labeling_zero_shot_models(selected_models: list) -> str:
    """
    Enables only the selected models in auto_labeling_zero_shot by commenting out all other models,
    preserving full lines including inline dictionaries.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []

    in_alzs = False
    in_model_block = False

    for line in lines:
        stripped = line.strip()

        # Enter/exit auto_labeling_zero_shot
        if '"auto_labeling_zero_shot": {' in line:
            in_alzs = True
            modified.append(line)
            continue
        if in_alzs and stripped == '},':
            in_alzs = False
            modified.append(line)
            continue

        # Enter/exit hf_models_zeroshot_objectdetection
        if in_alzs and '"hf_models_zeroshot_objectdetection": {' in line:
            in_model_block = True
            modified.append(line)
            continue
        if in_model_block and stripped == '},':
            in_model_block = False
            modified.append(line)
            continue

        # Process model lines
        if in_model_block:
            match = re.match(r'(\s*)(#\s*)?"([^"]+)":\s*{.*},?', line)
            if match:
                indent, comment, model_name = match.groups()
                is_selected = model_name in selected_models
                if is_selected:
                    uncommented = re.sub(r'^(\s*)#\s*', r'\1', line)
                    modified.append(uncommented)
                else:
                    if not line.lstrip().startswith("#"):
                        modified.append(f'{indent}# {line.lstrip()}')
                    else:
                        modified.append(line)
                continue

        # Default case
        modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified) + "\n")
    return f"Updated models: {', '.join(selected_models)}"

@mcp.tool()
def set_auto_labeling_zero_shot_threshold(threshold: float) -> str:
    """
    Sets the detection_threshold under auto_labeling_zero_shot to the specified float value.
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    in_auto_labeling_zero_shot = False

    for line in lines:
        stripped = line.strip()

        if '"auto_labeling_zero_shot": {' in line:
            in_auto_labeling_zero_shot = True
            modified.append(line)
            continue

        if in_auto_labeling_zero_shot and stripped.startswith('"detection_threshold":'):
            indent = line[:line.index('"')]
            modified.append(f'{indent}"detection_threshold": {threshold},')
            continue

        if in_auto_labeling_zero_shot and stripped == '},':
            in_auto_labeling_zero_shot = False

        modified.append(line)

    CONFIG_PATH.write_text("\n".join(modified) + "\n")
    return f"Detection threshold set to `{threshold}` in `auto_labeling_zero_shot`."

@mcp.tool()
def set_auto_labeling_zero_shot_classes(object_classes: list) -> str:
    """
    Replaces the object_classes list in auto_labeling_zero_shot with the provided list.
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    in_auto_labeling_zero_shot = False
    in_object_classes = False

    for line in lines:
        stripped = line.strip()

        if '"auto_labeling_zero_shot": {' in line:
            in_auto_labeling_zero_shot = True
            modified.append(line)
            continue

        # Begin replacing object_classes
        if in_auto_labeling_zero_shot and stripped.startswith('"object_classes": ['):
            in_object_classes = True
            indent = line[:line.index('"')]
            modified.append(f'{indent}"object_classes": [')
            for cls in object_classes:
                modified.append(f'{indent}    "{cls}",')
            modified.append(f'{indent}],')
            continue

        # Skip old object_classes lines
        if in_object_classes:
            if stripped.endswith('],') or stripped == ']':
                in_object_classes = False
            continue

        # End auto_labeling_zero_shot block
        if in_auto_labeling_zero_shot and stripped == '},':
            in_auto_labeling_zero_shot = False

        modified.append(line)

    CONFIG_PATH.write_text("\n".join(modified) + "\n")
    return f"Set `object_classes` to: {object_classes}"


@mcp.tool()
async def run_zero_shot_auto_labeling() -> str:
    """
    Run auto_labeling_zero_shot workflow and guide user to interpret results using Voxel51.
    """

    try:
        process = await asyncio.create_subprocess_exec(
            "python", "-u", str(MAIN_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(MAIN_PATH.parent),
        )
        stdout_data, stderr_data = await process.communicate()
        output = stdout_data.decode("utf-8", errors="ignore") if stdout_data else ""
        error_output = stderr_data.decode("utf-8", errors="ignore") if stderr_data else ""

        log_path = "output/logs/last_auto_labeling_zero_shot_log.txt"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(output)
            f.write("\n\n=== STDERR ===\n")
            f.write(error_output)
            f.write(f"\n\n=== EXIT CODE ===\n{process.returncode}")

        if process.returncode == 0:
            return (
                "Zero Shot Auto-Labeling completed successfully.\n"
            )
        else:
            return (
                f"Zero Shot Auto-Labeling failed with exit code {process.returncode}.\n"
                f"Error details: {error_output[-5000:]}\n"
                f"Full logs saved to `{log_path}`"
            )

    except Exception as e:
        return f"Error executing zero-shot auto labeling: {str(e)}"