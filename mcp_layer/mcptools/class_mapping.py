from mcptools import mcp
import re
import asyncio
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.dataset_loader import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "config" / "config.py"
MAIN_PATH = ROOT_DIR / "main.py"


@mcp.tool()
def list_class_mapping_models() -> list:
    """
    Lists available zero-shot classification models for class_mapping.
    """
    return [
        "Salesforce/blip2-itm-vit-g",
        "openai/clip-vit-large-patch14",
        "google/siglip-so400m-patch14-384",
        "kakaobrain/align-base",
        "BAAI/AltCLIP",
        "CIDAS/clipseg-rd64-refined"
    ]

@mcp.tool()
def configure_class_mapping_model(selected_model: str) -> str:
    """
    Enables only the selected model in class_mapping > hf_models_zeroshot_classification
    by commenting out all other models.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []
    in_class_mapping = False
    in_model_list = False

    for line in lines:
        stripped = line.strip()

        if '"class_mapping": {' in line:
            in_class_mapping = True
            modified.append(line)
            continue

        if in_class_mapping and '"hf_models_zeroshot_classification": [' in line:
            in_model_list = True
            modified.append(line)
            continue

        if in_model_list:
            if ']' in stripped:
                in_model_list = False
                modified.append(line)
                continue
            match = re.search(r'"([^"]+)"', stripped)
            if match:
                model_name = match.group(1)
                if model_name == selected_model:
                    modified.append(f'        "{model_name}",')
                else:
                    modified.append(f'        # "{model_name}",')
                continue

        if in_class_mapping and '}' in stripped:
            in_class_mapping = False

        modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified) + "\n")
    return f"Class mapping model updated to `{selected_model}`."


@mcp.tool()
def set_class_mapping_dataset_source(dataset_source: str) -> str:
    """
    Updates the `dataset_source` in the class_mapping workflow section of config.py.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []
    in_class_mapping = False

    for line in lines:
        stripped = line.strip()

        # Detect start of class_mapping block
        if '"class_mapping": {' in line:
            in_class_mapping = True
            modified.append(line)
            continue

        # If inside class_mapping block, look for dataset_source
        if in_class_mapping:
            if '"dataset_source":' in stripped:
                indent = " " * (len(line) - len(line.lstrip()))
                modified.append(f'{indent}"dataset_source": "{dataset_source}",')
                continue

            # Detect end of class_mapping block
            if "}" in stripped:
                in_class_mapping = False

        # Default case: keep the line
        modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified) + "\n")
    # Load and validate the dataset
    try:
        dataset, dataset_info = load_dataset({"name": dataset_source, "n_samples": None})
        label_classes = dataset.distinct("ground_truth.detections.label")
        return (
            f"Class Mapping dataset source set to `{dataset_source}`.\n"
            f"Available labels in the source dataset: {', '.join(label_classes)}"
        )

    except Exception as e:
        return f"Source dataset `{dataset_source}` was set in the config, but failed to load: {str(e)}"

@mcp.tool()
def set_class_mapping_dataset_target(dataset_target: str) -> str:
    """
    Updates the `dataset_target` in the class_mapping workflow section of config.py.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []
    in_class_mapping = False

    for line in lines:
        stripped = line.strip()

        # Detect start of class_mapping block
        if '"class_mapping": {' in line:
            in_class_mapping = True
            modified.append(line)
            continue

        # If inside class_mapping block, look for dataset_source
        if in_class_mapping:
            if '"dataset_target":' in stripped:
                indent = " " * (len(line) - len(line.lstrip()))
                modified.append(f'{indent}"dataset_target": "{dataset_target}",')
                continue

            # Detect end of class_mapping block
            if "}" in stripped:
                in_class_mapping = False

        # Default case: keep the line
        modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified) + "\n")
    dataset, dataset_info = load_dataset({"name": dataset_target, "n_samples": None})
    return f"Class Mapping dataset target set to `{dataset_target}`."

@mcp.tool()
def set_class_mapping_candidate_labels(candidate_labels: dict) -> str:
    """
    Updates the candidate_labels field in the class_mapping workflow section of config.py.
    Supports comment-preserving and minimal edits.
    Example input:
    {
        "Car": ["car", "van"],
        "Bike": ["motorbike/cycler"]
    }
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []
    in_class_mapping = False
    in_candidate_labels = False
    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        if '"class_mapping": {' in line:
            in_class_mapping = True
        if in_class_mapping and '"candidate_labels": {' in line:
            start_idx = i
        if start_idx is not None and in_class_mapping:
            if "}," in line or "}" in line.strip():
                end_idx = i + 1
                break

    if start_idx is None or end_idx is None:
        return "Failed to locate candidate_labels block."

    indent = " " * (len(lines[start_idx]) - len(lines[start_idx].lstrip()))
    new_block = [f'{indent}"candidate_labels": {{']
    for key, value_list in candidate_labels.items():
        value_str = ", ".join(f'"{v}"' for v in value_list)
        new_block.append(f'{indent}    "{key}": [{value_str}],')
    new_block.append(f'{indent}}},')

    updated_lines = lines[:start_idx] + new_block + lines[end_idx:]
    CONFIG_PATH.write_text("\n".join(updated_lines) + "\n")

    return f"Candidate labels updated successfully:\n{candidate_labels}"


@mcp.tool()
async def run_class_mapping() -> str:
    """Run main.py for class_mapping with improved subprocess handling."""

    try:
        # Use communicate() instead of readline loop to prevent deadlocks
        process = await asyncio.create_subprocess_exec(
            "python", "-u", str(MAIN_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(MAIN_PATH.parent),
        )

        # Get all output at once to prevent buffer issues
        stdout_data, stderr_data = await process.communicate()

        # Decode output
        output = stdout_data.decode("utf-8", errors="ignore") if stdout_data else ""
        error_output = stderr_data.decode("utf-8", errors="ignore") if stderr_data else ""

        # Print to terminal for debugging
        if output:
            print("STDOUT:", output)
        if error_output:
            print("STDERR:", error_output)

        # Extract tag summary from output
        tag_lines = []
        capture = False

        for line in error_output.splitlines():

            if "Tag Addition Results (Target Dataset Tags):" in line:
                capture = True
                tag_lines.append(line)  # Include the triggering line
                continue  # Avoid falling through to the elif
            elif capture and line.startswith("Zero Shot Classification model"):
                break
            elif capture:
                tag_lines.append(line)

        print(tag_lines)

        tag_summary = "\n".join(tag_lines).strip() if tag_lines else "No tag addition results found."

        print(tag_summary)


        # Save full logs
        log_path = "output/logs/last_class_mapping_log.txt"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(output)
            f.write("\n\n=== STDERR ===\n")
            f.write(error_output)
            f.write(f"\n\n=== EXIT CODE ===\n{process.returncode}")

        if process.returncode == 0:
            return (
                f"Class Mapping completed successfully.\n\n"
                f"**Tag Addition Summary:**\n```\n{tag_summary}\n```\n"
                f"Full logs saved to `{log_path}`"
            )
        else:
            return (
                f"Class Mapping failed with exit code {process.returncode}.\n"
                f"Error details: {error_output[-5000:]}\n"
                f"Full logs saved to `{log_path}`"
            )

    except Exception as e:
        return f"Error executing class mapping: {str(e)}"