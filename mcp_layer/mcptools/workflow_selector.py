from mcptools import mcp
import subprocess
import re
import asyncio
from pathlib import Path
import os
import ast
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.dataset_loader import load_dataset
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "config" / "config.py"
MAIN_PATH = ROOT_DIR / "main.py"
DEFAULT_DATASETS_YAML = ROOT_DIR / "config" / "datasets.yaml"


@mcp.tool()
def select_workflow(workflow_name: str) -> str:
    """
    Updates the SELECTED_WORKFLOW in config.py to the given workflow name.
    Example: "auto_labeling", "class_mapping", "anomaly_detection", etc.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []
    pattern = re.compile(r'^SELECTED_WORKFLOW\s*=\s*\[.*\]')

    for line in lines:
        if pattern.match(line):
            modified.append(f'SELECTED_WORKFLOW = ["{workflow_name}"]')
        else:
            modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified))
    return f"Workflow selected: `{workflow_name}`."


@mcp.tool()
def set_selected_dataset(dataset_name: str) -> str:
    """
    Updates SELECTED_DATASET section in config.py with the given dataset name.
    Always uses the full dataset (n_samples = None).
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    in_dataset_block = False

    for line in lines:
        if "SELECTED_DATASET = {" in line:
            in_dataset_block = True
            modified.append(line)
            continue

        if in_dataset_block:
            if '"name":' in line:
                indent = " " * (len(line) - len(line.lstrip()))
                modified.append(f'{indent}"name": "{dataset_name}",')
                continue
            elif '"n_samples":' in line:
                indent = " " * (len(line) - len(line.lstrip()))
                modified.append(f'{indent}"n_samples": None,')
                continue
            elif '"custom_view":' in line:
                indent = " " * (len(line) - len(line.lstrip()))
                modified.append(f'{indent}"custom_view": None,')
                continue
            elif "}" in line:
                modified.append(line)
                in_dataset_block = False
                continue

        modified.append(line)

    CONFIG_PATH.write_text("\n".join(modified))

    if not dataset_name.startswith("custom"):
        dataset, dataset_info = load_dataset({"name": dataset_name, "n_samples": None, "custom_view": None})

    return f"Dataset set to `{dataset_name}`."

@mcp.tool()
def switch_workflow(workflow_name: str) -> str:
    """
    Switch to a new workflow. This updates SELECTED_WORKFLOW in config.py.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []
    pattern = re.compile(r'^SELECTED_WORKFLOW\s*=\s*\[.*\]')

    for line in lines:
        if pattern.match(line):
            modified.append(f'SELECTED_WORKFLOW = ["{workflow_name}"]')
        else:
            modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified))
    return f"Switched to workflow: `{workflow_name}`."

@mcp.tool()
def reset_workflow_state() -> str:
    """
    Resets SELECTED_WORKFLOW and SELECTED_DATASET in config.py,
    allowing the user to start a new workflow.
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []

    for line in lines:
        # Reset SELECTED_WORKFLOW
        if line.strip().startswith("SELECTED_WORKFLOW"):
            modified.append('SELECTED_WORKFLOW = [""]')
        # Reset SELECTED_DATASET block
        elif "SELECTED_DATASET = {" in line:
            modified.append('SELECTED_DATASET = {')
            modified.append('    "name": "",')
            modified.append('    "n_samples": None')
            modified.append('}')
            # Skip lines until end of block
            while not line.strip().endswith("}"):
                line = next(iter(lines), "")
            continue
        else:
            modified.append(line)

    CONFIG_PATH.write_text("\n".join(modified))
    return "Workflow and dataset have been reset. You may now start a new workflow."


# The four you always want to include (in this order)
FIXED_DATASETS: List[str] = [
    "fisheye8k",
    "fisheye8k_mini",
    "mcity_fisheye_2000",
    "mcity_fisheye_2100",
]

_NAME_LINE = re.compile(r'^\s*-\s*name:\s*["\']?([^"\']+)["\']?\s*$', re.IGNORECASE)

def _extract_names_after_line(yaml_path: Path, start_line_1_based: int) -> List[str]:
    """
    Return dataset names found after the given 1-based line number.
    We scan the file tail and pick lines like:  - name: <value>  (quoted or not)
    """
    if not yaml_path.exists():
        return []

    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Convert 1-based "after line N" to a 0-based slice starting at index N
    # e.g., after line 52 => start at lines[52], which is line 53 in 1-based terms
    tail = lines[start_line_1_based:]

    found: List[str] = []
    for raw in tail:
        m = _NAME_LINE.match(raw)
        if not m:
            continue
        name = m.group(1).strip()
        if name and name not in found:
            found.append(name)
    return found

@mcp.tool()
def list_datasets() -> List[str]:
    """
    Returns a list of dataset names.
    - The first four are always: fisheye8k, fisheye8k_mini, mcity_fisheye_2000, mcity_fisheye_2100
    - Then we append names found in datasets.yaml AFTER the given line number (1-based)
    """
    path = DEFAULT_DATASETS_YAML
    after_line = 52

    dynamic_names = _extract_names_after_line(path, after_line)

    # Build final list: fixed first (dedup), then dynamic (excluding duplicates)
    out: List[str] = []
    for n in FIXED_DATASETS:
        if n not in out:
            out.append(n)
    for n in dynamic_names:
        if n not in out:
            out.append(n)

    return out