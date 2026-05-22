import importlib
import re
import sys
import os
import ast as _ast
from pathlib import Path
from typing import List, Optional

import fiftyone as fo
import fiftyone.core.odm as _foodm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.dataset_loader import load_dataset
from mcptools import mcp

import config.config as _cc
from config.config import WORKFLOW_STATE_DEFAULT

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

    CONFIG_PATH.write_text('\n'.join(modified) + "\n")
    return f"Workflow selected: `{workflow_name}`."


@mcp.tool()
def set_selected_dataset(dataset_name: str) -> str:
    """
    Updates SELECTED_DATASET section in config.py with the given dataset name.
    Always uses the full dataset (n_samples = None).
    """
    _foodm.get_db_conn()
    available = fo.list_datasets()
    if dataset_name not in available:
        yaml_names = _extract_names_after_line(DEFAULT_DATASETS_YAML, 0)
        fixed_names = ["fisheye8k", "fisheye8k_mini", "mcity_fisheye_2000", "mcity_fisheye_2100"]
        all_known = set(available + yaml_names + fixed_names)
        if dataset_name not in all_known:
            return f"DATASET_NOT_FOUND: '{dataset_name}' does not exist."

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

    CONFIG_PATH.write_text("\n".join(modified) + "\n")

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

    CONFIG_PATH.write_text('\n'.join(modified) + "\n")
    return f"Switched to workflow: `{workflow_name}`."


@mcp.tool()
def reset_workflow_state() -> str:
    """
    Resets SELECTED_WORKFLOW, SELECTED_DATASET, and WORKFLOW_STATE in config.py,
    allowing the user to start a new workflow.
    """
    src = CONFIG_PATH.read_text()
    tree = _ast.parse(src)
    lines = src.splitlines()

    # Reset WORKFLOW_STATE using importlib.reload (consistent with chat_pipeline.py)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign):
            for target in node.targets:
                if isinstance(target, _ast.Name) and target.id == "WORKFLOW_STATE":
                    importlib.reload(_cc)
                    defaults = dict(WORKFLOW_STATE_DEFAULT)
                    start = node.lineno - 1
                    end = node.end_lineno
                    lines[start:end] = [f"WORKFLOW_STATE = {repr(defaults)}"]
                    src = "\n".join(lines)
                    break

    # Reset SELECTED_WORKFLOW and SELECTED_DATASET using line replacement
    result_lines = []
    src_lines = src.split("\n")
    i = 0
    while i < len(src_lines):
        line = src_lines[i]
        if line.strip().startswith("SELECTED_WORKFLOW"):
            result_lines.append('SELECTED_WORKFLOW = [""]')
        elif "SELECTED_DATASET = {" in line:
            result_lines.append('SELECTED_DATASET = {')
            result_lines.append('    "name": "",')
            result_lines.append('    "n_samples": None,')
            result_lines.append('    "custom_view": None,')
            result_lines.append('}')
            while i < len(src_lines) and not src_lines[i].strip() == "}":
                i += 1
        else:
            result_lines.append(line)
        i += 1

    CONFIG_PATH.write_text("\n".join(result_lines) + "\n")
    return "Workflow, dataset, and session state have been reset. You may now start a new workflow."



# Dataset listing helpers

FIXED_DATASETS: List[str] = [
    "fisheye8k",
    "fisheye8k_mini",
    "mcity_fisheye_2000",
    "mcity_fisheye_2100",
]

_NAME_LINE = re.compile(r'^\s*-\s*name:\s*["\']?([^"\']+)["\']?\s*$', re.IGNORECASE)


def _extract_names_after_line(yaml_path: Path, start_line_1_based: int) -> List[str]:
    if not yaml_path.exists():
        return []
    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
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
    - Then appends names found in datasets.yaml after line 52
    """
    dynamic_names = _extract_names_after_line(DEFAULT_DATASETS_YAML, 52)
    out: List[str] = []
    for n in FIXED_DATASETS:
        if n not in out:
            out.append(n)
    for n in dynamic_names:
        if n not in out:
            out.append(n)
    return out