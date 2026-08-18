import asyncio
import importlib
import logging
import re
import shutil
import sys
import os
import time
import ast as _ast
from pathlib import Path
from typing import List
from unittest import mock

import fiftyone as fo
import fiftyone.core.odm as _foodm
from fastmcp import Context
from ruamel.yaml import YAML
from tqdm import tqdm as _tqdm_base

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))       # mcp_layer/
from utils.dataset_loader import load_dataset
from mcptools import mcp
import config.config as _cc

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "config.py"
DEFAULT_DATASETS_YAML = ROOT_DIR / "config" / "datasets.yaml"
# Mirrors UPLOAD_ROOT in mcptools/data_ingest.py — kept as a copy so that
# importing this module does not pull in the ingest tools.
UPLOAD_ROOT = Path(os.environ.get("MDE_UPLOAD_ROOT", ROOT_DIR / "mde_uploads")).resolve()


async def _load_dataset_with_progress(dataset_name: str, ctx) -> None:
    """Runs load_dataset() in a worker thread, relaying ~30s progress updates
    over the MCP log channel if the download goes through FiftyOne's HF loader.

    FiftyOne has no public progress callback, so this patches tqdm.tqdm itself
    for the duration of the call -- fiftyone.utils.huggingface imports tqdm
    locally per-call rather than at module level, so that's the only patch
    target that actually takes effect. No-op for datasets that don't hit that
    code path.
    """
    loop = asyncio.get_running_loop()
    last_emit = 0.0

    class _ProgressTqdm(_tqdm_base):
        def update(self, n=1):
            nonlocal last_emit
            result = super().update(n)
            if ctx and self.total:
                now = time.time()
                is_done = self.n >= self.total
                if now - last_emit >= 30 or is_done:
                    last_emit = now
                    pct = self.n / self.total * 100
                    msg = f"Downloading {dataset_name}: {self.n}/{self.total} ({pct:.0f}%)"
                    rate = self.format_dict.get("rate")
                    if rate and not is_done:
                        remaining = (self.total - self.n) / rate
                        msg += f", ~{_tqdm_base.format_interval(remaining)} remaining"
                    asyncio.run_coroutine_threadsafe(ctx.log(msg), loop)
            return result

    with mock.patch("tqdm.tqdm", _ProgressTqdm):
        await asyncio.to_thread(
            load_dataset,
            {"name": dataset_name, "n_samples": None, "custom_view": None},
        )


@mcp.tool()
def select_workflow(workflow_name: str) -> str:
    """
    Updates SELECTED_WORKFLOW in config.py to the given workflow name.
    Example: "auto_labeling", "class_mapping", "anomaly_detection", etc.
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    pattern = re.compile(r"^SELECTED_WORKFLOW\s*=\s*\[.*\]")
    for line in lines:
        if pattern.match(line):
            modified.append(f'SELECTED_WORKFLOW = ["{workflow_name}"]')
        else:
            modified.append(line)
    CONFIG_PATH.write_text("\n".join(modified).rstrip("\n") + "\n")
    return f"Workflow selected: `{workflow_name}`."


def _prune_stale_custom_entries() -> None:
    """Drop datasets.yaml entries for load_custom_dataset datasets no longer in
    FiftyOne (e.g. after manual cleanup) -- these have no lazy-load fallback, so
    a stale entry would otherwise pass selection and fail deep inside the run.
    """
    try:
        _foodm.get_db_conn()
        existing = set(fo.list_datasets())
    except Exception as e:
        logging.warning(f"[DATASETS] Could not verify against FiftyOne, skipping prune: {e}")
        return

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(sequence=4, offset=2)
    try:
        with open(DEFAULT_DATASETS_YAML, "r") as f:
            data = yaml.load(f)
        entries = data.get("datasets", []) or []
        stale = [
            d for d in entries
            if str(d.get("loader_fct", "")) == "load_custom_dataset" and str(d.get("name", "")) not in existing
        ]
        if not stale:
            return
        data["datasets"] = [d for d in entries if d not in stale]
        with open(DEFAULT_DATASETS_YAML, "w") as f:
            yaml.dump(data, f)
        logging.warning(f"[DATASETS] Pruned stale entries no longer in FiftyOne: {[str(d.get('name')) for d in stale]}")
    except Exception as e:
        logging.warning(f"[DATASETS] Failed to prune stale entries: {e}")


@mcp.tool()
async def set_selected_dataset(dataset_name: str, ctx: Context = None) -> str:
    """
    Updates SELECTED_DATASET section in config.py with the given dataset name.
    Always uses the full dataset (n_samples = None).
    """
    _prune_stale_custom_entries()
    _foodm.get_db_conn()
    available = fo.list_datasets()
    if dataset_name not in available:
        all_known = set(available) | set(all_dataset_names())
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

    CONFIG_PATH.write_text("\n".join(modified).rstrip("\n") + "\n")

    if not dataset_name.startswith("custom"):
        await _load_dataset_with_progress(dataset_name, ctx)

    return f"Dataset set to `{dataset_name}`."

def _yaml_entry_exists(dataset_name: str) -> bool:
    """True if datasets.yaml holds an entry with this name."""
    return any(str(d["name"]) == dataset_name for d in _yaml_dataset_entries())


def _remove_dataset_entry(dataset_name: str) -> bool:
    """Delete the datasets.yaml entry for `dataset_name`. Returns True if one was removed.

    _prune_stale_custom_entries only drops load_custom_dataset entries, so
    entries with any other loader_fct must be removed here.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(sequence=4, offset=2)
    try:
        with open(DEFAULT_DATASETS_YAML, "r") as f:
            data = yaml.load(f)
        entries = data.get("datasets", []) or []
        kept = [d for d in entries if str(d.get("name", "")) != dataset_name]
        if len(kept) == len(entries):
            return False
        data["datasets"] = kept
        with open(DEFAULT_DATASETS_YAML, "w") as f:
            yaml.dump(data, f)
        logging.info(f"[DATASETS] Removed datasets.yaml entry: {dataset_name}")
        return True
    except Exception as e:
        logging.warning(f"[DATASETS] Failed to remove entry '{dataset_name}': {e}")
        return False


def _upload_dirs_for_dataset(dataset_name: str) -> list[Path]:
    """Return the mde_uploads/job_* directories that hold this dataset's media.

    Paths outside UPLOAD_ROOT are ignored, so a dataset that points at media
    elsewhere on disk (HF cache, a shared drive) never puts that media at risk.
    Call this BEFORE the FiftyOne dataset is deleted.
    """
    dirs: dict[str, Path] = {}
    try:
        dataset = fo.load_dataset(dataset_name)
        for filepath in dataset.values("filepath"):
            if not filepath:
                continue
            try:
                rel = Path(filepath).resolve().relative_to(UPLOAD_ROOT)
            except ValueError:
                continue  # media lives outside the upload area
            if rel.parts:
                job_dir = UPLOAD_ROOT / rel.parts[0]
                dirs[str(job_dir)] = job_dir
    except Exception as e:
        logging.warning(f"[DATASETS] Could not resolve upload dirs for '{dataset_name}': {e}")
    return list(dirs.values())


@mcp.tool()
async def delete_dataset(dataset_name: str, delete_files: bool = False, ctx: Context = None) -> str:
    """
    Permanently deletes a dataset: the FiftyOne dataset and its datasets.yaml entry.

    Set delete_files=True to also erase the uploaded media under mde_uploads/.
    This cannot be undone. Only call it after the user explicitly confirms.
    The default datasets that ship in config/datasets.yaml cannot be deleted;
    only ingested datasets can.
    """
    dataset_name = (dataset_name or "").strip()
    if not dataset_name:
        return "DATASET_NOT_FOUND: no dataset name was given."

    if dataset_name in default_dataset_names():
        return (
            f"PROTECTED_DATASET: '{dataset_name}' is a default dataset and cannot be deleted."
        )

    _foodm.get_db_conn()
    in_fiftyone = dataset_name in fo.list_datasets()
    in_yaml = _yaml_entry_exists(dataset_name)
    if not in_fiftyone and not in_yaml:
        return f"DATASET_NOT_FOUND: '{dataset_name}' does not exist."

    # Resolve the media directories first — the sample filepaths are gone
    # once the FiftyOne dataset is deleted.
    upload_dirs = _upload_dirs_for_dataset(dataset_name) if (delete_files and in_fiftyone) else []

    done: list[str] = []
    if in_fiftyone:
        if ctx:
            await ctx.log(f"Deleting FiftyOne dataset {dataset_name}...")
        await asyncio.to_thread(fo.delete_dataset, dataset_name)
        done.append("removed from FiftyOne")

    if in_yaml and _remove_dataset_entry(dataset_name):
        done.append("removed from datasets.yaml")

    # config.py must not keep pointing at a dataset that no longer exists.
    importlib.reload(_cc)
    if _cc.SELECTED_DATASET.get("name", "") == dataset_name:
        lines = CONFIG_PATH.read_text().split("\n")
        lines = _reset_selected_dataset_block(lines)
        CONFIG_PATH.write_text("\n".join(lines).rstrip("\n") + "\n")
        importlib.reload(_cc)
        done.append("cleared SELECTED_DATASET")

    for job_dir in upload_dirs:
        try:
            shutil.rmtree(job_dir)
            logging.info(f"[DATASETS] Deleted upload directory: {job_dir}")
            done.append(f"deleted files in {job_dir.name}")
        except Exception as e:
            logging.warning(f"[DATASETS] Failed to delete {job_dir}: {e}")
            done.append(f"FAILED to delete files in {job_dir.name}: {e}")

    return f"Dataset `{dataset_name}` deleted ({'; '.join(done)})."


def _reset_selected_dataset_block(lines: list[str]) -> list[str]:
    """Return `lines` with the SELECTED_DATASET block reset to empty defaults."""
    result_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "SELECTED_DATASET = {" in line:
            result_lines.append("SELECTED_DATASET = {")
            result_lines.append('    "name": "",')
            result_lines.append('    "n_samples": None,')
            result_lines.append('    "custom_view": None,')
            result_lines.append("}")
            while i < len(lines) and lines[i].strip() != "}":
                i += 1
        else:
            result_lines.append(line)
        i += 1
    return result_lines


@mcp.tool()
def switch_workflow(workflow_name: str) -> str:
    """
    Switch to a new workflow. Updates SELECTED_WORKFLOW and clears SELECTED_DATASET
    in config.py — every caller treats this as a full reset (WorkflowState is reset
    before this tool is ever invoked), so config.py must not keep pointing at the
    previous workflow's dataset.
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    pattern = re.compile(r"^SELECTED_WORKFLOW\s*=\s*\[.*\]")
    for line in lines:
        if pattern.match(line):
            modified.append(f'SELECTED_WORKFLOW = ["{workflow_name}"]')
        else:
            modified.append(line)
    modified = _reset_selected_dataset_block(modified)
    CONFIG_PATH.write_text("\n".join(modified).rstrip("\n") + "\n")
    return f"Switched to workflow: `{workflow_name}`."


@mcp.tool()
def reset_workflow_state() -> str:
    """
    Resets SELECTED_WORKFLOW, SELECTED_DATASET, and WORKFLOW_STATE in config.py,
    allowing the user to start a new workflow.
    """
    # Reset WORKFLOW_STATE via WorkflowState — typed, validated, correct schema
    from validate_workflow_state import WorkflowState
    WorkflowState.reset()

    # Reset SELECTED_WORKFLOW and SELECTED_DATASET via line replacement
    src_lines = CONFIG_PATH.read_text().split("\n")
    result_lines = []
    for line in src_lines:
        if line.strip().startswith("SELECTED_WORKFLOW"):
            result_lines.append('SELECTED_WORKFLOW = [""]')
        else:
            result_lines.append(line)
    result_lines = _reset_selected_dataset_block(result_lines)

    CONFIG_PATH.write_text("\n".join(result_lines).rstrip("\n") + "\n")
    return "Workflow, dataset, and session state have been reset. You may now start a new workflow."


# Dataset listing

# Loader used by every ingested dataset (see workflows/data_ingest.py:
# append_dataset_entry) and by the custom_dataset* placeholder slots. Any other
# loader means the entry ships with the repo, i.e. it is a default dataset.
INGESTED_LOADER_FCT = "load_custom_dataset"

_NAME_LINE = re.compile(r"^\s*-\s*name:\s*[\"']?([^\"']+)[\"']?\s*$", re.IGNORECASE)


def _extract_names(yaml_path: Path) -> List[str]:
    """Line-based name extraction — fallback for when the YAML does not parse."""
    if not yaml_path.exists():
        return []
    found: List[str] = []
    for raw in yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _NAME_LINE.match(raw)
        if not m:
            continue
        name = m.group(1).strip()
        if name and name not in found:
            found.append(name)
    return found


def _yaml_dataset_entries() -> List[dict]:
    """All datasets.yaml entries, in file order. Empty list if the file is unreadable."""
    try:
        with open(DEFAULT_DATASETS_YAML, "r") as f:
            data = YAML().load(f)
        return [d for d in (data.get("datasets", []) or []) if d.get("name")]
    except Exception as e:
        logging.warning(f"[DATASETS] Could not read {DEFAULT_DATASETS_YAML}: {e}")
        return []


def default_dataset_names() -> List[str]:
    """Names of the datasets that ship with the repo, in datasets.yaml order.

    These are protected from delete_dataset -- removing one would edit a
    tracked config file rather than clean up something the user created.
    """
    return [
        str(d["name"])
        for d in _yaml_dataset_entries()
        if str(d.get("loader_fct", "")) != INGESTED_LOADER_FCT
    ]


def all_dataset_names() -> List[str]:
    """Every name in datasets.yaml, defaults first, then ingested datasets."""
    entries = _yaml_dataset_entries()
    if not entries:
        return _extract_names(DEFAULT_DATASETS_YAML)
    defaults = default_dataset_names()
    out = list(defaults)
    for d in entries:
        name = str(d["name"])
        if name not in out:
            out.append(name)
    return out


@mcp.tool()
def list_datasets() -> List[str]:
    """
    Returns all available dataset names from datasets.yaml.
    The default datasets appear first, followed by ingested datasets.
    """
    _prune_stale_custom_entries()
    return all_dataset_names()