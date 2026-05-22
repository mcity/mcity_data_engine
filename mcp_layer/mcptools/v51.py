import sys
import time
import logging
import importlib
import subprocess
import traceback
from pathlib import Path

import fiftyone as fo
from mcptools import mcp
import fiftyone.core.odm as _foodm
import config.config as _cc

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "config.py"


def _read_config_state() -> dict:
    try:
        importlib.reload(_cc)
        return dict(_cc.WORKFLOW_STATE)
    except Exception as e:
        logging.warning(f"[V51] Error reading WORKFLOW_STATE: {e}")
        return {}


@mcp.tool()
def launch_voxel51_session(dataset_name: str = "") -> str:
    target_dataset = dataset_name.strip()

    if not target_dataset:
        try:
            state = _read_config_state()
            labeled = state.get("labeled_dataset_name", "")
            base = state.get("dataset_name", "")
            target_dataset = labeled if labeled else base
            logging.warning(f"[V51] Resolved from config state: labeled='{labeled}' base='{base}' -> using '{target_dataset}'")
        except Exception as e:
            logging.warning(f"[V51] Error reading config state: {e}")

    if not target_dataset:
        return "Could not determine which dataset to visualize. Please provide a dataset name."

    try:
        _foodm.get_db_conn()
        dataset = fo.load_dataset(target_dataset)
        logging.warning(f"[V51] Direct load succeeded: '{target_dataset}', {len(dataset)} samples")
    except Exception as e:
        logging.warning(f"[V51] Direct load failed: {e} — retrying")
        for attempt in range(5):
            time.sleep(2)
            try:
                _foodm.get_db_conn()
                dataset = fo.load_dataset(target_dataset)
                logging.warning(f"[V51] Load succeeded on attempt {attempt + 2}")
                break
            except Exception as e2:
                logging.warning(f"[V51] Attempt {attempt + 2} failed: {e2}")
        else:
            return f"Dataset '{target_dataset}' could not be loaded after 10 seconds. Please try again."

    try:
        proc = subprocess.Popen(
            ["python", str(ROOT_DIR / "session_v51.py"), target_dataset],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(ROOT_DIR),
        )
        logging.warning(f"[V51] Launched PID={proc.pid} for dataset '{target_dataset}'")
        return (
            f"Voxel51 session launched for dataset '{target_dataset}'. "
            f"Open your browser and go to: http://localhost:5151"
        )
    except Exception as e:
        return f"Failed to launch Voxel51 session: {e}\n{traceback.format_exc()}"