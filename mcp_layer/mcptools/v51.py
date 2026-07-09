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
from host_utils import resolve_host

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
        # session_v51.py always binds the same fixed port (V51_PORT). A second
        # instance launched while one is already running fails to bind and dies
        # silently -- kill any existing session first so the new dataset actually
        # takes over the port instead of leaving a stale dataset displayed.
        subprocess.run(
            ["pkill", "-f", "session_v51.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)

        log_path = ROOT_DIR / "session_v51.log"
        with open(log_path, "a") as log_file:
            proc = subprocess.Popen(
                [sys.executable, str(ROOT_DIR / "session_v51.py"), target_dataset],
                stdout=log_file,
                stderr=log_file,
                cwd=str(ROOT_DIR),
            )
        logging.warning(f"[V51] Launched PID={proc.pid} for dataset '{target_dataset}'")
        host = resolve_host()
        return (
            f"Voxel51 session launched for dataset '{target_dataset}'. "
            f"Open your browser and go to: http://{host}:5151"
        )
    except Exception as e:
        return f"Failed to launch Voxel51 session: {e}\n{traceback.format_exc()}"