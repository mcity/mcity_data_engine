import fiftyone as fo
import subprocess
import os
import signal
import re
import logging
import json as _json
from pathlib import Path
from mcptools import mcp
import time

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT_DIR / "config" / "config.py"
WORKFLOW_STATE_FILE = ROOT_DIR / "output" / "workflow_state.json"


@mcp.tool()
def launch_voxel51_session(dataset_name: str = "") -> str:
    target_dataset = dataset_name.strip()

    if not target_dataset:
        try:
            if WORKFLOW_STATE_FILE.exists():
                state = _json.loads(WORKFLOW_STATE_FILE.read_text())
                labeled = state.get("labeled_dataset_name", "")
                base = state.get("dataset_name", "")
                target_dataset = labeled if labeled else base
                logging.warning(f"[V51] Resolved from state file: '{target_dataset}'")
        except Exception as e:
            logging.warning(f"[V51] Error reading state file: {e}")

    if not target_dataset:
        try:
            config_text = CONFIG_PATH.read_text()
            m = re.search(r'SELECTED_DATASET\s*=\s*\{[^}]*"name":\s*"([^"]+)"', config_text)
            if m:
                target_dataset = m.group(1)
                logging.warning(f"[V51] Resolved from config.py: '{target_dataset}'")
        except Exception as e:
            logging.warning(f"[V51] Error reading config.py: {e}")

    if not target_dataset:
        return "Could not determine which dataset to visualize. Please provide a dataset name."

    try:
        dataset = fo.load_dataset(target_dataset)
        logging.warning(f"[V51] Direct load succeeded: '{target_dataset}', {len(dataset)} samples")
    except Exception as e:
        logging.warning(f"[V51] Direct load failed: {e} — retrying")
        for attempt in range(5):
            time.sleep(2)
            try:
                dataset = fo.load_dataset(target_dataset)
                logging.warning(f"[V51] Load succeeded on attempt {attempt + 2}")
                break
            except Exception as e2:
                logging.warning(f"[V51] Attempt {attempt + 2} failed: {e2}")
        else:
            return f"Dataset '{target_dataset}' could not be loaded after 10 seconds. Please try again."

    try:
        kill_result = subprocess.run(["lsof", "-ti", ":5151"], capture_output=True, text=True)
        for pid in kill_result.stdout.strip().split():
            try:
                os.kill(int(pid), signal.SIGTERM)
                logging.warning(f"[V51] Killed existing session PID={pid}")
            except Exception:
                pass
    except Exception:
        pass

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
        import traceback
        return f"Failed to launch Voxel51 session: {e}\n{traceback.format_exc()}"