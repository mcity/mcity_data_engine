from mcptools import mcp
import subprocess
import re
import asyncio
from pathlib import Path
import os
import ast

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "config" / "config.py"
MAIN_PATH = ROOT_DIR / "main.py"


@mcp.tool()
def launch_voxel51_session() -> str:
    """
    Launches the Voxel51 session by running session_v51.py asynchronously.
    """
    import subprocess

    try:
        subprocess.Popen(
            ['python', 'session_v51.py'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return "Voxel51 session launched successfully in the background."
    except Exception as e:
        return f"Failed to launch Voxel51 session.\nError: {str(e)}"
