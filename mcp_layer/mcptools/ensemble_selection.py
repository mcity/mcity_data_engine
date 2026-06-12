from mcptools import mcp
import asyncio
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "config" / "config.py"
MAIN_PATH = ROOT_DIR / "main.py"


@mcp.tool()
def set_ensemble_selection_parameters(
    agreement_threshold: int = None,
    iou_threshold: float = None,
    max_bbox_size: float = None,
) -> str:
    """
    Modify only the specified hyperparameters for the auto_labeling workflow.
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    in_ensemble_selection = False

    # Only include keys that are not None
    updated_keys = {}

    if agreement_threshold is not None:
        updated_keys["\"agreement_threshold\""] = str(agreement_threshold)
    if iou_threshold is not None:
        updated_keys["\"iou_threshold\""] = str(iou_threshold)
    if max_bbox_size is not None:
        updated_keys["\"max_bbox_size\""] = str(max_bbox_size)

    for line in lines:
        if '"ensemble_selection": {' in line:
            in_ensemble_selection = True
            modified.append(line)
            continue

        if in_ensemble_selection:
            stripped = line.strip()
            key = stripped.split(":")[0]
            if key in updated_keys:
                line = f"        {key}: {updated_keys[key]},"
            elif stripped.startswith("}"):
                in_ensemble_selection = False

        modified.append(line)

    CONFIG_PATH.write_text("\n".join(modified).rstrip("\n") + "\n")
    return "Ensemble Selection Parameters updated successfully."

@mcp.tool()
def set_ensemble_selection_classes(positive_classes: list) -> str:
    """
    Sets the positive_classes field in the ensemble_selection section of config.py.
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []
    in_ensemble_selection = False
    in_positive_classes = False

    for line in lines:
        stripped = line.strip()

        if '"ensemble_selection": {' in line:
            in_ensemble_selection = True
            modified.append(line)
            continue

        if in_ensemble_selection and stripped.startswith('"positive_classes": ['):
            in_positive_classes = True
            indent = line[:line.index('"')]
            modified.append(f'{indent}"positive_classes": [')
            for cls in positive_classes:
                modified.append(f'{indent}    "{cls}",')
            modified.append(f'{indent}],')
            continue

        if in_positive_classes:
            if stripped.endswith('],') or stripped == ']':
                in_positive_classes = False
            continue

        if in_ensemble_selection and stripped == '},':
            in_ensemble_selection = False

        modified.append(line)

    CONFIG_PATH.write_text("\n".join(modified).rstrip("\n") + "\n")
    return f"Set `positive_classes` to: {positive_classes}"

@mcp.tool()
async def run_ensemble_selection() -> str:
    """
    Run ensemble_selection workflow and guide user to interpret results using Voxel51.
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

        log_path = "output/logs/last_ensemble_selection_log.txt"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(output)
            f.write("\n\n=== STDERR ===\n")
            f.write(error_output)
            f.write(f"\n\n=== EXIT CODE ===\n{process.returncode}")

        if process.returncode == 0:
            return (
                "Ensemble Selection completed successfully. The final predictions now reflect consensus across multiple zero-shot models.\n"
            )
        else:
            return (
                f"Ensemble selection failed with exit code {process.returncode}.\n"
                f"Error details: {error_output[-5000:]}\n"
                f"Full logs saved to `{log_path}`"
            )

    except Exception as e:
        return f"Error executing ensemble selection: {str(e)}"