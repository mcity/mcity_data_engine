from mcptools import mcp  # shared instance from __init__.py
import re
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # two levels up from anomaly_detection.py

CONFIG_PATH = ROOT_DIR / "config" / "config.py"
MAIN_PATH = ROOT_DIR / "main.py"

@mcp.tool()
def list_anomaly_detection_models() -> list:
    """
    Lists available Anomalib image models for anomaly detection.
    """
    return [
        "Padim",
        "EfficientAd",
        "Draem",
        "Cfa"
    ]

@mcp.tool()
def configure_anomaly_detection_model(selected_model: str) -> str:
    """
    Enables only the selected model in anomaly_detection > anomalib_image_models
    by commenting out all other models.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []

    in_anomaly_block = False
    in_model_block = False
    found = False

    for line in lines:
        stripped = line.strip()

        # Detect start of anomaly_detection block
        if '"anomaly_detection": {' in line:
            in_anomaly_block = True
            modified.append(line)
            continue

        # Detect start of anomalib_image_models dict
        if in_anomaly_block and '"anomalib_image_models": {' in line:
            in_model_block = True
            modified.append(line)
            continue

        # Detect end of anomalib_image_models dict
        if in_model_block:
            if stripped.startswith('}'):
                in_model_block = False
                modified.append(line)
                continue

            # Match lines like "Padim": {},
            match = re.search(r'"([^"]+)"\s*:', stripped)
            if match:
                model_name = match.group(1)
                if model_name.lower() == selected_model.lower():
                    modified.append(f'            "{model_name}": {{}},')
                    found = True
                else:
                    # comment out line if not already commented
                    if not line.strip().startswith('#'):
                        modified.append(f'            # "{model_name}": {{}},')
                    else:
                        modified.append(line)
                continue

        # Exit anomaly_detection section
        if in_anomaly_block and stripped == '}':
            in_anomaly_block = False

        modified.append(line)

    if not found:
        return f"Model `{selected_model}` not found in `anomalib_image_models`. Valid options: Padim, EfficientAd, Draem, Cfa."

    CONFIG_PATH.write_text('\n'.join(modified).rstrip('\n') + '\n')
    return f"Anomaly Detection model updated to `{selected_model}`."


@mcp.tool()
def set_anomaly_detection_data_source(location: str, rare_class: str) -> str:
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []

    in_anomaly = False
    in_data_prep = False
    in_dataset = False

    for line in lines:
        stripped = line.strip()

        if '"anomaly_detection": {' in line:
            in_anomaly = True
            modified.append(line)
            continue

        if in_anomaly and '"data_preparation": {' in stripped:
            in_data_prep = True
            modified.append(line)
            continue

        if in_data_prep and re.match(r'^\s*"\w+"\s*:\s*{', line):
            in_dataset = True
            modified.append(line)
            continue

        if in_dataset:
            if '"location":' in stripped:
                indent = " " * (len(line) - len(stripped))
                modified.append(f'{indent}"location": "{location}",')
                continue
            elif '"rare_class":' in stripped:
                indent = " " * (len(line) - len(stripped))
                modified.append(f'{indent}"rare_class": "{rare_class}",')
                continue
            elif stripped == '}' or stripped.endswith('},'):
                in_dataset = False
                modified.append(line)
                continue

        if in_data_prep and stripped == '}':
            in_data_prep = False
            modified.append(line)
            continue

        if in_anomaly and stripped == '}':
            in_anomaly = False

        modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified).rstrip('\n') + '\n')
    return f"Set anomaly_detection data_preparation to location: `{location}`, rare_class: `{rare_class}`"

@mcp.tool()
def set_anomaly_detection_hyperparams(
    mode: list = None,
    epochs: int = None,
    early_stop_patience: int = None,
) -> str:
    """
    Updates the 'mode', 'epochs', and 'early_stop_patience' values inside the anomaly_detection workflow section of config.py.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []
    in_anomaly_block = False

    updated = {
        '"mode"': str(mode) if mode else None,
        '"epochs"': str(epochs) if epochs is not None else None,
        '"early_stop_patience"': str(early_stop_patience) if early_stop_patience is not None else None
    }

    for line in lines:
        if '"anomaly_detection": {' in line:
            in_anomaly_block = True
            modified.append(line)
            continue

        if in_anomaly_block:
            key = line.strip().split(":")[0]
            if key in updated and updated[key] is not None:
                indent = " " * (len(line) - len(line.lstrip()))
                modified.append(f'{indent}{key}: {updated[key]},')
                continue
            if line.strip().startswith("}"):
                in_anomaly_block = False

        modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified).rstrip('\n') + '\n')
    return "Anomaly detection hyperparameters updated successfully."

@mcp.tool()
async def run_anomaly_detection() -> str:
    """Run main.py for anomaly_detection and return the last 500 characters from combined output."""

    try:
        process = await asyncio.create_subprocess_exec(
            "python", "-u", str(MAIN_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(MAIN_PATH.parent),
        )

        stdout_data, stderr_data = await process.communicate()

        # Decode outputs safely
        output = stdout_data.decode("utf-8", errors="ignore") if stdout_data else ""
        error_output = stderr_data.decode("utf-8", errors="ignore") if stderr_data else ""

        # Print for debugging
        if output:
            print("STDOUT:", output)
        if error_output:
            print("STDERR:", error_output)

        # Extract tag summary from output
        anom_lines = []
        capture = False

        for line in error_output.splitlines():

            if "wandb: Run summary:" in line:
                capture = True
                anom_lines.append(line)  # Include the triggering line
                continue  # Avoid falling through to the elif
            elif capture and line.startswith("wandb:   default/version_0/pixel_F1Max"):
                anom_lines.append(line)
                break
            elif capture:
                anom_lines.append(line)

        anom_lines_t = anom_lines[4:]

        print(anom_lines_t)

        anom_summary = "\n".join(anom_lines_t).strip() if anom_lines_t else "No metrics found."

        print(anom_summary)


        # Save full logs
        log_path = "output/logs/last_anomaly_detection_log.txt"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(output)
            f.write("\n\n=== STDERR ===\n")
            f.write(error_output)
            f.write(f"\n\n=== EXIT CODE ===\n{process.returncode}")

        if process.returncode == 0:
            return (
                f"Anomaly Detection completed successfully.\n\n"
                f"**Results Summary:**\n```\n{anom_summary}\n```\n"
                f"Full logs saved to `{log_path}`"
            )
        else:
            return (
                f"Anomaly detection failed with exit code {process.returncode}.\n"
                f"Error details: {error_output[-5000:]}\n"
                f"Full logs saved to `{log_path}`"
            )

    except Exception as e:
        return f"Error executing anomaly detection: {str(e)}"