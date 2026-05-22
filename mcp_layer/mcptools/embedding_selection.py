from mcptools import mcp
import re
import asyncio
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "config" / "config.py"
MAIN_PATH = ROOT_DIR / "main.py"



@mcp.tool()
def list_embedding_selection_models() -> list:
    """
    Lists available models for embedding_selection.
    """
    return [
        "clip-vit-base32-torch",
        "open-clip-torch",
        "dinov2-vits14-torch",
        "dinov2-vits14-reg-torch",
        "mobilenet-v2-imagenet-torch",
        "resnet152-imagenet-torch",
        "vgg19-imagenet-torch",
        "classification-transformer-torch",
        "detection-transformer-torch",
        "zero-shot-detection-transformer-torch",
        "zero-shot-classification-transformer-torch",
    ]

@mcp.tool()
def configure_embedding_selection_model(selected_model: str) -> str:
    """
    Enables only the selected model in Embedding_selection > embedding_models
    by commenting out all other models.
    """
    emb_lines = CONFIG_PATH.read_text().split('\n')
    emb_modified = []
    in_e_s = False
    in_emb_model_list = False

    for line in emb_lines:
        stripped = line.strip()

        if '"embedding_selection": {' in line:
            in_e_s = True
            emb_modified.append(line)
            continue

        if in_e_s and '"embedding_models": [' in line:
            in_emb_model_list = True
            emb_modified.append(line)
            continue

        if in_emb_model_list:
            if ']' in stripped:
                in_emb_model_list = False
                emb_modified.append(line)
                continue

            match = re.search(r'"([^"]+)"', stripped)
            if match:
                model_name = match.group(1)
                if model_name == selected_model:
                    emb_modified.append(f'        "{model_name}",')
                else:
                    emb_modified.append(f'        # "{model_name}",')
                continue

        if in_e_s and '}' in stripped:
            in_e_s = False

        emb_modified.append(line)

    CONFIG_PATH.write_text('\n'.join(emb_modified) + "\n")
    return f"Embedding Selection model updated to `{selected_model}`."



@mcp.tool()
def set_embedding_selection_params(
    compute_representativeness: float = 0.99,
    compute_unique_images_greedy: float = 0.01,
    compute_unique_images_deterministic: float = 0.99,
    compute_similar_images: float = 0.03,
    neighbour_count: int = 3
) -> str:
    """
    Updates parameters under the embedding_selection workflow in config.py.
    """
    lines = CONFIG_PATH.read_text().split("\n")
    modified = []

    in_embedding_block = False
    in_parameters_block = False

    updates = {
        "compute_representativeness": compute_representativeness,
        "compute_unique_images_greedy": compute_unique_images_greedy,
        "compute_unique_images_deterministic": compute_unique_images_deterministic,
        "compute_similar_images": compute_similar_images,
        "neighbour_count": neighbour_count,
    }

    for line in lines:
        stripped = line.strip()

        if '"embedding_selection": {' in line:
            in_embedding_block = True
            modified.append(line)
            continue

        if in_embedding_block and '"parameters": {' in line:
            in_parameters_block = True
            modified.append(line)
            continue

        if in_parameters_block:
            if stripped.startswith("}"):
                in_parameters_block = False
                modified.append(line)
                continue

            match = re.match(r'"([^"]+)"\s*:', stripped)
            if match:
                key = match.group(1)
                if key in updates:
                    indent = " " * (len(line) - len(line.lstrip()))
                    val = updates[key]
                    val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                    modified.append(f'{indent}"{key}": {val_str},')
                    continue

        if in_embedding_block and stripped.startswith("}"):
            in_embedding_block = False

        modified.append(line)

    CONFIG_PATH.write_text("\n".join(modified) + "\n")
    return "Embedding Selection parameters updated successfully."

@mcp.tool()
async def run_embedding_selection() -> str:
    """
    Run embedding_selection workflow and guide user to interpret results using Voxel51.
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

        log_path = "output/logs/last_embedding_selection_log.txt"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n")
            f.write(output)
            f.write("\n\n=== STDERR ===\n")
            f.write(error_output)
            f.write(f"\n\n=== EXIT CODE ===\n{process.returncode}")

        if process.returncode == 0:
            return (
                "Embedding Selection completed successfully.\n\n"
                "Launch Voxel51 to explore results:\n"
                "- Filter using the `embedding_selection` field to explore subsets like `greedy_center`, "
                "`representativeness_center`, and `greedy_neighbour`.\n"
                "- Use sliders on scalar fields such as:\n"
                "  • `embedding_selection_count` — shows how many times each image was selected\n"
                "  • `representativeness_cluster_center`\n"
                "  • `uniqueness_`\n"
                "  • `distance` — to analyze similarity\n"
                "- These allow you to visually inspect how your hyperparameter choices shaped the curated dataset.\n\n"
                f"Full logs saved to `{log_path}`"
            )
        else:
            return (
                f"Embedding Selection failed with exit code {process.returncode}.\n"
                f"Error details: {error_output[-5000:]}\n"
                f"Full logs saved to `{log_path}`"
            )

    except Exception as e:
        return f"Error executing embedding selection: {str(e)}"