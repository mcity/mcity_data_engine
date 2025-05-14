from fastmcp import FastMCP
import subprocess
import re
import asyncio
from pathlib import Path

mcp = FastMCP("AutoLabeling Agent")

CONFIG_PATH = Path("/home/dataengine/Downloads/mcity_data_engine/config/config.py")
MAIN_PATH = Path("/home/dataengine/Downloads/mcity_data_engine/main.py")

@mcp.tool()
def configure_auto_labeling(selected_source: str, selected_model: str) -> str:
    """
    Enable the selected model source and model inside config.py for the auto_labeling workflow.
    """
    lines = CONFIG_PATH.read_text().split('\n')
    modified = []

    in_model_source = False
    in_selected_source = False
    in_model_dict = False
    in_config_list = False

    source_key_pattern = re.compile(rf'^\s*"{selected_source}"\s*:\s*{{')
    model_line_pattern = re.compile(r'^\s*"([^"]+)"\s*:')
    config_line_pattern = re.compile(r'^\s*"([^"]+\.py)"')

    for line in lines:
        stripped = line.strip()

        if '"model_source": [' in line:
            in_model_source = True
            modified.append(line)
            continue
        if in_model_source:
            if ']' in stripped:
                in_model_source = False
                modified.append(line)
                continue
            match = re.search(r'"([^"]+)"', stripped)
            if match:
                source = match.group(1)
                if source == selected_source:
                    modified.append(f'        "{source}",')
                else:
                    modified.append(f'        # "{source}",')
                continue

        if source_key_pattern.match(line):
            in_selected_source = True
            modified.append(line)
            continue

        if in_selected_source:
            if stripped.startswith("}"):
                in_selected_source = False
                modified.append(line)
                continue

            if '"models": {' in line and selected_source == "ultralytics":
                in_model_dict = True
                modified.append(line)
                continue

            if in_model_dict:
                if stripped.startswith("}"):
                    in_model_dict = False
                    modified.append(line)
                    continue
                match = model_line_pattern.match(line)
                if match:
                    model_name = match.group(1)
                    if model_name == selected_model:
                        modified.append(line.lstrip('#').strip())
                    else:
                        modified.append("#" + line if not line.strip().startswith("#") else line)
                    continue

            if selected_source == "hf_models_objectdetection":
                match = model_line_pattern.match(line)
                if match:
                    model_name = match.group(1)
                    if model_name == selected_model:
                        modified.append(line.lstrip('#').strip())
                    else:
                        modified.append("#" + line if not line.strip().startswith("#") else line)
                    continue

            if '"configs": [' in line and selected_source == "custom_codetr":
                in_config_list = True
                modified.append(line)
                continue

            if in_config_list:
                if ']' in stripped:
                    in_config_list = False
                    modified.append(line)
                    continue
                match = config_line_pattern.match(line)
                if match:
                    config_path = match.group(1)
                    if selected_model in config_path:
                        modified.append(line.lstrip('#').strip())
                    else:
                        modified.append("#" + line if not line.strip().startswith("#") else line)
                    continue

        modified.append(line)

    CONFIG_PATH.write_text('\n'.join(modified))
    return f"✅ Config updated to use `{selected_model}` from `{selected_source}`."

@mcp.tool()
def list_model_sources_and_models() -> dict:
    """
    Lists valid model sources and models for auto_labeling.
    """
    return {
        "ultralytics": ["yolo12n", "yolo12x"],
        "hf_models_objectdetection": [
            "microsoft/conditional-detr-resnet-50",
            "Omnifact/conditional-detr-resnet-101-dc5",
            "facebook/detr-resnet-50",
            "facebook/detr-resnet-50-dc5",
            "facebook/detr-resnet-101",
            "facebook/detr-resnet-101-dc5",
            "facebook/deformable-detr-detic",
            "facebook/deformable-detr-box-supervised",
            "SenseTime/deformable-detr",
            "SenseTime/deformable-detr-with-box-refine",
            "jozhang97/deta-swin-large",
            "jozhang97/deta-swin-large-o365",
            "hustvl/yolos-base"
        ],
        "custom_codetr": [
            "co_deformable_detr_r50_1x_coco.py",
            "co_dino_5scale_vit_large_coco.py"
        ]
    }


@mcp.tool()
async def run_auto_labeling() -> str:
    """Run main.py and stream logs in real time."""
    process = await asyncio.create_subprocess_exec(
        "python", str(MAIN_PATH),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    output = ""
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="ignore")
        print(decoded, end="")  # Shows in your terminal
        output += decoded

    await process.wait()

    if process.returncode == 0:
        return f"✅ Auto-labeling finished successfully.\n\n{output}"
    else:
        return f"❌ Auto-labeling failed with exit code {process.returncode}.\n\n{output}"


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
