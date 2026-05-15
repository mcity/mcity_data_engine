from mcptools import mcp
import fiftyone as fo
import fiftyone.types as fot
import logging
import os
import json
import zipfile
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
CVAT_URL = os.getenv("CVAT_URL", "https://app.cvat.ai")
CVAT_TOKEN = os.getenv("CVAT_ACCESS_TOKEN")
CVAT_TASKS_FILE = ROOT_DIR / "output" / "cvat_tasks.json"


def _load_task_registry() -> dict:
    if CVAT_TASKS_FILE.exists():
        with open(CVAT_TASKS_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_task_registry(registry: dict):
    CVAT_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CVAT_TASKS_FILE, "w") as f:
        json.dump(registry, f, indent=2)


@mcp.tool()
def export_to_cvat(dataset_name: str, with_predictions: bool = False) -> str:
    if not CVAT_TOKEN:
        return "CVAT_ACCESS_TOKEN not set in .env"

    try:
        dataset = fo.load_dataset(dataset_name)
    except Exception as e:
        return f"Failed to load dataset '{dataset_name}': {e}"

    image_paths = [sample.filepath for sample in dataset]
    schema = dataset.get_field_schema()

    try:
        from cvat_sdk import make_client
        from cvat_sdk.api_client.model.data_request import DataRequest
        import tempfile, zipfile

        # Determine label field for predictions
        label_field = None
        classes = []
        if with_predictions:
            pred_fields = [f for f in schema.keys() if f.startswith("pred_od_")]
            if "predictions" in schema:
                label_field = "predictions"
            elif "ground_truth" in schema:
                label_field = "ground_truth"
            elif pred_fields:
                label_field = pred_fields[0]
                logging.info(f"Using RF-DETR prediction field: {label_field}")

            if label_field:
                classes = dataset.distinct(f"{label_field}.detections.label")

        with make_client(CVAT_URL, access_token=CVAT_TOKEN) as client:
            # Create task with labels
            task = client.tasks.create({
                "name": dataset_name,
                "labels": [{"name": c} for c in classes],
            })
            task_id = task.id
            logging.info(f"Created CVAT task {task_id} for dataset '{dataset_name}'")

            # Upload images
            file_objects = [open(p, "rb") for p in image_paths]
            try:
                client.api_client.tasks_api.create_data(
                    id=task_id,
                    data_request=DataRequest(
                        image_quality=70,
                        client_files=file_objects,
                    ),
                    _content_type="multipart/form-data",
                )
            finally:
                for f in file_objects:
                    f.close()

            # Upload annotations if predictions exist
            if with_predictions and label_field:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    dataset.export(
                        export_dir=tmp_dir,
                        dataset_type=fo.types.CVATImageDataset,
                        label_field=label_field,
                    )
                    xml_path = os.path.join(tmp_dir, "labels.xml")
                    zip_path = os.path.join(tmp_dir, "annotations.zip")
                    with zipfile.ZipFile(zip_path, "w") as zf:
                        zf.write(xml_path, "annotations.xml")

                    task = client.tasks.retrieve(task_id)
                    task.import_annotations(
                        format_name="CVAT 1.1",
                        filename=zip_path,
                    )
                    logging.info(f"Annotations uploaded to CVAT task {task_id}")

            # Save task registry
            registry = _load_task_registry()
            registry[dataset_name] = {
                "task_id": task_id,
                "uploaded_at": str(Path(__file__).stat().st_mtime),
                "with_predictions": with_predictions,
            }
            _save_task_registry(registry)

            msg = (
                f"Dataset '{dataset_name}' uploaded to CVAT successfully.\n"
                f"Task ID: {task_id}\n"
                f"Images: {len(image_paths)}\n"
                f"Open in CVAT: {CVAT_URL}/tasks/{task_id}"
            )
            if with_predictions and label_field:
                msg += f"\nPredictions uploaded with labels: {classes}"

            logging.info(msg)
            return msg

    except Exception as e:
        return f"CVAT upload failed: {e}"


@mcp.tool()
def import_from_cvat(dataset_name: str) -> str:
    """
    Download annotations from CVAT for a previously uploaded dataset,
    and save them as a new FiftyOne dataset named <dataset_name>_labeled.
    """
    if not CVAT_TOKEN:
        return "CVAT_ACCESS_TOKEN not set in .env"

    # Look up task_id from registry
    registry = _load_task_registry()
    if dataset_name not in registry:
        return f"No CVAT task found for dataset '{dataset_name}'. Please upload it first."

    task_id = registry[dataset_name]["task_id"]
    labeled_name = f"{dataset_name}_labeled"

    try:
        from cvat_sdk import make_client
        import requests

        # Download annotations XML via REST API
        # Step 1: Initiate export
        headers = {"Authorization": f"Bearer {CVAT_TOKEN}"}
        export_url = f"{CVAT_URL}/api/tasks/{task_id}/dataset/export"
        params = {"save_images": "False", "format": "CVAT for images 1.1"}

        response = requests.post(export_url, headers=headers, params=params)
        if response.status_code not in (200, 201, 202):
            return f"Failed to initiate export: {response.status_code} {response.text}"

        rq_id = response.json().get("rq_id")
        if not rq_id:
            return f"No rq_id in export response: {response.text}"

        # Step 2: Poll for completion
        import time
        result_url = None
        status_url = f"{CVAT_URL}/api/requests/{rq_id}"
        for _ in range(30):
            time.sleep(3)
            status_response = requests.get(status_url, headers=headers)
            print(f"DEBUG status_response status: {status_response.status_code}")
            print(f"DEBUG status_response text: {status_response.text[:200]}")
            status_data = status_response.json()
            status = status_data.get("status")
            if status == "finished":
                result_url = status_data.get("result_url")
                print(f"DEBUG status_data: {status_data}")
                break
            elif status == "failed":
                return f"CVAT export failed: {status_data}"
        else:
            return "CVAT export timed out after 90 seconds."

        # Step 3: Download the file
        download_response = requests.get(result_url, headers=headers)
        if download_response.status_code != 200:
            return f"Failed to download file: {download_response.status_code}"

        # Save annotations to temp file
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "annotations.zip")
            xml_path = os.path.join(tmp_dir, "annotations.xml")

            # Response might be zip or raw XML
            with open(zip_path, "wb") as f:
                f.write(download_response.content)

            # Try to unzip
            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(tmp_dir)
                # Find the XML file
                xml_files = list(Path(tmp_dir).rglob("*.xml"))
                if not xml_files:
                    return "No XML annotation file found in CVAT export."
                xml_path = str(xml_files[0])
            except zipfile.BadZipFile:
                # Already raw XML
                xml_path = zip_path

            # Load original dataset to get image paths
            original_dataset = fo.load_dataset(dataset_name)
            data_path = str(Path(original_dataset.first().filepath).parent)

            # Delete existing labeled dataset if it exists
            if labeled_name in fo.list_datasets():
                fo.delete_dataset(labeled_name)

            # Load as new FiftyOne dataset
            try:
                labeled_dataset = fo.Dataset.from_dir(
                    dataset_type=fot.CVATImageDataset,
                    data_path=data_path,
                    labels_path=xml_path,
                    name=labeled_name,
                )
            except AttributeError:
                return (
                    f"The CVAT task {task_id} has no annotations yet. "
                    f"Please annotate the images in CVAT first, then try importing again."
                )

            # Rename detections to ground_truth
            for sample in labeled_dataset:
                if sample.has_field("detections"):
                    sample["ground_truth"] = sample["detections"]
                    sample.clear_field("detections")
                    sample.save()

            if "detections" in labeled_dataset.get_field_schema():
                labeled_dataset.delete_sample_field("detections")

            labeled_dataset.persistent = True

            msg = (
                f"Annotations imported successfully from CVAT task {task_id}.\n"
                f"New dataset '{labeled_name}' created with {len(labeled_dataset)} samples.\n"
                f"Labels saved as 'ground_truth' field."
            )
            logging.info(msg)
            return msg

    except Exception as e:
        import traceback
        return f"CVAT import failed: {e}\n{traceback.format_exc()}"