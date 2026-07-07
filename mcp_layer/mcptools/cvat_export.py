import os
import json
import time
import zipfile
import tempfile
import logging
import requests
import traceback
from pathlib import Path
from dotenv import load_dotenv

import fiftyone as fo
import fiftyone.types as fot
from cvat_sdk import make_client
from cvat_sdk.api_client.model.data_request import DataRequest
from mcptools import mcp

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
CVAT_URL        = os.getenv("CVAT_URL", "https://app.cvat.ai")
CVAT_TASKS_FILE = ROOT_DIR / "output" / "cvat_tasks.json"

def _get_cvat_token() -> str | None:
    """Read token fresh each call so credential changes take effect without restart."""
    load_dotenv(override=True)
    return os.getenv("CVAT_ACCESS_TOKEN")


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
def export_to_cvat(
    dataset_name: str,
    with_predictions: bool = False,
    classes: list = None,
) -> str:
    CVAT_TOKEN = _get_cvat_token()
    if not CVAT_TOKEN:
        return "CVAT_ACCESS_TOKEN not set in .env"

    try:
        # fo.load_dataset() returns a process-wide singleton keyed by name — if this
        # process already loaded this dataset earlier (e.g. during selection/listing),
        # its in-memory schema can be stale relative to fields another process (the
        # auto-labeling subprocess) just wrote. reload() forces a resync from Mongo.
        dataset = fo.load_dataset(dataset_name)
        dataset.reload()
    except Exception as e:
        return f"Failed to load dataset '{dataset_name}': {e}"

    try:
        image_paths = [sample.filepath for sample in dataset]
        schema = dataset.get_field_schema()

        label_field = None
        classes = list(classes) if classes else []
        if with_predictions:
            pred_fields = [f for f in schema.keys() if f.startswith("pred_od_")]
            # Prediction field may not be in MongoDB yet; poll for it.
            if not pred_fields:
                for attempt in range(10):
                    time.sleep(2)
                    dataset.reload()
                    schema = dataset.get_field_schema()
                    pred_fields = [f for f in schema.keys() if f.startswith("pred_od_")]
                    if pred_fields:
                        image_paths = [sample.filepath for sample in dataset]
                        break
                else:
                    return (
                        f"No prediction field found on dataset '{dataset_name}' after 20 seconds. "
                        f"Inference may not have completed correctly. Please check the auto-labeling logs."
                    )
            if "predictions" in schema:
                label_field = "predictions"
            elif "ground_truth" in schema:
                label_field = "ground_truth"
            elif pred_fields:
                label_field = pred_fields[0]
                logging.info(f"Using prediction field: {label_field}")

            if label_field:
                classes = dataset.distinct(f"{label_field}.detections.label")

        with make_client(CVAT_URL, access_token=CVAT_TOKEN) as client:
            task = client.tasks.create({
                "name": dataset_name,
                "labels": [{"name": c} for c in classes],
            })
            task_id = task.id
            logging.info(f"Created CVAT task {task_id} for dataset '{dataset_name}'")

            _UPLOAD_RETRIES = 3
            upload_ok = False
            for attempt in range(_UPLOAD_RETRIES):
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
                    upload_ok = True
                    break
                except Exception as upload_err:
                    err_str_u = str(upload_err)
                    is_transient = any(
                        code in err_str_u for code in ("504", "502", "503", "Gateway Timeout")
                    )
                    if is_transient and attempt < _UPLOAD_RETRIES - 1:
                        logging.warning(
                            f"[CVAT] Upload timeout (attempt {attempt+1}/{_UPLOAD_RETRIES}), "
                            f"retrying in {5*(attempt+1)}s..."
                        )
                        time.sleep(5 * (attempt + 1))
                        continue
                    try:
                        client.tasks.remove(task_id)
                        logging.warning(f"[CVAT] Cleaned up orphaned task {task_id}")
                    except Exception:
                        pass
                    raise upload_err
                finally:
                    for f in file_objects:
                        f.close()

            if not upload_ok:
                # Should not reach here (raise above), but guard against logic errors.
                raise RuntimeError("Image upload failed after all retries.")

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

            registry = _load_task_registry()
            registry[dataset_name] = {
                "task_id": task_id,
                "uploaded_at": str(Path(__file__).stat().st_mtime),
                "with_predictions": with_predictions,
                "manual_classes": classes if not with_predictions else [],
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
            elif classes:
                msg += f"\nLabels configured: {classes}"

            logging.info(msg)
            return msg

    except Exception as e:
        err_str = str(e)
        tb = traceback.format_exc()

        if any(code in err_str for code in ("504", "502", "503", "Gateway Timeout", "Service Unavailable")):
            return (
                "CVAT_TIMEOUT_ERROR: CVAT timed out while uploading images. "
                "This usually happens with large datasets or when the CVAT server is under load. "
                "Please try again."
            )
        if "403" in err_str or "Forbidden" in err_str:
            if "maximum number of tasks" in err_str or "maximum number of tasks" in tb:
                return (
                    "CVAT_TASK_LIMIT_REACHED: Your CVAT account has reached the maximum number of tasks. "
                    "Please delete some existing tasks at app.cvat.ai to free up space, then try again."
                )
            return (
                "CVAT_FORBIDDEN: Access denied by CVAT. "
                "Please check your CVAT_ACCESS_TOKEN in .env is valid and has not expired."
            )
        if "401" in err_str or "Unauthorized" in err_str:
            return (
                "CVAT_AUTH_ERROR: CVAT authentication failed. "
                "Please check your CVAT_ACCESS_TOKEN in .env."
            )
        if "404" in err_str or "Not Found" in err_str:
            return (
                "CVAT_NOT_FOUND: The CVAT task or resource was not found. "
                "It may have been deleted. Please try exporting again."
            )
        if "ConnectionError" in tb or "ConnectTimeout" in tb:
            return (
                "CVAT_CONNECTION_ERROR: Could not reach CVAT at the configured URL. "
                f"Please check CVAT_URL in .env (currently: {CVAT_URL})."
            )

        logging.warning(f"[CVAT] export_to_cvat failed: {tb}")
        return f"CVAT upload failed: {err_str}"


@mcp.tool()
def import_from_cvat(dataset_name: str) -> str:
    CVAT_TOKEN = _get_cvat_token()
    if not CVAT_TOKEN:
        return "CVAT_ACCESS_TOKEN not set in .env"

    registry = _load_task_registry()
    if dataset_name not in registry:
        return f"No CVAT task found for dataset '{dataset_name}'. Please upload it first."

    task_id = registry[dataset_name]["task_id"]
    labeled_name = f"{dataset_name}_labeled"

    try:
        headers = {"Authorization": f"Bearer {CVAT_TOKEN}"}
        export_url = f"{CVAT_URL}/api/tasks/{task_id}/dataset/export"
        params = {"save_images": "False", "format": "CVAT for images 1.1"}

        response = requests.post(export_url, headers=headers, params=params)
        if response.status_code not in (200, 201, 202):
            return f"Failed to initiate export: {response.status_code} {response.text}"

        rq_id = response.json().get("rq_id")
        if not rq_id:
            return f"No rq_id in export response: {response.text}"

        result_url = None
        status_url = f"{CVAT_URL}/api/requests/{rq_id}"
        for _ in range(30):
            time.sleep(3)
            status_response = requests.get(status_url, headers=headers)
            status_data = status_response.json()
            status = status_data.get("status")
            if status == "finished":
                result_url = status_data.get("result_url")
                break
            elif status == "failed":
                return f"CVAT export failed: {status_data}"
        else:
            return "CVAT export timed out after 90 seconds."

        download_response = requests.get(result_url, headers=headers)
        if download_response.status_code != 200:
            return f"Failed to download annotations: {download_response.status_code}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "annotations.zip")

            with open(zip_path, "wb") as f:
                f.write(download_response.content)

            try:
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(tmp_dir)
                xml_files = list(Path(tmp_dir).rglob("*.xml"))
                if not xml_files:
                    return "No XML annotation file found in CVAT export."
                xml_path = str(xml_files[0])
            except zipfile.BadZipFile:
                xml_path = zip_path

            original_dataset = fo.load_dataset(dataset_name)
            data_path = str(Path(original_dataset.first().filepath).parent)

            if labeled_name in fo.list_datasets():
                fo.delete_dataset(labeled_name)

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

            for sample in labeled_dataset:
                if sample.has_field("detections"):
                    sample["ground_truth"] = sample["detections"]
                    sample.clear_field("detections")
                    sample.save()

            if "detections" in labeled_dataset.get_field_schema():
                labeled_dataset.delete_sample_field("detections")

            labeled_dataset.persistent = True

            for attempt in range(5):
                existing = fo.list_datasets()
                if labeled_name in existing:
                    verify = fo.load_dataset(labeled_name)
                    if len(verify) > 0:
                        break
                time.sleep(2)
            else:
                return (
                    f"Import appeared to succeed but dataset '{labeled_name}' "
                    f"could not be verified in FiftyOne after 10 seconds. "
                    f"Please try importing again."
                )

            msg = (
                f"Annotations imported successfully from CVAT task {task_id}.\n"
                f"New dataset '{labeled_name}' created with {len(labeled_dataset)} samples.\n"
                f"Labels saved as 'ground_truth' field."
            )
            logging.info(msg)
            return msg

    except Exception as e:
        return f"CVAT import failed: {e}\n{traceback.format_exc()}"