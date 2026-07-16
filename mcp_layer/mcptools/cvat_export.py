import os
import json
import time
import uuid
import asyncio
import zipfile
import tempfile
import logging
import requests
import traceback
from pathlib import Path
from dotenv import load_dotenv

import fiftyone as fo
import fiftyone.types as fot
from fastmcp import Context
from cvat_sdk import make_client
from mcptools import mcp

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
CVAT_URL        = os.getenv("CVAT_URL", "https://app.cvat.ai")
CVAT_TASKS_FILE = ROOT_DIR / "output" / "cvat_tasks.json"

# Uploads go through CVAT's chunked uploader (create_from_data), which handles
# large payloads fine -- this cap just keeps datasets that fit in fewer tasks
# from using more of the account's limited task budget than they need to.
CVAT_MAX_FILES_PER_TASK = 11000

# Account-wide total, not reserved per export -- export_to_cvat checks current
# usage and batches into whatever's actually free. HARD_LIMIT is the fallback
# ceiling when slots are scarce and batches must grow past the preferred size.
CVAT_MAX_CONCURRENT_TASKS = 3
CVAT_UPLOAD_HARD_LIMIT = 35000


def _compute_cvat_batches(image_paths: list, max_tasks: int) -> list:
    """Split image_paths into <= max_tasks batches, each within
    CVAT_UPLOAD_HARD_LIMIT files. Returns [] if the dataset can't fit.

    max_tasks is the number of task slots actually free on the account right
    now (CVAT_MAX_CONCURRENT_TASKS minus whatever's already there), not the
    raw account ceiling -- a dataset needing exactly 3 tasks has zero margin
    if even one unrelated task already exists.
    """
    import math

    total = len(image_paths)
    if max_tasks <= 0:
        return []
    if total <= CVAT_MAX_FILES_PER_TASK:
        batch_size = total
    else:
        batches_at_preferred_size = math.ceil(total / CVAT_MAX_FILES_PER_TASK)
        if batches_at_preferred_size <= max_tasks:
            batch_size = CVAT_MAX_FILES_PER_TASK
        else:
            batch_size = math.ceil(total / max_tasks)
            if batch_size > CVAT_UPLOAD_HARD_LIMIT:
                return []

    return [image_paths[i:i + batch_size] for i in range(0, total, batch_size)]

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


def _make_cvat_progress_reporter(ctx, loop, task_name: str):
    """Bridges CVAT SDK's (synchronous) ProgressReporter callbacks to ctx.log()
    (async), the same run_coroutine_threadsafe pattern used for HF downloads
    in workflow_selector.py's _load_dataset_with_progress."""
    from cvat_sdk.core.progress import ProgressReporter

    class _Reporter(ProgressReporter):
        def __init__(self):
            self._total = 0
            self._current = 0
            self._last_emit = 0.0

        def start(self, total, *, desc=None):
            self._total = total
            self._current = 0

        def start2(self, total, *, desc=None, **kwargs):
            self.start(total, desc=desc)

        def report_status(self, progress):
            self._current = progress
            self._maybe_emit()

        def advance(self, delta):
            self._current += delta
            self._maybe_emit()

        def _maybe_emit(self):
            if not self._total:
                return
            now = time.time()
            is_done = self._current >= self._total
            if now - self._last_emit >= 15 or is_done:
                self._last_emit = now
                pct = self._current / self._total * 100
                msg = f"Uploading {task_name}: {self._current}/{self._total} ({pct:.0f}%)"
                asyncio.run_coroutine_threadsafe(ctx.log(msg), loop)

    return _Reporter()


def _create_and_upload_batch(
    client, task_name: str, batch_paths: list, classes: list, ctx=None, loop=None,
) -> int:
    """Create one CVAT task and upload its images via CVAT's own chunked
    uploader (create_from_data -> DataUploader), which auto-splits into
    <=100MB requests and falls back to TUS for any single file larger than
    that. This replaces a raw single-multipart create_data() call that sent
    every image in the batch as one unbounded request -- that was hitting
    server-side connection drops (499) on large batches; this is the upload
    path CVAT's own SDK recommends for local files (see cvat-ai/cvat #1616,
    PR #3692 "Large files uploads"). Runs synchronously -- callers should
    offload via asyncio.to_thread.
    """
    from cvat_sdk.core.proxies.tasks import ResourceType

    pbar = _make_cvat_progress_reporter(ctx, loop, task_name) if ctx and loop else None

    _RETRIES = 3
    for attempt in range(_RETRIES):
        try:
            task = client.tasks.create_from_data(
                spec={"name": task_name, "labels": [{"name": c} for c in classes]},
                resources=[Path(p) for p in batch_paths],
                resource_type=ResourceType.LOCAL,
                data_params={"image_quality": 70},
                pbar=pbar,
            )
            logging.info(
                f"Created CVAT task {task.id} ('{task_name}') with {len(batch_paths)} images"
            )
            return task.id
        except Exception as upload_err:
            err_str_u = str(upload_err)
            is_transient = any(
                code in err_str_u for code in (
                    "504", "502", "503", "Gateway Timeout",
                    "Status Code: 499", "Client Closed Request",
                )
            )
            # create_from_data creates + uploads in one call, so a failed upload
            # may leave the task behind -- remove it by name before retrying.
            for t in client.tasks.list():
                if t.name == task_name:
                    try:
                        client.tasks.remove_by_ids([t.id])
                        logging.warning(f"[CVAT] Cleaned up partial task {t.id} before retry")
                    except Exception:
                        pass
            if is_transient and attempt < _RETRIES - 1:
                logging.warning(
                    f"[CVAT] Upload failed (attempt {attempt+1}/{_RETRIES}), "
                    f"retrying in {5*(attempt+1)}s..."
                )
                time.sleep(5 * (attempt + 1))
                continue
            raise

    # Unreachable (the loop always returns or raises), but keeps type checkers happy.
    raise RuntimeError("Image upload failed after all retries.")


def _attach_predictions(client, task_id: int, batch_view, label_field: str) -> None:
    """Export a batch's predictions as CVAT XML and attach them to its task.
    Runs synchronously -- callers should offload via asyncio.to_thread."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        batch_view.export(
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


@mcp.tool()
async def export_to_cvat(
    dataset_name: str,
    with_predictions: bool = False,
    classes: list = None,
    ctx: Context = None,
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

        loop = asyncio.get_running_loop()
        created_task_ids = []
        with make_client(CVAT_URL, access_token=CVAT_TOKEN) as client:
            # Check slots actually free right now, not just the account ceiling.
            existing_task_count = len(client.tasks.list())
            available_slots = CVAT_MAX_CONCURRENT_TASKS - existing_task_count
            batches = _compute_cvat_batches(image_paths, available_slots)
            if not batches:
                return (
                    f"CVAT export not possible for '{dataset_name}': {len(image_paths)} images "
                    f"need more than {available_slots} task(s) to fit at the ~{CVAT_UPLOAD_HARD_LIMIT}-file "
                    f"per-task upload limit. This CVAT account allows {CVAT_MAX_CONCURRENT_TASKS} tasks "
                    f"total, and {existing_task_count} already exist. "
                    f"Please delete some existing tasks at {CVAT_URL}, use Label Studio instead, "
                    f"or export a smaller subset."
                )
            num_batches = len(batches)

            try:
                for batch_idx, batch_paths in enumerate(batches, start=1):
                    task_name = (
                        dataset_name if num_batches == 1
                        else f"{dataset_name}_part{batch_idx}of{num_batches}"
                    )
                    if ctx:
                        await ctx.log(
                            f"Uploading batch {batch_idx}/{num_batches} "
                            f"({len(batch_paths)} images) to CVAT..."
                        )
                    task_id = await asyncio.to_thread(
                        _create_and_upload_batch, client, task_name, batch_paths, classes, ctx, loop,
                    )
                    created_task_ids.append(task_id)
                    if ctx:
                        await ctx.log(
                            f"Batch {batch_idx}/{num_batches} uploaded -> CVAT task {task_id}"
                        )

                    if with_predictions and label_field:
                        batch_view = dataset.select_by("filepath", batch_paths)
                        await asyncio.to_thread(
                            _attach_predictions, client, task_id, batch_view, label_field,
                        )
            except Exception:
                # A later batch failing shouldn't leave earlier ones orphaned.
                if created_task_ids:
                    try:
                        client.tasks.remove_by_ids(created_task_ids)
                        logging.warning(f"[CVAT] Cleaned up orphaned tasks {created_task_ids} after batch failure")
                    except Exception:
                        pass
                raise

            registry = _load_task_registry()
            registry[dataset_name] = {
                "task_ids": created_task_ids,
                "uploaded_at": str(Path(__file__).stat().st_mtime),
                "with_predictions": with_predictions,
                "manual_classes": classes if not with_predictions else [],
            }
            _save_task_registry(registry)

            task_lines = "\n".join(
                f"  - Task {tid}: {CVAT_URL}/tasks/{tid}" for tid in created_task_ids
            )
            msg = (
                f"Dataset '{dataset_name}' uploaded to CVAT successfully "
                f"({num_batches} task{'s' if num_batches > 1 else ''}).\n"
                f"Task IDs: {', '.join(map(str, created_task_ids))}\n"
                f"Images: {len(image_paths)}\n"
                f"{task_lines}"
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

        if any(code in err_str for code in (
            "504", "502", "503", "Gateway Timeout", "Service Unavailable",
            "Status Code: 499", "Client Closed Request",
        )):
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


def _download_and_import_task(task_id: int, data_path: str, dataset_name: str) -> tuple:
    """Download one CVAT task's annotations and import them into a throwaway
    temp dataset. Returns (temp_dataset_name, error_message); exactly one is
    None. Runs synchronously -- callers should offload via asyncio.to_thread.
    """
    CVAT_TOKEN = _get_cvat_token()
    headers = {"Authorization": f"Bearer {CVAT_TOKEN}"}

    # Guard against a stale registry merging another dataset's annotations in
    # (real risk if two datasets share generic filenames like "0001.jpg").
    task_resp = requests.get(f"{CVAT_URL}/api/tasks/{task_id}", headers=headers)
    if task_resp.status_code != 200:
        return None, f"Could not verify task {task_id} ownership: {task_resp.status_code} {task_resp.text}"
    task_name = task_resp.json().get("name", "")
    if not (task_name == dataset_name or task_name.startswith(f"{dataset_name}_part")):
        return None, (
            f"Task {task_id} is named '{task_name}', which doesn't match dataset "
            f"'{dataset_name}' — skipped to avoid merging mismatched annotations. "
            f"The task registry (output/cvat_tasks.json) may be stale."
        )

    export_url = f"{CVAT_URL}/api/tasks/{task_id}/dataset/export"
    params = {"save_images": "False", "format": "CVAT for images 1.1"}

    response = requests.post(export_url, headers=headers, params=params)
    if response.status_code not in (200, 201, 202):
        return None, f"Failed to initiate export: {response.status_code} {response.text}"

    rq_id = response.json().get("rq_id")
    if not rq_id:
        return None, f"No rq_id in export response: {response.text}"

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
            logging.warning(f"[CVAT] annotation export job failed: {status_data}")
            return None, (
                "CVAT reported that the annotation export job failed. "
                "This is often a transient server-side issue — please try again in a moment."
            )
    else:
        return None, f"CVAT export timed out after 90 seconds (task {task_id})."

    download_response = requests.get(result_url, headers=headers)
    if download_response.status_code != 200:
        return None, f"Failed to download annotations: {download_response.status_code}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "annotations.zip")

        with open(zip_path, "wb") as f:
            f.write(download_response.content)

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp_dir)
            xml_files = list(Path(tmp_dir).rglob("*.xml"))
            if not xml_files:
                return None, f"No XML annotation file found in CVAT export (task {task_id})."
            xml_path = str(xml_files[0])
        except zipfile.BadZipFile:
            xml_path = zip_path

        tmp_name = f"__cvat_import_tmp_{task_id}_{uuid.uuid4().hex[:8]}"
        try:
            tmp_dataset = fo.Dataset.from_dir(
                dataset_type=fot.CVATImageDataset,
                data_path=data_path,
                labels_path=xml_path,
                name=tmp_name,
            )
        except AttributeError:
            return None, f"CVAT task {task_id} has no annotations yet."

        for sample in tmp_dataset:
            if sample.has_field("detections"):
                sample["ground_truth"] = sample["detections"]
                sample.clear_field("detections")
                sample.save()
        if "detections" in tmp_dataset.get_field_schema():
            tmp_dataset.delete_sample_field("detections")
        tmp_dataset.persistent = True

    return tmp_name, None


@mcp.tool()
async def import_from_cvat(dataset_name: str, ctx: Context = None) -> str:
    CVAT_TOKEN = _get_cvat_token()
    if not CVAT_TOKEN:
        return "CVAT_ACCESS_TOKEN not set in .env"

    registry = _load_task_registry()
    if dataset_name not in registry:
        return f"No CVAT task found for dataset '{dataset_name}'. Please upload it first."

    # "task_ids" (plural, list) is the current format; fall back to the older
    # single "task_id" entries from before multi-task chunking existed.
    task_ids = registry[dataset_name].get("task_ids")
    if not task_ids:
        legacy_id = registry[dataset_name].get("task_id")
        task_ids = [legacy_id] if legacy_id else []
    if not task_ids:
        return f"No CVAT task IDs recorded for dataset '{dataset_name}'. Please upload it first."

    labeled_name = f"{dataset_name}_labeled"
    num_tasks = len(task_ids)

    try:
        original_dataset = fo.load_dataset(dataset_name)
        data_path = str(Path(original_dataset.first().filepath).parent)

        if labeled_name in fo.list_datasets():
            fo.delete_dataset(labeled_name)
        final_dataset = fo.Dataset(name=labeled_name)

        errors = []
        for i, task_id in enumerate(task_ids, start=1):
            if ctx:
                await ctx.log(f"Importing annotations from CVAT task {i}/{num_tasks} (ID {task_id})...")
            tmp_name, err = await asyncio.to_thread(_download_and_import_task, task_id, data_path, dataset_name)
            if err:
                logging.warning(f"[CVAT] task {task_id}: {err}")
                errors.append(f"Task {task_id}: {err}")
                continue
            tmp_dataset = fo.load_dataset(tmp_name)
            final_dataset.merge_samples(tmp_dataset)
            fo.delete_dataset(tmp_name)
            if ctx:
                await ctx.log(f"Task {i}/{num_tasks} (ID {task_id}) merged — {len(final_dataset)} samples so far")

        if len(final_dataset) == 0:
            fo.delete_dataset(labeled_name)
            detail = "\n".join(errors) if errors else "no annotations were found."
            return (
                f"None of the {num_tasks} CVAT task(s) for '{dataset_name}' produced annotations.\n{detail}\n"
                f"Please annotate the images in CVAT first, then try importing again."
            )

        final_dataset.persistent = True

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
            f"Annotations imported successfully from {num_tasks} CVAT task(s).\n"
            f"New dataset '{labeled_name}' created with {len(final_dataset)} samples.\n"
            f"Labels saved as 'ground_truth' field."
        )
        if errors:
            msg += "\n\nSome tasks had no annotations and were skipped:\n" + "\n".join(errors)
        logging.info(msg)
        return msg

    except Exception as e:
        logging.warning(f"[CVAT] import_from_cvat failed: {traceback.format_exc()}")
        return f"CVAT import failed: {e}"