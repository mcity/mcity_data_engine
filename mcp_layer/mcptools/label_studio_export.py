import os
import json
import time
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv

import fiftyone as fo
from label_studio_sdk import LabelStudio
from mcptools import mcp

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]

LS_URL = os.getenv("LS_URL", "https://app.humansignal.com")

def _get_ls_token() -> str | None:
    """Read token fresh each call so credential changes take effect without restart."""
    load_dotenv(override=True)
    return os.getenv("LS_TOKEN")

LS_TASKS_FILE = ROOT_DIR / "output" / "ls_tasks.json"


# Internal helpers

def _get_client():
    """Return an authenticated LabelStudio SDK client."""
    token = _get_ls_token()
    if not token:
        raise RuntimeError("LS_TOKEN not set in .env")
    return LabelStudio(base_url=LS_URL, api_key=token, timeout=60)


def _get_http(client):
    """Return the internal HttpClient for multipart/raw requests."""
    return client._client_wrapper.httpx_client


def _load_registry() -> dict:
    if LS_TASKS_FILE.exists():
        with open(LS_TASKS_FILE) as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict):
    LS_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LS_TASKS_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def _build_label_config(classes: list[str]) -> str:
    """
    Generate a Label Studio XML label config for rectangle labeling.
    If no classes are provided (manual path), labels are left empty and the
    annotator adds them manually in the UI.
    """
    label_tags = "\n    ".join(
        f'<Label value="{c}"/>' for c in classes
    ) if classes else ""
    return f"""<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    {label_tags}
  </RectangleLabels>
</View>"""


def _upload_image(http, project_id: int, path: Path) -> list[int]:
    """
    Upload a single image to a Label Studio project.
    Returns a list of task IDs, resolving the async import job if needed.
    """
    import mimetypes
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"

    with open(path, "rb") as f:
        resp = http.request(
            f"/api/projects/{project_id}/import",
            method="POST",
            files={"file": (path.name, f, mime)},
            params={"return_task_ids": "true"},
        )

    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Image upload failed for {path.name}: "
            f"{resp.status_code} {resp.text[:200]}"
        )

    import_job_id = resp.json().get("import")
    if not import_job_id:
        # Older API versions return task_ids directly.
        return resp.json().get("task_ids", [])

    # Poll import job until completed.
    for _ in range(15):
        job_resp = http.request(
            f"/api/projects/{project_id}/imports/{import_job_id}",
            method="GET",
        )
        if job_resp.status_code == 200:
            job = job_resp.json()
            if job.get("status") in ("completed", "finished") or job.get("task_ids"):
                return job.get("task_ids", [])
        time.sleep(1)

    raise RuntimeError(
        f"Import job {import_job_id} did not complete within 15 seconds."
    )


def _upload_images_concurrent(
    http, project_id: int, paths: list[Path], max_workers: int = 3
) -> dict[str, int]:
    """
    Upload images one-per-POST using a thread pool, return {filename: task_id}.

    LS creates exactly one task per POST regardless of how many files are packed
    into a single multipart body, so per-file uploads are required for N tasks.
    Concurrency removes the per-image latency; max_workers=3 stays under LS
    cloud rate limits (429 appears above ~5 simultaneous uploads).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(paths)
    t0 = time.time()
    logging.warning(f"[LS] Uploading {total} image(s) ({max_workers} concurrent)…")

    def _upload_one(path: Path) -> tuple[str, int]:
        for attempt in range(4):
            try:
                task_ids = _upload_image(http, project_id, path)
                if not task_ids:
                    raise RuntimeError(f"No task_id returned for {path.name}")
                return path.name, task_ids[0]
            except RuntimeError as e:
                if "429" in str(e) and attempt < 3:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logging.warning(f"[LS] 429 on {path.name}, retry in {wait}s…")
                    time.sleep(wait)
                else:
                    raise

    fname_to_task_id: dict[str, int] = {}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_upload_one, p): p for p in paths}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                fname, task_id = future.result()
                fname_to_task_id[fname] = task_id
            except Exception as e:
                errors.append(str(e))
            if done % 10 == 0 or done == total:
                logging.warning(f"[LS] {done}/{total} uploaded…")

    if errors:
        logging.warning(f"[LS] {len(errors)} upload error(s): {errors[:3]}")

    logging.warning(
        f"[LS] {len(fname_to_task_id)}/{total} images uploaded in {time.time()-t0:.1f}s"
    )
    return fname_to_task_id


def _detections_to_ls_results(detections) -> list:
    """
    Convert FiftyOne detections to Label Studio rectanglelabels result format.
    FiftyOne bbox: [x, y, w, h] normalized [0,1] -> LS percentages [0,100].
    """
    results = []
    for det in detections:
        x, y, w, h = det.bounding_box
        results.append({
            "from_name": "label",
            "to_name":   "image",
            "type":      "rectanglelabels",
            "value": {
                "x":               x * 100,
                "y":               y * 100,
                "width":           w * 100,
                "height":          h * 100,
                "rectanglelabels": [det.label],
            },
        })
    return results


def _attach_as_annotations(http, task_id: int, detections, label_field: str):
    """
    Import model detections as annotations (not predictions) so they appear
    as editable boxes in the Label Studio labeling interface.

    Predictions are read-only and cause a blank image when clicking Label.
    Annotations are editable and become ground truth after the annotator submits.
    """
    results = _detections_to_ls_results(detections)
    if not results:
        return

    resp = http.request(
        f"/api/tasks/{task_id}/annotations",
        method="POST",
        json={
            "result":        results,
            "was_cancelled": False,
            "ground_truth":  False,
            "lead_time":     0,
        },
    )
    if resp.status_code not in (200, 201):
        logging.warning(
            f"[LS] Annotation upload failed for task {task_id}: "
            f"{resp.status_code} {resp.text[:200]}"
        )


def _export_snapshot(http, project_id: int) -> list[dict]:
    """
    Create an export snapshot, poll until ready, and download as JSON.
    Returns a list of task dicts.
    """
    snap_resp = http.request(
        f"/api/projects/{project_id}/exports",
        method="POST",
        json={"title": f"export_{int(time.time())}"},
    )
    if snap_resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Export snapshot creation failed: "
            f"{snap_resp.status_code} {snap_resp.text[:200]}"
        )

    export_id = snap_resp.json().get("id")

    for _ in range(20):
        status_resp = http.request(
            f"/api/projects/{project_id}/exports/{export_id}",
            method="GET",
        )
        if status_resp.status_code == 200:
            if status_resp.json().get("status") in ("completed", "exported"):
                break
        time.sleep(2)
    else:
        raise RuntimeError("Export snapshot did not complete within 40 seconds.")

    dl_resp = http.request(
        f"/api/projects/{project_id}/exports/{export_id}/download",
        method="GET",
        params={"exportType": "JSON"},
    )
    if dl_resp.status_code != 200:
        raise RuntimeError(
            f"Export download failed: {dl_resp.status_code} {dl_resp.text[:200]}"
        )

    return dl_resp.json() if isinstance(dl_resp.json(), list) else []


def _ls_result_to_fo_detection(result: dict, img_w: int, img_h: int):
    """
    Convert a single Label Studio rectanglelabels result to a FiftyOne Detection.
    LS value coords are percentages [0,100]; FiftyOne expects normalized [0,1].
    """
    val = result.get("value", {})
    labels = val.get("rectanglelabels", [])
    if not labels:
        return None

    x = val.get("x", 0) / 100
    y = val.get("y", 0) / 100
    w = val.get("width", 0) / 100
    h = val.get("height", 0) / 100

    return fo.Detection(
        label=labels[0],
        bounding_box=[x, y, w, h],
    )


# Public MCP tools

@mcp.tool()
def get_labeling_backend() -> dict:
    """
    Check which annotation backends are configured in .env and return the active one.
    Used by the agent at Step 3 to decide whether to present a backend choice.

    Returns a dict with:
      - cvat_available: bool
      - ls_available: bool
      - active_backend: "cvat" | "label_studio" | "none"
      - message: human-readable summary
    """
    cvat_token = os.getenv("CVAT_ACCESS_TOKEN", "").strip()
    load_dotenv(override=True)
    ls_token   = os.getenv("LS_TOKEN", "").strip()

    cvat_ok = bool(cvat_token)
    ls_ok   = bool(ls_token)

    if cvat_ok and ls_ok:
        active  = "cvat"   # CVAT is default when both are configured
        message = (
            "Both CVAT and Label Studio credentials are configured. "
            "CVAT is the default. You can use either — which would you prefer?"
        )
    elif cvat_ok:
        active  = "cvat"
        message = "CVAT credentials found. Using CVAT for annotation."
    elif ls_ok:
        active  = "label_studio"
        message = "Label Studio credentials found. Using Label Studio for annotation."
    else:
        active  = "none"
        message = (
            "No annotation backend credentials found in .env. "
            "Please add CVAT_ACCESS_TOKEN or LS_TOKEN to your .env file."
        )

    return {
        "cvat_available":    cvat_ok,
        "ls_available":      ls_ok,
        "active_backend":    active,
        "message":           message,
    }


@mcp.tool()
def set_labeling_backend(backend: str) -> str:
    """
    Set the active annotation backend for this session.
    Validates that credentials for the chosen backend exist in .env.
    backend: "cvat" or "label_studio"
    """
    backend = backend.strip().lower()
    if backend not in ("cvat", "label_studio"):
        return f"Invalid backend '{backend}'. Choose 'cvat' or 'label_studio'."

    if backend == "cvat" and not os.getenv("CVAT_ACCESS_TOKEN", "").strip():
        return (
            "LS_BACKEND_ERROR: CVAT_ACCESS_TOKEN is not set in .env. "
            "Please add it before selecting CVAT as your backend."
        )
    if backend == "label_studio" and not os.getenv("LS_TOKEN", "").strip():
        return (
            "LS_BACKEND_ERROR: LS_TOKEN is not set in .env. "
            "Please add it before selecting Label Studio as your backend."
        )

    return f"Labeling backend set to '{backend}'."


@mcp.tool()
def export_to_label_studio(
    dataset_name: str,
    with_predictions: bool = False,
    classes: list = None,
) -> str:
    """
    Export a FiftyOne dataset to Label Studio for annotation.

    Manual path (with_predictions=False): uploads images only.
    Auto path (with_predictions=True): uploads images and attaches model
    predictions as editable annotations for review and correction.

    Creates a new Label Studio project per dataset and saves project_id +
    task_ids to output/ls_tasks.json for later import.
    """
    if not _get_ls_token():
        return "LS_TOKEN not set in .env"

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
        client = _get_client()
        http   = _get_http(client)

        schema      = dataset.get_field_schema()
        pred_fields = [f for f in schema if f.startswith("pred_od_")]
        label_field = None

        # Normalize classes: LLM may pass "car, bus" (str), ["car","bus"] (list), or None.
        if isinstance(classes, str):
            manual_classes = [c.strip() for c in classes.split(",") if c.strip()]
        elif isinstance(classes, list):
            manual_classes = [str(c).strip() for c in classes if str(c).strip()]
        else:
            manual_classes = []

        logging.warning(f"[LS] export_to_label_studio called: dataset={dataset_name}, "
                       f"with_predictions={with_predictions}, raw_classes={classes}, "
                       f"resolved_manual_classes={manual_classes}")
        classes = []

        if manual_classes and not with_predictions:
            classes = manual_classes
            logging.warning(f"[LS] Manual path — using provided classes: {classes}")

        if with_predictions:
            if not pred_fields:
                # Wait up to 20s for the prediction field to appear after inference.
                for _ in range(10):
                    time.sleep(2)
                    dataset.reload()
                    schema      = dataset.get_field_schema()
                    pred_fields = [f for f in schema if f.startswith("pred_od_")]
                    if pred_fields:
                        break
                else:
                    return (
                        f"No prediction field found on dataset '{dataset_name}' "
                        f"after 20 seconds. Inference may not have completed. "
                        f"Please check auto-labeling logs."
                    )

            label_field = (
                "predictions" if "predictions" in schema
                else "ground_truth" if "ground_truth" in schema
                else pred_fields[0]
            )
            classes = dataset.distinct(f"{label_field}.detections.label")
            logging.info(f"[LS] Using label field: {label_field}, classes: {classes}")
        elif not classes:
            classes = []

        label_config = _build_label_config(classes)
        logging.warning(f"[LS] Creating project with label_config:\n{label_config}")
        project = client.projects.create(
            title=dataset_name,
            label_config=label_config,
        )
        project_id = project.id
        logging.warning(f"[LS] Created project {project_id} for dataset '{dataset_name}'")

        # PATCH label config separately — projects.create() sometimes ignores it
        # on HumanSignal cloud, so this guarantees labels appear in the UI.
        if classes:
            patch_resp = http.request(
                f"/api/projects/{project_id}",
                method="PATCH",
                json={"label_config": label_config},
            )
            logging.warning(f"[LS] Label config PATCH status: {patch_resp.status_code}")

        samples = list(dataset)
        image_paths = [Path(s.filepath) for s in samples]

        # --- Upload (concurrent, one POST per image) ----------------------
        fname_to_task_id = _upload_images_concurrent(http, project_id, image_paths)
        all_task_ids = list(fname_to_task_id.values())

        # --- Attach predictions (auto path only) --------------------------
        if with_predictions and label_field and fname_to_task_id:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            samples_with_dets = [
                (Path(s.filepath).name, s[label_field])
                for s in samples
                if s.get_field(label_field) is not None
                and s[label_field] is not None
                and s[label_field].detections
            ]

            t_ann = time.time()
            ann_ok = ann_skip = 0

            def _attach_one(fname, dets):
                task_id = fname_to_task_id.get(fname)
                if not task_id:
                    return False
                _attach_as_annotations(http, task_id, dets.detections, label_field)
                return True

            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {
                    pool.submit(_attach_one, fname, dets): fname
                    for fname, dets in samples_with_dets
                }
                for future in as_completed(futures):
                    if future.result():
                        ann_ok += 1
                    else:
                        ann_skip += 1

            logging.warning(
                f"[LS] Annotations attached: {ann_ok} ok / {ann_skip} skipped "
                f"in {time.time() - t_ann:.1f}s"
            )

        registry = _load_registry()
        registry[dataset_name] = {
            "project_id":      project_id,
            "task_ids":        all_task_ids,
            "with_predictions": with_predictions,
            "label_field":     label_field,
        }
        _save_registry(registry)

        msg = (
            f"Dataset '{dataset_name}' exported to Label Studio.\n"
            f"Project ID : {project_id}\n"
            f"Images     : {len(image_paths)}\n"
            f"Tasks      : {len(all_task_ids)}\n"
            f"Open in LS : {LS_URL}/projects/{project_id}/"
        )
        if with_predictions and label_field:
            msg += f"\nPredictions attached with classes: {classes}"

        logging.info(msg)
        return msg

    except Exception as e:
        tb = traceback.format_exc()
        logging.warning(f"[LS] export_to_label_studio failed:\n{tb}")

        err = str(e)
        if "LS_TOKEN" in err or "401" in err or "Invalid token" in err:
            return (
                "LS_AUTH_ERROR: Label Studio authentication failed. "
                "Please check LS_TOKEN in .env."
            )
        if "ConnectionError" in tb or "ConnectTimeout" in tb:
            return (
                "LS_CONNECTION_ERROR: Could not reach Label Studio at "
                f"{LS_URL}. Please check LS_URL in .env."
            )
        return f"Label Studio export failed: {err}"


@mcp.tool()
def import_from_label_studio(dataset_name: str) -> str:
    """
    Download completed annotations from Label Studio for a previously exported
    dataset and save them as a new FiftyOne dataset named <dataset_name>_labeled
    with labels stored in the 'ground_truth' field.
    """
    if not _get_ls_token():
        return "LS_TOKEN not set in .env"

    registry = _load_registry()
    if dataset_name not in registry:
        return (
            f"No Label Studio project found for dataset '{dataset_name}'. "
            f"Please export it first."
        )

    project_id  = registry[dataset_name]["project_id"]
    labeled_name = f"{dataset_name}_labeled"

    try:
        client = _get_client()
        http   = _get_http(client)

        exported_tasks = _export_snapshot(http, project_id)

        if not exported_tasks:
            return (
                f"LS_NO_ANNOTATIONS: No tasks found in Label Studio project {project_id}. "
                f"Please annotate the images in Label Studio first, then try importing again."
            )

        tasks_with_annotations = [t for t in exported_tasks if t.get("annotations")]
        if not tasks_with_annotations:
            return (
                f"LS_NO_ANNOTATIONS: No annotations have been submitted yet in "
                f"Label Studio project {project_id} ({len(exported_tasks)} tasks found, "
                f"0 annotated). Please open Label Studio, draw your boxes, click Submit "
                f"on each task, then come back and try again."
            )

        original_dataset = fo.load_dataset(dataset_name)
        path_map = {
            Path(s.filepath).name: s.filepath
            for s in original_dataset
        }

        if labeled_name in fo.list_datasets():
            fo.delete_dataset(labeled_name)

        labeled_dataset = fo.Dataset(name=labeled_name, persistent=True)
        samples_created = 0
        samples_skipped = 0

        for task in exported_tasks:
            annotations = task.get("annotations", [])
            if not annotations:
                continue

            data       = task.get("data", {})
            # LS stores the image URL in task data; extract the filename.
            image_ref  = data.get("image", "") or data.get("$undefined$", "")
            filename   = image_ref.split("/")[-1].split("?")[0]  # strip query params

            # LS prepends a random UUID to uploaded filenames (e.g. "8c7230d6-000001.jpg").
            # Try a direct match first, then strip the UUID prefix.
            filepath = path_map.get(filename)
            if not filepath:
                parts = filename.split("-", 1)
                if len(parts) == 2:
                    filepath = path_map.get(parts[1])

            if not filepath:
                logging.warning(f"[LS] Could not match task image '{filename}' to local file")
                samples_skipped += 1
                continue

            # Use the first completed annotation (most recent human review).
            ann     = annotations[0]
            results = ann.get("result", [])

            fo_detections = []
            for r in results:
                if r.get("type") != "rectanglelabels":
                    continue
                det = _ls_result_to_fo_detection(r, img_w=1, img_h=1)
                if det:
                    fo_detections.append(det)

            sample = fo.Sample(filepath=filepath)
            if fo_detections:
                sample["ground_truth"] = fo.Detections(detections=fo_detections)

            labeled_dataset.add_sample(sample)
            samples_created += 1

        labeled_dataset.save()

        if samples_created == 0:
            return (
                f"LS_NO_ANNOTATIONS: Annotations were found in Label Studio but none "
                f"could be matched to local image files "
                f"({samples_skipped} task(s) skipped — image filenames did not match). "
                f"Please check that the exported images match the project tasks."
            )

        for _ in range(5):
            if labeled_name in fo.list_datasets():
                verify = fo.load_dataset(labeled_name)
                if len(verify) > 0:
                    break
            time.sleep(2)
        else:
            return (
                f"Import appeared to succeed but dataset '{labeled_name}' "
                f"could not be verified in FiftyOne. Please try again."
            )

        msg = (
            f"Annotations imported from Label Studio project {project_id}.\n"
            f"Dataset '{labeled_name}' created with {samples_created} samples.\n"
            f"Labels saved as 'ground_truth' field."
        )
        if samples_skipped:
            msg += f"\n{samples_skipped} task(s) skipped (image path not matched)."

        logging.info(msg)
        return msg

    except Exception as e:
        return f"Label Studio import failed: {e}\n{traceback.format_exc()}"