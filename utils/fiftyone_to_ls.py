import requests as _requests
import os
import time
import copy

class _LSUsers:
    def __init__(self, client):
        self._client = client
    def me(self):
        return self._client.make_request("GET", "/api/current-user").json()

class _LSProject:
    def __init__(self, client, data):
        self._client      = client
        self._data        = data
        self.id           = data.get("id")
        self.label_config = data.get("label_config", "")
        self.title        = data.get("title", "")
        self.params       = data

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"_LSProject has no attribute '{name}'")

    def import_tasks(self, tasks):
        return self._client.make_request(
            "POST", f"/api/projects/{self.id}/import", json=tasks
        ).json()

    def reimport(self, file_upload_ids):
        import json as _json
        payload = _json.dumps({
            "file_upload_ids": file_upload_ids,
            "files_as_tasks_list": False,
        })

        self._client.headers.update({"Content-Type": "application/json"})
        return self._client.make_request(
            "POST", f"/api/projects/{self.id}/reimport", data=payload
        ).json()

    def get_tasks(self, selected_ids=None, only_ids=False):
        """
        line 380: project.get_tasks(selected_ids=task_ids)
        line after upload: project.get_tasks(only_ids=True)
        """
        if selected_ids:
            params = [("project", self.id)] + [("ids[]", tid) for tid in selected_ids]
            r = self._client.make_request("GET", "/api/tasks", params=params).json()
        else:
            r = self._client.make_request(
                "GET", f"/api/projects/{self.id}/tasks"
            ).json()

        if isinstance(r, list):
            tasks = r
        elif "tasks" in r:
            tasks = r["tasks"]
        elif "results" in r:
            tasks = r["results"]
        else:
            tasks = []

        if only_ids:
            return [t["id"] for t in tasks]
        return tasks

    def get_task(self, task_id):
        return self._client.make_request("GET", f"/api/tasks/{task_id}").json()

    def create_prediction(self, task, predictions):
        """line 536: single prediction"""
        return self._client.make_request(
            "POST", "/api/predictions",
            json={"task": task, "result": predictions},
        ).json()

    def create_predictions(self, predictions):
        """line 374: bulk predictions"""
        results = []
        for pred in predictions:
            r = self._client.make_request(
                "POST", "/api/predictions", json=pred
            ).json()
            results.append(r)
        return results

    def connect_local_import_storage(self, local_store_path, **kwargs):
        try:
            payload = {
                "path": local_store_path,
                "project": self.id,
                "title": f"FiftyOne local — project {self.id}",
                "use_blob_urls": True,
                **kwargs,
            }
            resp = self._client.make_request(
                "POST", "/api/storages/localfiles/", json=payload
            ).json()
            storage_id = resp.get("id")
            if storage_id:
                self._client.make_request(
                    "POST", f"/api/storages/localfiles/{storage_id}/sync"
                )
            return resp
        except Exception as e:
            print(f"connect_local_import_storage unavailable on cloud LS: {e}")
            return {}

    def set_params(self, **kwargs):
        resp = self._client.make_request(
            "PATCH", f"/api/projects/{self.id}", json=kwargs
        ).json()
        if isinstance(resp, dict):
            self._data.update(resp)
            self.params = self._data
        return resp

    def update(self, **kwargs):
        return self.set_params(**kwargs)

    def delete_tasks(self, task_ids):
        return self._client.make_request(
            "POST", f"/api/projects/{self.id}/tasks/bulk-delete",
            json={"ids": task_ids},
        ).json()

    def delete(self):
        self._client.make_request("DELETE", f"/api/projects/{self.id}")

class _BearerClient:
    CHUNK_SIZE = 3

    def __init__(self, url, api_key, *args, **kwargs):
        self.url       = url.rstrip("/")
        self._pat      = api_key
        self._access   = None
        self._token_ts = 0
        self._session  = _requests.Session()
        self.users     = _LSUsers(self)
        self._refresh_access_token()
        self.versions  = self.get_versions()
        print(f"Connected — Label Studio {self.versions.get('version', '?')}")

    def _refresh_access_token(self):
        resp = _requests.post(
            f"{self.url}/api/token/refresh",
            headers={"Content-Type": "application/json"},
            json={"refresh": self._pat},
        )
        resp.raise_for_status()
        data = resp.json()
        self._access = (
            data.get("access")
            or data.get("token")
            or data.get("access_token")
        )
        if not self._access:
            raise ValueError(f"Token refresh failed: {data}")
        self._token_ts = time.time()
        self._session.headers.clear()
        self._session.headers.update({
            "Authorization": f"Bearer {self._access}",
        })

    def _get_token(self):
        if time.time() - self._token_ts > 240:
            self._refresh_access_token()
        return self._access

    @property
    def headers(self):
        """line 352: self._client.headers.update({"Content-Type": ...})"""
        return self._session.headers

    def _raw_request(self, method, url, **kwargs):
        """Single HTTP call with one auto-retry on 401."""
        self._get_token()
        resp = self._session.request(method, url, **kwargs)
        if resp.status_code == 401:
            self._refresh_access_token()
            resp = self._session.request(method, url, **kwargs)
        return resp

    def make_request(self, method, path, **kwargs):
        url = f"{self.url}{path}"

        files = kwargs.get("files")
        if (
            files is not None
            and "/import" in path
            and method.upper() == "POST"
            and len(files) > self.CHUNK_SIZE
        ):
            return self._chunked_file_upload(url, files, kwargs)

        json_body = kwargs.get("json")
        if (
            json_body is not None
            and isinstance(json_body, list)
            and "/import" in path
            and method.upper() == "POST"
            and len(json_body) > self.CHUNK_SIZE
        ):
            return self._chunked_json_upload(url, json_body, kwargs)

        resp = self._raw_request(method, url, **kwargs)
        if resp.status_code == 413:
            raise RuntimeError(
                f"413 on single-item upload to {path}. "
                f"Image may be too large for the server limit."
            )
        resp.raise_for_status()
        return resp

    def _chunked_file_upload(self, url, files, original_kwargs):
        """
        Upload files in small batches. Each batch returns file_upload_ids.
        We accumulate all ids and return a fake response matching what
        labelstudio.py expects: resp.json()["file_upload_ids"]
        """
        chunk_size = self.CHUNK_SIZE
        all_upload_ids = []
        total   = len(files)
        n_chunks = (total + chunk_size - 1) // chunk_size
        print(f"   Chunking {total} files → {n_chunks} batches of ≤{chunk_size}")

        for i in range(0, total, chunk_size):
            chunk = files[i : i + chunk_size]
            kw    = {k: v for k, v in original_kwargs.items() if k != "files"}
            kw["files"] = chunk

            max_retries = 8
            backoff = 5
            for attempt in range(max_retries):
                resp = self._raw_request("POST", url, **kw)

                if resp.status_code == 413 and chunk_size > 1:
                    self.CHUNK_SIZE = max(1, chunk_size // 2)
                    print(f"   413 — reducing chunk size to {self.CHUNK_SIZE} and retrying")
                    return self._chunked_file_upload(url, files, original_kwargs)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", backoff))
                    wait = max(retry_after, backoff)
                    print(f"   429 Too Many Requests — waiting {wait}s before retry (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    backoff = min(backoff * 2, 120)
                    continue

                break
            else:
                resp.raise_for_status()

            resp.raise_for_status()
            ids = resp.json().get("file_upload_ids", [])
            all_upload_ids.extend(ids)
            print(f"   ✓ Batch {i // chunk_size + 1}/{n_chunks} — {len(ids)} files uploaded")

        return _FakeResponse({"file_upload_ids": all_upload_ids})

    def _chunked_json_upload(self, url, tasks, original_kwargs):
        """Upload URL-based task lists in chunks."""
        chunk_size = self.CHUNK_SIZE
        last_resp  = None
        total      = len(tasks)
        n_chunks   = (total + chunk_size - 1) // chunk_size
        print(f"   Chunking {total} JSON tasks → {n_chunks} batches of ≤{chunk_size}")

        for i in range(0, total, chunk_size):
            chunk = tasks[i : i + chunk_size]
            kw    = {k: v for k, v in original_kwargs.items() if k != "json"}
            kw["json"] = chunk

            resp = self._raw_request("POST", url, **kw)

            if resp.status_code == 413 and chunk_size > 1:
                self.CHUNK_SIZE = max(1, chunk_size // 2)
                print(f"   413 — reducing chunk size to {self.CHUNK_SIZE} and retrying")
                return self._chunked_json_upload(url, tasks, original_kwargs)

            resp.raise_for_status()
            last_resp = resp
            print(f"   ✓ Batch {i // chunk_size + 1}/{n_chunks} uploaded")

        return last_resp

    def get_versions(self):
        return self.make_request("GET", "/api/version").json()

    def check_connection(self):
        try:
            self.get_versions()
            return True
        except Exception:
            return False

    def list_projects(self):
        data    = self.make_request("GET", "/api/projects").json()
        results = data if isinstance(data, list) else data.get("results", [])
        return [_LSProject(self, p) for p in results]

    def start_project(self, **kwargs):
        data = self.make_request("POST", "/api/projects", json=kwargs).json()
        return _LSProject(self, data)

    def get_project(self, project_id):
        data = self.make_request("GET", f"/api/projects/{project_id}").json()
        return _LSProject(self, data)

    def get_tasks(self, project_id, **kwargs):
        r = self.make_request("GET", f"/api/tasks?project={project_id}").json()
        return r if isinstance(r, list) else r.get("tasks", [])

    def get_task(self, task_id):
        return self.make_request("GET", f"/api/tasks/{task_id}").json()

    def get_annotations(self, project_id):
        return self.make_request(
            "GET", f"/api/projects/{project_id}/export?exportType=JSON"
        ).json()

class _FakeResponse:
    """
    Mimics requests.Response for chunked uploads.
    labelstudio.py calls upload_resp.json()["file_upload_ids"] on the result.
    """
    def __init__(self, data):
        self._data       = data
        self.status_code = 200

    def json(self):
        return self._data

    def raise_for_status(self):
        pass

# ── Inject BEFORE fiftyone import ────────────────────────────────────────────
import label_studio_sdk
label_studio_sdk.Client = _BearerClient
print("Bearer client with chunked file upload injected")

import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

LS_URL = "https://app.humansignal.com"
LS_PAT = os.getenv("LABEL_STUDIO_API_KEY")

if not LS_PAT:
    raise ValueError("Set LABEL_STUDIO_API_KEY env var")

fo.annotation_config.backends["labelstudio"] = {
    "config_cls": "fiftyone.utils.labelstudio.LabelStudioBackendConfig",
    "url": LS_URL,
    "api_key": LS_PAT,
}

dataset = load_from_hub(
    "mcity-engineering/gameday_ds",
    dataset_name="gameday_ds",
    persistent=True,
    overwrite=True,
)
print(f"Dataset size: {len(dataset)}")

classes = dataset.distinct("ground_truth.detections.label")
print(f"Classes ({len(classes)}): {classes}")

anno_key = "mcity_ls1"

dataset.annotate(
    anno_key,
    backend="labelstudio",
    label_schema={
        "new_ground_truth": {
            "type": "detections",
            "classes": classes,
        },
    },
    url=LS_URL,
    api_key=LS_PAT,
    project_name="gameday_ds_annotation",
    launch_editor=False,
)

print(f"Done | anno_key: {anno_key}")