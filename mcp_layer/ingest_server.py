# ui_ingest_server.py
import asyncio
import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcptools.data_ingest import _run_data_ingest_streaming_core
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Holds queues for jobs: job_id -> asyncio.Queue
JOB_LOGS: dict[str, asyncio.Queue[str | None]] = {}


async def run_ingest_and_stream(job_id, zip_path, fps, train, val, test):
    """
    Run the ingestion and push events to the JOB_LOGS queue for this job_id.
    """
    q = JOB_LOGS[job_id]

    async def emit(ev: dict):
        # Each event is JSON-encoded for SSE
        await q.put(json.dumps(ev))

    try:
        await _run_data_ingest_streaming_core(
            zip_path=zip_path,
            fps=fps,
            split_percentages=[train, val, test],
            dataset_prefix="custom_dataset",
            annotation_format="auto",
            emit=emit
        )
    except Exception as e:
        await q.put(json.dumps({"type": "error", "data": {"msg": str(e)}}))
    finally:
        # Signal the end of the stream
        await q.put(None)


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    fps: int = Form(...),
    train: float = Form(...),
    val: float = Form(...),
    test: float = Form(...)
):
    """
    Upload a zip file for ingestion. Starts ingestion in background and returns job_id.
    """
    tmp_path = Path("/tmp") / f"{uuid.uuid4()}_{file.filename}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    job_id = str(uuid.uuid4())
    JOB_LOGS[job_id] = asyncio.Queue()

    asyncio.create_task(run_ingest_and_stream(job_id, str(tmp_path), fps, train, val, test))

    return {"status": "started", "job_id": job_id}


@app.get("/ingest/stream/{job_id}")
async def ingest_stream(job_id: str):
    """
    Stream ingestion logs/events to the client as Server-Sent Events (SSE).
    """
    q = JOB_LOGS.get(job_id)
    if q is None:
        return StreamingResponse(iter(["event: done\ndata: {}\n\n"]), media_type="text/event-stream")

    async def event_generator():
        # Optional SSE retry instruction for client reconnects
        yield "retry: 1000\n\n"
        while True:
            msg = await q.get()
            if msg is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield f"event: log\ndata: {msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
