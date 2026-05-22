# mcptools/data_ingest.py
from mcptools import mcp
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, List, Tuple, Awaitable, Callable
import uuid


ROOT_DIR = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = Path(os.environ.get("MDE_UPLOAD_ROOT", ROOT_DIR / "mde_uploads")).resolve()
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
# Type for streaming emit callbacks
EmitFn = Callable[[dict], Awaitable[None]]

async def _emit(maybe_emit: Optional[EmitFn], payload: dict):
    """Emit to provided callback or MCP event bus."""
    if maybe_emit:
        await maybe_emit(payload)
    else:
        await mcp.yield_event(payload)

# Paths
sys.path.append(str(ROOT_DIR))
CONFIG_PATH = ROOT_DIR / "config" / "config.py"
MAIN_PATH = ROOT_DIR / "main.py"
LOG_DIR = ROOT_DIR / "output" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----- helpers -----
def _extract_zip(zip_path: str) -> str:
    tmp_root = Path(tempfile.mkdtemp(prefix="ingest_zip_"))
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_root)
    entries = [p for p in tmp_root.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return str(entries[0])
    return str(tmp_root)

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def _set_selected_workflow(src: str) -> str:
    pattern = r'^\s*SELECTED_WORKFLOW\s*=\s*\[.*?\]\s*$'
    repl = 'SELECTED_WORKFLOW = ["data_ingest"]'
    if re.search(pattern, src, flags=re.MULTILINE | re.DOTALL):
        return re.sub(pattern, repl, src, flags=re.MULTILINE | re.DOTALL)
    return repl + "\n" + src

def _patch_data_ingest_block(
    src: str,
    dataset_dir: str,
    fps: int,
    split_percentages: list[float],
    dataset_name: Optional[str],
    annotation_format: Optional[str],
) -> str:
    def upsert_key(block: str, key: str, value_src: str) -> str:
        patt = rf'(["\']{re.escape(key)}["\']\s*:\s*)(.*?)(,?\s*\n)'
        if re.search(patt, block, flags=re.DOTALL):
            def _repl(m: re.Match) -> str:
                return f"{m.group(1)}{value_src},\n"
            return re.sub(patt, _repl, block, flags=re.DOTALL)
        return re.sub(r'\}\s*$', f'    "{key}": {value_src},\n' + "}", block)

    fps_val = int(fps)
    splits_val = json.dumps([float(x) for x in split_percentages])

    start = src.find('"data_ingest": {')
    if start == -1:
        start = src.find("'data_ingest': {")

    if start == -1:
        m = re.search(r'WORKFLOWS\s*=\s*\{', src)
        block_lines = [
            '    "data_ingest": {',
            f'        "dataset_dir": {repr(dataset_dir)},',
            f'        "fps": {fps_val},',
            f'        "split_percentages": {splits_val},',
        ]
        if dataset_name:
            block_lines.append(f'        "dataset_name": {repr(dataset_name)},')
        if annotation_format:
            block_lines.append(f'        "annotation_format": {repr(annotation_format)},')
        block_lines.append('    },\n')
        block = "\n".join(block_lines)
        if not m:
            inject = "\n".join(["", "WORKFLOWS = {", block, "}", ""])
            return src + inject
        insert_at = src.find("{", m.end())
        if insert_at == -1:
            insert_at = m.end()
        return src[:insert_at + 1] + "\n" + block + src[insert_at + 1:]

    i, brace, end = start, 0, None
    while i < len(src):
        ch = src[i]
        if ch == "{": brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                end = i
                break
        i += 1
    if end is None:
        return src

    block_src = src[start:end + 1]
    block_new = block_src
    block_new = upsert_key(block_new, "dataset_dir", repr(dataset_dir))
    block_new = upsert_key(block_new, "fps", str(fps_val))
    block_new = upsert_key(block_new, "split_percentages", splits_val)
    if dataset_name:
        block_new = upsert_key(block_new, "dataset_name", repr(dataset_name))
    if annotation_format:
        block_new = upsert_key(block_new, "annotation_format", repr(annotation_format))
    return src[:start] + block_new + src[end + 1:]

def _parse_name_samples(text: str) -> Tuple[Optional[str], Optional[int]]:
    name, samples = None, None
    m = re.search(r'(Created dataset|Dataset(?:\s+name)?):?\s*([A-Za-z0-9_\-\.]+)', text, flags=re.IGNORECASE)
    if m: name = m.group(2)
    m2 = re.search(r'(?:samples\s*=\s*|)(\d+)\s+samples', text, flags=re.IGNORECASE)
    if m2:
        try: samples = int(m2.group(1))
        except: pass
    if samples is None:
        m3 = re.search(r'samples\s*=\s*(\d+)', text, flags=re.IGNORECASE)
        if m3:
            try: samples = int(m3.group(1))
            except: pass
    return name, samples

async def _stream_subprocess(proc: asyncio.subprocess.Process, on_line):
    async def _reader(stream, source):
        while True:
            line = await stream.readline()
            if not line:
                break
            await on_line(source, line.decode("utf-8", errors="ignore"))
    await asyncio.gather(_reader(proc.stdout, "stdout"), _reader(proc.stderr, "stderr"))

# ---- CORE STREAMING IMPLEMENTATION ----
async def _run_data_ingest_streaming_core(
    zip_path: Optional[str] = None,
    source_path: Optional[str] = None,
    fps: int = 2,
    split_percentages: Optional[List[float]] = None,
    dataset_prefix: Optional[str] = None,
    annotation_format: Optional[str] = None,
    emit: Optional[EmitFn] = None,
):
    if not zip_path and not source_path:
        await _emit(emit, {"type": "error", "data": {"msg": "Provide zip_path or source_path"}})
        return

    extracted_tmp = None
    backup_path = CONFIG_PATH.with_suffix(".py.bak")
    combined_lines: List[str] = []
    ds_name, samples, final_rc = None, None, None

    try:
        dataset_dir = _extract_zip(zip_path) if zip_path else source_path
        if zip_path:
            # we extracted under /tmp -> move to persistent, shared path
            extracted_tmp = Path(dataset_dir)
            persistent_dir = UPLOAD_ROOT / f"job_{uuid.uuid4().hex}"
            persistent_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted_tmp), str(persistent_dir))
            dataset_dir = str(persistent_dir)
            extracted_tmp = None  # moved; nothing left to delete
        await _emit(emit, {"type": "log", "data": {"msg": f"Prepared dataset_dir: {dataset_dir}"}})

        original = _read(CONFIG_PATH)
        _write(backup_path, original)
        patched = _set_selected_workflow(original)
        patched = _patch_data_ingest_block(
            patched, dataset_dir, int(fps), split_percentages or [0.7, 0.15, 0.15],
            dataset_prefix, annotation_format
        )
        _write(CONFIG_PATH, patched)
        await _emit(emit, {"type": "log", "data": {"msg": "Config patched for data_ingest"}})

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(MAIN_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(MAIN_PATH.parent)
        )

        async def on_line(source: str, line: str):
            nonlocal ds_name, samples
            combined_lines.append(f"[{source}] {line}")
            text = line.strip()
            # Target the specific ingestion line directly
            ingest_match = re.search(r"Ingesting dataset:\s*([A-Za-z0-9_\-\.]+)", text)
            if ingest_match:
                ds_name = ingest_match.group(1)

            _, s = _parse_name_samples(text)
            if s: samples = s
            await _emit(emit, {"type": "log", "data": {"msg": text}})

        await _stream_subprocess(proc, on_line)
        final_rc = await proc.wait()

        log_path = LOG_DIR / "last_data_ingest_log.txt"
        _write(log_path, "".join(combined_lines))

        try:
            if backup_path.exists():
                _write(CONFIG_PATH, _read(backup_path))
                backup_path.unlink(missing_ok=True)
            else:
                logging.warning("[INGEST] Backup not found — config.py not restored")
        except Exception as restore_err:
            logging.warning(f"[INGEST] Failed to restore config.py: {restore_err}")

        if not ds_name:
            ds_name = dataset_prefix or "custom_dataset"

        if final_rc == 0:
            await _emit(emit, {
                "type": "agent",
                "data": {
                    "msg": (
                        f"✅ Ingestion complete. Dataset **{ds_name}**"
                    ),
                    "dataset": ds_name,
                    "samples": samples,
                },
            })
            await _emit(emit, {"type": "done", "data": {"ok": True}})
        else:
            await _emit(emit, {"type": "error", "data": {"msg": f"data_ingest failed with exit code {final_rc}"}})

    except Exception as e:
        logging.exception("run_data_ingest_streaming failed")
        await _emit(emit, {"type": "error", "data": {"msg": str(e)}})
        try:
            if backup_path.exists():
                _write(CONFIG_PATH, _read(backup_path))
                backup_path.unlink(missing_ok=True)
        except: pass
    finally:
        if extracted_tmp and extracted_tmp.exists():
            shutil.rmtree(extracted_tmp, ignore_errors=True)

# ---- MCP TOOL WRAPPER (no emit in signature) ----
@mcp.tool(name="run_data_ingest_streaming", description="Streaming: run data_ingest via main.py and emit live events")
async def run_data_ingest_streaming(
    zip_path: Optional[str] = None,
    source_path: Optional[str] = None,
    fps: int = 2,
    split_percentages: Optional[List[float]] = None,
    dataset_prefix: Optional[str] = None,
    annotation_format: Optional[str] = None,
):
    await _run_data_ingest_streaming_core(
        zip_path=zip_path,
        source_path=source_path,
        fps=fps,
        split_percentages=split_percentages,
        dataset_prefix=dataset_prefix,
        annotation_format=annotation_format,
        emit=None  # MCP uses its own event bus
    )

# ---- NON-STREAMING TOOL ----
async def _run_main_and_capture() -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(MAIN_PATH),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(MAIN_PATH.parent)
    )
    out, err = await proc.communicate()
    return (
        out.decode("utf-8", errors="ignore") if out else "",
        err.decode("utf-8", errors="ignore") if err else "",
        proc.returncode,
    )

@mcp.tool(name="run_data_ingest", description="Run data_ingest via main.py after patching config; returns JSON with logs")
async def run_data_ingest_tool(
    zip_path: Optional[str] = None,
    source_path: Optional[str] = None,
    fps: int = 2,
    split_percentages: Optional[List[float]] = None,
    dataset_prefix: Optional[str] = None,
    annotation_format: Optional[str] = None,
):

    if not zip_path and not source_path:
        return json.dumps({"status": "error", "message": "Provide zip_path or source_path"})

    extracted_tmp = None
    try:
        dataset_dir = _extract_zip(zip_path) if zip_path else source_path
        if zip_path:
            extracted_tmp = Path(dataset_dir)
            persistent_dir = UPLOAD_ROOT / f"job_{uuid.uuid4().hex}"
            persistent_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted_tmp), str(persistent_dir))
            dataset_dir = str(persistent_dir)
            extracted_tmp = None

        original = _read(CONFIG_PATH)
        backup_path = CONFIG_PATH.with_suffix(".py.bak")
        _write(backup_path, original)

        patched = _set_selected_workflow(original)
        patched = _patch_data_ingest_block(
            patched,
            dataset_dir=dataset_dir,
            fps=int(fps),
            split_percentages=split_percentages or [0.7, 0.15, 0.15],
            dataset_name=dataset_prefix,
            annotation_format=annotation_format,
        )
        _write(CONFIG_PATH, patched)

        stdout_text, stderr_text, rc = await _run_main_and_capture()
        combined = ""
        if stdout_text:
            combined += "=== STDOUT ===\n" + stdout_text + "\n"
        if stderr_text:
            combined += "=== STDERR ===\n" + stderr_text + "\n"

        log_path = LOG_DIR / "last_data_ingest_log.txt"
        _write(log_path, combined)

        name, samples = _parse_name_samples(combined)
        if not name:
            name = dataset_prefix or "custom_dataset"

        _write(CONFIG_PATH, original)
        try:
            backup_path.unlink(missing_ok=True)
        except Exception:
            pass

        tail = combined[-5000:] if len(combined) > 5000 else combined
        status = "ok" if rc == 0 else "error"
        return json.dumps({
            "status": status,
            "name": name,
            "samples": samples,
            "returncode": rc,
            "log_path": str(log_path),
            "log_tail": tail,
        })

    except Exception as e:
        logging.exception("run_data_ingest failed")
        try:
            if CONFIG_PATH.with_suffix(".py.bak").exists():
                _write(CONFIG_PATH, _read(CONFIG_PATH.with_suffix(".py.bak")))
                CONFIG_PATH.with_suffix(".py.bak").unlink(missing_ok=True)
        except Exception:
            pass
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        if extracted_tmp and extracted_tmp.exists():
            try:
                shutil.rmtree(extracted_tmp, ignore_errors=True)
            except Exception:
                pass