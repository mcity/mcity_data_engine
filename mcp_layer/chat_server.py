import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastmcp.client.transports import SSETransport

sys.path.append(os.path.dirname(__file__))

from chat_pipeline import ChatPipeline
from host_utils import resolve_host
from llm_clients import ClaudeClient, GeminiClient, GroqClient, OpenAIClient
from tool_schema import tools
from validate_workflow_state import LabelingBackend, AutoLabelingPhase, LabelingPath, WorkflowState

load_dotenv()

_LLM_PROVIDERS = {
    "openai": OpenAIClient,
    "groq": GroqClient,
    "gemini": GeminiClient,
    "claude": ClaudeClient,
    "anthropic": ClaudeClient,  # alias
}

llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
if llm_provider not in _LLM_PROVIDERS:
    logging.warning(
        f"[STARTUP] Unknown LLM_PROVIDER '{llm_provider}'. "
        f"Valid values: {list(_LLM_PROVIDERS)}. Falling back to 'openai'."
    )
    llm_provider = "openai"

def _reset_state_on_startup() -> None:
    """Reset persisted state on startup so each server launch begins clean."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from validate_workflow_state import WorkflowState
        WorkflowState().save()
        logging.warning("[STARTUP] Reset WORKFLOW_STATE to defaults.")
    except Exception as e:
        logging.warning(f"[STARTUP] Could not reset WORKFLOW_STATE: {e}")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _reset_state_on_startup()
    app.state.llm = _LLM_PROVIDERS[llm_provider]()
    host = resolve_host()
    app.state.mcp_transport = SSETransport(url=f"http://{host}:8000/sse")
    yield


app = FastAPI(lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = (
    Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"
).read_text()


def _attach_source(message: str, source: str | None) -> str:
    """Append a [source: ...] tag when the LLM supplied one, leave message untouched otherwise."""
    if source and source.strip():
        return f"{message.strip()}\n[source: {source.strip()}]"
    return message


def filter_tools_for_state(all_tools: list, state) -> list:
    """Return tools valid for the current step; fails open on None state or exceptions."""
    if state is None:
        return all_tools
    try:
        valid_names = state.valid_tool_names()
    except Exception:
        logging.warning("[FILTER] valid_tool_names() raised — returning full tool list")
        return all_tools
    if valid_names is None:
        return all_tools
    return [t for t in all_tools if t["function"]["name"] in valid_names]


def _build_state_hint(state=None) -> str:
    """Return SESSION_STATE string injected before each user message."""
    try:
        if state is None:
            state = WorkflowState.load()
        if not state.workflow_name:
            return ""
        parts = [f"workflow={state.workflow_name}"]
        if state.dataset_confirmed and state.dataset_name:
            parts.append(f"dataset={state.dataset_name}")
        else:
            parts.append(
                "dataset=not confirmed — "
                "NEXT STEP: wait for the user to name a dataset, then call set_selected_dataset immediately. "
                "Do NOT infer or reuse a dataset name from earlier in the conversation — "
                "the user must explicitly type a dataset name in their CURRENT message. "
                "Do NOT call switch_workflow or select_workflow again."
            )
        al = state.auto_labeling
        if al:
            if al.labeling_backend == LabelingBackend.BOTH:
                parts.append(
                    "backend=AWAITING_CHOICE — user must choose annotation backend. "
                    "Classify intent and act: "
                    "user names a backend preference → call set_labeling_backend(backend=...) immediately; "
                    "user provides a new dataset name → call set_selected_dataset; "
                    "user provides both dataset and backend → call set_selected_dataset then set_labeling_backend."
                )
            elif al.labeling_backend and not al.labeling_path:
                parts.append(
                    f"LABELING_BACKEND: {al.labeling_backend} — already confirmed. "
                    f"NEXT STEP: present Manual vs Auto Labeling options if the user has not yet chosen. "
                    f"user wants a different backend → call set_labeling_backend with the new backend; "
                    f"user wants a different dataset → call set_selected_dataset; "
                    f"user chose manual labeling → call set_labeling_path('manual'); "
                    f"user chose auto labeling → call set_labeling_path('auto')."
                )
            elif al.labeling_backend:
                parts.append(f"backend={al.labeling_backend}")
            if al.labeling_path:
                if al.labeling_path == LabelingPath.MANUAL and not al.manual_classes:
                    export_fn = (
                        "export_to_label_studio"
                        if al.labeling_backend == LabelingBackend.LABEL_STUDIO
                        else "export_to_cvat"
                    )
                    parts.append(
                        f"labeling_path=manual — awaiting annotation class names. "
                        f"When user provides class names → call "
                        f"{export_fn}(dataset_name='{state.dataset_name or '?'}', classes=[...])."
                    )
                else:
                    parts.append(f"labeling_path={al.labeling_path}")
            if (al.manual_classes and not al.cvat_task_id and not al.ls_task_ids
                    and not al.phase):
                classes_s = ", ".join(al.manual_classes)
                if al.export_confirmed:
                    parts.append(
                        f"manual_classes=[{classes_s}], export_confirmed=True — "
                        f"call the export tool immediately with these classes."
                    )
                else:
                    parts.append(
                        f"manual_classes=[{classes_s}] — classes provided, awaiting export confirmation. "
                        f"When user confirms, call confirm_export() then the export tool with these classes."
                    )
            if al.phase in (AutoLabelingPhase.ANNOTATING, AutoLabelingPhase.TRAINING, AutoLabelingPhase.COMPLETE):
                action = {
                    AutoLabelingPhase.ANNOTATING: "export complete — awaiting annotation",
                    AutoLabelingPhase.TRAINING:   "auto-labeling complete — awaiting import",
                    AutoLabelingPhase.COMPLETE:   "labels imported — workflow complete",
                }[al.phase]
                parts.append(
                    f"phase={al.phase} ({action}). "
                    f"Parameters are LOCKED — do not reconfigure. "
                    f"If the user asks to change a parameter: tell them the workflow is locked. "
                    f"If they want to discard all progress and restart from dataset selection: "
                    f"call switch_workflow(workflow_name='{state.workflow_name}') to reset all state."
                )
            if al.models_listed and not al.model_configured:
                parts.append(
                    "models_listed=True — user has seen the model list. "
                    "When the user names a model (e.g. 'rfdetr_2xlarge', 'yolo11n', 'facebook/detr-resnet-50') "
                    "→ call configure_auto_labeling(selected_source=<infer from name>, selected_model=<exact name>) immediately. "
                    "Do NOT call set_selected_dataset or set_labeling_backend when the user is naming a model. "
                    "If user describes their use case or asks for advice, use send_reply to recommend options "
                    "and end with 'Which model would you like to use?' — then wait for their reply."
                )
            if al.model_configured:
                if not al.auto_labeling_complete:
                    if al.run_awaiting_confirmation and not al.run_confirmed:
                        parts.append(
                            "model=configured, run summary shown — awaiting user confirmation. "
                            "user confirms (yes, proceed, go ahead, etc.) → call confirm_run; "
                            "user requests changes → call set_auto_labeling_hyperparams with ONLY changed values; "
                            "to change the model → call configure_auto_labeling immediately "
                            "(no need to re-list models if the user already named one)"
                        )
                    else:
                        parts.append(
                            "model=configured — "
                            "user confirms defaults or says ready (e.g. 'these parameters are good', 'looks good', 'go ahead', 'yes') "
                            "→ call run_auto_labeling immediately — do NOT use send_reply to ask for confirmation first, "
                            "run_auto_labeling shows its own pre-flight summary and handles the confirmation step itself; "
                            "any message with a hyperparam name or value (e.g. 'epochs 20', 'set learning rate to 0.001', '5 epochs') "
                            "→ call set_auto_labeling_hyperparams immediately with ONLY the changed values — "
                            "do NOT re-call configure_auto_labeling for hyperparam-only requests; "
                            "to change the model → call configure_auto_labeling immediately "
                            "(no need to re-list models if the user already named one)"
                        )
                else:
                    parts.append("model=configured")
            if al.auto_labeling_complete:
                parts.append("auto_labeling=complete")
            if al.labels_imported:
                parts.append(
                    "workflow_complete — "
                    "if user names a specific workflow: call switch_workflow with that name; "
                    "if user does not name one: send_reply with the workflow list and ask which one. "
                    "Generic words like 'done', 'ok', 'thanks', 'exit' do NOT name a workflow — "
                    "respond with send_reply asking if they want to start another workflow or are finished."
                )
            if not al.phase and not al.auto_labeling_complete and (al.labeling_path or al.model_configured):
                backend_rule = (
                    "to change the backend → call set_labeling_backend directly; "
                )
                path_rule = (
                    "user wants to switch labeling approach / use auto generated instead / use manual instead → "
                    "call set_labeling_path('auto' or 'manual') — "
                    "dataset, backend, and classes are preserved, only path-specific state resets; "
                    "user explicitly wants to start completely over and discard everything → "
                    f"call switch_workflow(workflow_name='{state.workflow_name}') — "
                    "clears ALL state including dataset, returning to dataset selection; "
                )
                parts.append(
                    "RECONFIGURABLE (Zone A — before the workflow locks): "
                    "to change the dataset → call set_selected_dataset; "
                    "to change the model → call configure_auto_labeling "
                    "(call list_model_sources_and_models first if user doesn't know the model name); "
                    + backend_rule
                    + path_rule
                )
        cm = state.class_mapping
        if cm:
            if cm.model_configured and not cm.source_dataset_set:
                parts.append(
                    "class_mapping: model_configured | "
                    "NEXT STEP: get source dataset, call set_class_mapping_dataset_source"
                )
            elif cm.source_dataset_set and not cm.target_dataset_set:
                parts.append(
                    "class_mapping: source_set | "
                    "NEXT STEP: get target dataset, call set_class_mapping_dataset_target"
                )
            elif cm.target_dataset_set and not cm.candidate_labels_set:
                parts.append(
                    "class_mapping: target_set | "
                    "NEXT STEP: get class mapping from user, call set_class_mapping_candidate_labels"
                )
            elif cm.candidate_labels_set:
                parts.append(
                    "class_mapping: fully_configured | "
                    "call run_class_mapping when user confirms"
                )

        ad = state.anomaly_detection
        if ad:
            if ad.model_configured and not ad.data_source_set:
                parts.append(
                    "anomaly_detection: model_configured | "
                    "NEXT STEP: get location + rare_class, call set_anomaly_detection_data_source"
                )
            elif ad.data_source_set:
                parts.append(
                    "anomaly_detection: data_source_set | "
                    "hyperparams: call set_anomaly_detection_hyperparams if user adjusts; "
                    "when ready: call run_anomaly_detection"
                )

        es = state.embedding_selection
        if es:
            if es.model_configured:
                parts.append(
                    "embedding_selection: model_configured | "
                    "params: call set_embedding_selection_params if user adjusts; "
                    "when ready: call run_embedding_selection"
                )

        zs = state.auto_labeling_zero_shot
        if zs:
            if zs.models_configured and not zs.threshold_set:
                parts.append(
                    "zero_shot: models_configured | "
                    "NEXT STEP: get threshold, call set_auto_labeling_zero_shot_threshold"
                )
            elif zs.threshold_set and not zs.classes_set:
                parts.append(
                    "zero_shot: threshold_set | "
                    "NEXT STEP: get object classes, call set_auto_labeling_zero_shot_classes"
                )
            elif zs.classes_set:
                parts.append(
                    "zero_shot: fully_configured | "
                    "call run_zero_shot_auto_labeling when user confirms"
                )

        ens = state.ensemble_selection
        if ens:
            if ens.params_set and not ens.classes_set:
                parts.append(
                    "ensemble: params_set | "
                    "NEXT STEP: get positive classes, call set_ensemble_selection_classes"
                )
            elif ens.classes_set:
                parts.append(
                    "ensemble: fully_configured | "
                    "call run_ensemble_selection when user confirms"
                )

        parts.append(
            "ALWAYS AVAILABLE: user wants a completely different workflow → "
            "use send_reply to list the six workflows and ask which one — "
            "do NOT call switch_workflow until the user names a specific workflow."
        )
        return "SESSION_STATE: " + " | ".join(parts)
    except Exception:
        return ""


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """SSE endpoint. Events: status, log, progress, reply (terminal), error."""
    data    = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])

    MAX_HISTORY_TURNS = 8
    if len(history) > MAX_HISTORY_TURNS:
        history = history[-MAX_HISTORY_TURNS:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    try:
        _state = WorkflowState.load()
    except Exception:
        _state = None

    if _state:
        al = _state.auto_labeling
        logging.warning(
            f"[STREAM STATE] loaded: workflow={_state.workflow_name!r} "
            f"dataset={_state.dataset_name!r} confirmed={_state.dataset_confirmed} "
            f"backend={al.labeling_backend if al else ''!r} "
            f"path={al.labeling_path if al else ''!r} "
            f"phase={al.phase if al else ''!r}"
        )

    if _state and _state.workflow_just_reset:
        messages.append({"role": "system", "content": (
            "WORKFLOW_RESET: The previous workflow session has completely ended. "
            "All parameters (dataset, backend, model, classes) have been cleared. "
            "The conversation history above belongs to a DIFFERENT session — "
            "do NOT reuse any dataset name, backend, model, or configuration from it. "
            "The user must explicitly provide all values from scratch in this new session."
        )})
        _state.workflow_just_reset = False
        _state.save()

    state_hint = _build_state_hint(_state)
    if state_hint:
        messages.append({"role": "system", "content": state_hint})
        logging.warning(f"[STREAM HINT] {state_hint}")
    messages.append({"role": "user", "content": message})

    active_tools = filter_tools_for_state(tools, _state)
    logging.warning(
        f"[STREAM TOOLS] Active ({len(active_tools)}): "
        f"{[t['function']['name'] for t in active_tools]}"
    )

    event_queue: asyncio.Queue = asyncio.Queue()

    async def progress_cb(event_type: str, evt_data: dict) -> None:
        await event_queue.put((event_type, evt_data))

    async def run_pipeline() -> None:
        try:
            current_tools = active_tools
            pipeline = None
            MAX_AGENTIC_ITERATIONS = 5

            for iteration in range(MAX_AGENTIC_ITERATIONS):
                tool_choice = "required" if iteration == 0 else "auto"

                try:
                    assistant_message = await request.app.state.llm.chat(
                        messages, tools=current_tools, tool_choice=tool_choice
                    )
                except Exception as e:
                    err = str(e).lower()
                    msg = (
                        "The request timed out reaching the AI service. Please try again."
                        if "timeout" in err or "connecttimeout" in err
                        else "Something went wrong connecting to the AI service. Please try again."
                    )
                    await event_queue.put(("error", {"message": msg}))
                    return

                if not (hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls):
                    reply = assistant_message.content or ""
                    logging.warning(f"[STREAM DECISION] iter={iteration} → end_turn (no tool call)")
                    await event_queue.put(("reply", {"message": reply}))
                    return

                tool_calls = assistant_message.tool_calls
                logging.warning(
                    f"[STREAM DECISION] iter={iteration} tool_choice={tool_choice!r} → "
                    f"{[f'{c.function.name}({c.function.arguments[:60]})' for c in tool_calls]}"
                )

                if len(tool_calls) == 1 and tool_calls[0].function.name == "send_reply":
                    try:
                        args  = json.loads(tool_calls[0].function.arguments)
                        reply = _attach_source(args.get("message", ""), args.get("source"))
                    except Exception:
                        reply = "Something went wrong. Please try again."
                    await event_queue.put(("reply", {"message": reply}))
                    return

                messages.append({
                    "role":       "assistant",
                    "content":    assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id":       c.id,
                            "type":     "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                        }
                        for c in tool_calls
                    ],
                })

                if pipeline is None:
                    pipeline = ChatPipeline(mcp_transport=request.app.state.mcp_transport, llm=request.app.state.llm)

                tool_results, early_reply = await pipeline.run(
                    tool_calls, messages, progress_cb=progress_cb
                )

                if early_reply is not None:
                    await event_queue.put(("reply", {"message": early_reply}))
                    return

                if iteration > 0:
                    logging.warning(
                        f"[STREAM] Agentic loop: iteration {iteration + 1} — "
                        f"tools called: {[r['name'] for r in tool_results]}"
                    )

                current_tools = filter_tools_for_state(tools, pipeline.state)

                tools_called = [r["name"] for r in tool_results]
                if (
                    pipeline.state.workflow_name
                    and not pipeline.state.dataset_confirmed
                    and "list_datasets" not in tools_called
                ):
                    messages.append({
                        "role":    "system",
                        "content": (
                            "Reminder: the user has selected a workflow but has not yet "
                            "confirmed a dataset. Show the user the dataset list returned "
                            "by list_datasets above. Do NOT list datasets from memory."
                        ),
                    })

            logging.warning(f"[STREAM] Exceeded {MAX_AGENTIC_ITERATIONS} agentic iterations")
            await event_queue.put(("reply", {"message": "I wasn't able to complete this step. Please try again."}))

        except Exception as e:
            logging.warning(f"[STREAM] run_pipeline exception: {e}")
            await event_queue.put(("error", {"message": "An unexpected error occurred. Please try again."}))
        finally:
            await event_queue.put(None)

    async def generate():
        task = asyncio.create_task(run_pipeline())
        try:
            while True:
                item = await event_queue.get()
                if item is None:
                    break
                event_type, evt_data = item
                yield f"event: {event_type}\ndata: {json.dumps(evt_data)}\n\n"
        finally:
            await task

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)