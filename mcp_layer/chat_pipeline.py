# mcp_layer/chat_pipeline.py

import asyncio
import json
import logging
import re
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import SSETransport

from validate_workflow_state import (
    WorkflowState, AutoLabelingState, ClassMappingState,
    AnomalyDetectionState, EmbeddingSelectionState,
    ZeroShotAutoLabelingState, EnsembleSelectionState,
    WORKFLOW_DEPENDENCIES, validate_tool_input,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.py"
MAIN_PATH   = Path(__file__).resolve().parents[1] / "main.py"

# Human-readable status emitted to the UI before each tool call.
TOOL_STATUS_MESSAGES: dict[str, str] = {
    "select_workflow":                       "Setting up workflow...",
    "switch_workflow":                       "Switching workflow...",
    "set_selected_dataset":                  "Confirming dataset...",
    "list_datasets":                         "Fetching available datasets...",
    "list_model_sources_and_models":         "Fetching available models...",
    "configure_auto_labeling":               "Configuring model selection...",
    "set_auto_labeling_hyperparams":         "Updating hyperparameters...",
    "run_auto_labeling":                     "Starting auto-labeling — this may take several minutes...",
    "export_to_cvat":                        "Exporting dataset to CVAT...",
    "import_from_cvat":                      "Importing annotations from CVAT...",
    "export_to_label_studio":                "Exporting dataset to Label Studio...",
    "import_from_label_studio":              "Importing annotations from Label Studio...",
    "get_labeling_backend":                  "Detecting annotation backend...",
    "set_labeling_backend":                  "Configuring annotation backend...",
    "launch_voxel51_session":                "Launching Voxel51 visualization...",
    "list_class_mapping_models":             "Fetching class mapping models...",
    "configure_class_mapping_model":         "Configuring class mapping model...",
    "run_class_mapping":                     "Running class mapping...",
    "list_anomaly_detection_models":         "Fetching anomaly detection models...",
    "configure_anomaly_detection_model":     "Configuring anomaly detection model...",
    "run_anomaly_detection":                 "Running anomaly detection...",
    "list_embedding_selection_models":       "Fetching embedding selection models...",
    "configure_embedding_selection_model":   "Configuring embedding model...",
    "run_embedding_selection":               "Running embedding selection...",
    "list_zsal":                             "Fetching zero-shot models...",
    "configure_auto_labeling_zero_shot_models": "Configuring zero-shot models...",
    "run_zero_shot_auto_labeling":           "Running zero-shot auto-labeling...",
    "set_ensemble_selection_parameters":     "Setting ensemble parameters...",
    "set_ensemble_selection_classes":        "Setting ensemble classes...",
    "run_ensemble_selection":                "Running ensemble selection...",
}

# Strip ANSI escape codes and bare CR from subprocess output.
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')


def unwrap_tool_output(raw) -> str:
    """Normalize any LLM/MCP output type to a plain string."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if hasattr(raw, "text"):
        return (raw.text or "").replace("\\n", "\n").strip()
    if isinstance(raw, list):
        parts = [unwrap_tool_output(x) for x in raw]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(raw, dict):
        if "text" in raw and isinstance(raw["text"], str):
            return raw["text"].replace("\\n", "\n").strip()
        if "content" in raw and isinstance(raw["content"], list):
            return unwrap_tool_output(raw["content"])
        if "data" in raw and isinstance(raw["data"], dict) and "msg" in raw["data"]:
            return str(raw["data"]["msg"]).replace("\\n", "\n").strip()
        for key in ("message", "detail"):
            if key in raw and isinstance(raw[key], str):
                return raw[key].replace("\\n", "\n").strip()
    return str(raw).strip()


class ChatPipeline:
    """
    Owns all processing between the HTTP endpoint and the MCP tools.

    Invariant: every tool result is appended to `messages` immediately after
    execution in `_dispatch`, so the OpenAI message history never has an
    assistant tool_call_id without a matching tool result.
    """

    def __init__(self, mcp_transport: SSETransport, llm):
        self.transport = mcp_transport
        self.llm = llm

        self.state: WorkflowState = WorkflowState()

        # Per-request hyperparam caches. Not persisted to WorkflowState;
        # the MCP tools write them directly to the WORKFLOWS section of config.py.
        self.hyperparam_cache = {
            "mode": ["train", "inference"],
            "epochs": 10,
            "early_stop_patience": 5,
            "early_stop_threshold": 0,
            "learning_rate": 5e-5,
            "weight_decay": 0.0001,
            "max_grad_norm": 0.01,
        }
        self.hyperparam_cache_anomaly = {
            "mode": ["train", "inference"],
            "epochs": 12,
            "early_stop_patience": 5,
        }
        self.embedding_selection_cache = {
            "compute_representativeness": 0.99,
            "compute_unique_images_greedy": 0.01,
            "compute_unique_images_deterministic": 0.99,
            "compute_similar_images": 0.03,
            "neighbour_count": 3,
        }
        self.ensemble_selection_cache = {
            "iou_threshold": 0.5,
            "max_bbox_size": 0.1,
        }

        # Deferred system message injections — consumed in _build_reply after all
        # tool results are appended, which is required by OpenAI message ordering.
        self._confirmed_dataset: str | None = None
        self._detected_backend: dict | None = None

        # Set by run() when the caller wants streaming progress events.
        # Signature: async (event_type: str, data: dict) -> None
        self._progress_cb = None

    async def run(self, tool_calls: list, messages: list, progress_cb=None) -> tuple[list, str | None]:
        """
        Process all tool calls for one request. Loads WorkflowState fresh from
        config.py so state changes from previous requests are always visible.

        progress_cb — optional async callable(event_type: str, data: dict) used
        by the /chat/stream endpoint to push status/log/progress events to the UI.

        Returns (tool_results, early_reply).
        """
        self.state = WorkflowState.load()
        self._progress_cb = progress_cb

        tool_results = []
        logging.warning(
            f"[PIPELINE] Processing {len(tool_calls)} tool call(s): "
            f"{[c.function.name for c in tool_calls]}"
        )

        async with Client(self.transport) as mcp_client:
            for call in tool_calls:
                fn_name = call.function.name

                try:
                    fn_args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                ok, err, fn_args = validate_tool_input(fn_name, fn_args)
                if not ok:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "content": err,
                    })
                    tool_results.append({
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "fn_args": fn_args,
                        "result": err,
                    })
                    logging.warning(f"[PIPELINE] Tool input validation failed for {fn_name}: {err}")
                    continue

                result = await self._dispatch(fn_name, fn_args, call, mcp_client, messages, progress_cb)
                tool_results.append({
                    "tool_call_id": call.id,
                    "name": fn_name,
                    "fn_args": fn_args,
                    "result": result,
                })

        # If set_selected_dataset and select/switch_workflow fired in the same batch,
        # the workflow reset wipes dataset_confirmed. Re-apply it if both succeeded.
        workflow_reset_this_batch = any(
            r["name"] in ("select_workflow", "switch_workflow")
            for r in tool_results
        )
        dataset_set_this_batch = next(
            (r for r in tool_results if r["name"] == "set_selected_dataset"), None
        )
        if (
            workflow_reset_this_batch
            and dataset_set_this_batch
            and "DATASET_NOT_FOUND" not in unwrap_tool_output(
                dataset_set_this_batch.get("result", "")
            )
        ):
            dataset_name = dataset_set_this_batch.get("fn_args", {}).get("dataset_name", "")
            if dataset_name and not self.state.dataset_confirmed:
                logging.warning(
                    f"[PIPELINE] Re-applying dataset confirmation for '{dataset_name}' "
                    f"after workflow reset in same batch"
                )
                self.state.dataset_name = dataset_name
                self.state.dataset_confirmed = True
                self._confirmed_dataset = dataset_name
                self.state.save()

        early_reply = await self._build_reply(tool_results, messages)
        return tool_results, early_reply

    async def _dispatch(self, fn_name, fn_args, call, mcp_client, messages, progress_cb=None) -> str:
        """
        Execute a single tool call with precondition checks and state updates.
        Always appends the result to messages before returning.
        """
        if progress_cb and fn_name != "send_reply":
            status = TOOL_STATUS_MESSAGES.get(fn_name, f"Running {fn_name.replace('_', ' ')}...")
            await progress_cb("status", {"message": status})

        try:
            if fn_name == "send_reply":
                msg = fn_args.get("message", "")
                src = fn_args.get("source", "")
                content = f"{msg.strip()}\n[source: {src.strip()}]" if src and src.strip() else msg
                result = content

            elif fn_name in ("select_workflow", "switch_workflow"):
                result = await self._handle_select_or_switch_workflow(fn_name, fn_args, mcp_client)
                content = result

            elif fn_name == "set_selected_dataset":
                result = await self._handle_set_selected_dataset(fn_args, mcp_client)
                content = result

            elif fn_name == "list_model_sources_and_models":
                if not self.state.dataset_confirmed or not self.state.dataset_name:
                    result = (
                        "DATASET_NOT_CONFIRMED: A dataset must be confirmed before "
                        "selecting a model. Please call set_selected_dataset first."
                    )
                    content = result
                else:
                    if self.state.auto_labeling is None:
                        self.state.auto_labeling = AutoLabelingState()
                    if not self.state.auto_labeling.labeling_path:
                        self.state.auto_labeling.labeling_path = "auto"
                        self.state.save()
                    result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
                    # Mark models as listed so configure_auto_labeling can verify
                    # the user selected from real options, not a hallucinated name.
                    self.state.auto_labeling.models_listed = True
                    self.state.save()
                    content = result

            elif fn_name == "configure_auto_labeling":
                result = await self._handle_configure_auto_labeling(fn_args, mcp_client)
                content = result

            elif fn_name == "set_auto_labeling_hyperparams":
                result = await self._handle_set_auto_labeling_hyperparams(fn_args, mcp_client)
                content = result

            elif fn_name == "configure_class_mapping_model":
                result = await self._handle_configure_class_mapping_model(fn_args, mcp_client)
                content = result

            elif fn_name == "set_class_mapping_dataset_source":
                result = await self._handle_set_class_mapping_dataset_source(fn_args, mcp_client)
                content = result

            elif fn_name == "set_class_mapping_dataset_target":
                result = await self._handle_set_class_mapping_dataset_target(fn_args, mcp_client)
                content = result

            elif fn_name == "set_class_mapping_candidate_labels":
                result = await self._handle_set_class_mapping_candidate_labels(fn_args, mcp_client)
                content = result

            elif fn_name == "run_class_mapping":
                result = await self._handle_run_class_mapping(mcp_client)
                content = result

            elif fn_name == "configure_anomaly_detection_model":
                result = await self._handle_configure_anomaly_detection_model(fn_args, mcp_client)
                content = result

            elif fn_name == "set_anomaly_detection_data_source":
                result = await self._handle_set_anomaly_detection_data_source(fn_args, mcp_client)
                content = result

            elif fn_name == "set_anomaly_detection_hyperparams":
                result = await self._handle_set_anomaly_detection_hyperparams(fn_args, mcp_client)
                content = result

            elif fn_name == "run_anomaly_detection":
                result = await self._handle_run_anomaly_detection(mcp_client)
                content = result

            elif fn_name == "configure_embedding_selection_model":
                result = await self._handle_configure_embedding_selection_model(fn_args, mcp_client)
                content = result

            elif fn_name == "set_embedding_selection_params":
                result = await self._handle_set_embedding_selection_params(fn_args, mcp_client)
                content = result

            elif fn_name == "run_embedding_selection":
                result = await self._handle_run_embedding_selection(mcp_client)
                content = result

            elif fn_name == "configure_auto_labeling_zero_shot_models":
                result = await self._handle_configure_zero_shot_models(fn_args, mcp_client)
                content = result

            elif fn_name == "set_auto_labeling_zero_shot_threshold":
                result = await self._handle_set_zero_shot_threshold(fn_args, mcp_client)
                content = result

            elif fn_name == "set_auto_labeling_zero_shot_classes":
                result = await self._handle_set_zero_shot_classes(fn_args, mcp_client)
                content = result

            elif fn_name == "run_zero_shot_auto_labeling":
                result = await self._handle_run_zero_shot(mcp_client)
                content = result

            elif fn_name == "set_ensemble_selection_parameters":
                result = await self._handle_set_ensemble_selection_parameters(fn_args, mcp_client)
                content = result

            elif fn_name == "set_ensemble_selection_classes":
                result = await self._handle_set_ensemble_classes(fn_args, mcp_client)
                content = result

            elif fn_name == "run_ensemble_selection":
                result = await self._handle_run_ensemble_selection(mcp_client)
                content = result

            elif fn_name == "export_to_cvat":
                result = await self._handle_export_to_cvat(fn_args, mcp_client)
                content = result

            elif fn_name == "run_auto_labeling":
                if progress_cb:
                    result = await self._handle_run_auto_labeling_streaming(fn_args, progress_cb)
                else:
                    result = await self._handle_run_auto_labeling(fn_args, mcp_client)
                content = result

            elif fn_name == "import_from_cvat":
                result = await self._handle_import_from_cvat(fn_args, mcp_client)
                content = result

            elif fn_name == "get_labeling_backend":
                result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
                # unwrap_tool_output may produce a Python repr (single-quoted dict)
                # rather than valid JSON, so we read env vars directly instead.
                try:
                    import ast as _ast, os as _os
                    cvat_ok = bool(_os.getenv("CVAT_ACCESS_TOKEN", "").strip())
                    ls_ok   = bool(_os.getenv("LS_TOKEN", "").strip())
                    if cvat_ok and ls_ok:
                        active = "cvat"
                    elif ls_ok:
                        active = "label_studio"
                    elif cvat_ok:
                        active = "cvat"
                    else:
                        active = "none"
                    if active in ("cvat", "label_studio"):
                        if self.state.auto_labeling is None:
                            self.state.auto_labeling = AutoLabelingState()
                        self.state.auto_labeling.labeling_backend = active
                        self.state.save()
                except Exception:
                    pass
                content = result

            elif fn_name == "set_labeling_backend":
                result = await self._handle_set_labeling_backend(fn_args, mcp_client)
                content = result

            elif fn_name == "export_to_label_studio":
                result = await self._handle_export_to_label_studio(fn_args, mcp_client)
                content = result

            elif fn_name == "import_from_label_studio":
                result = await self._handle_import_from_label_studio(fn_args, mcp_client)
                content = result

            elif fn_name == "launch_voxel51_session":
                result = await self._handle_launch_voxel51(fn_args, mcp_client)
                content = result

            else:
                logging.warning(f"[PIPELINE] Calling MCP tool: {fn_name} with args: {fn_args}")
                result = unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))
                logging.warning(f"[PIPELINE] MCP tool {fn_name} returned")
                content = result

        except Exception as e:
            logging.warning(f"[PIPELINE] Exception in {fn_name}: {e}")
            result = str(e)
            content = result

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": fn_name,
            "content": content,
        })
        logging.warning(f"[PIPELINE] Appended tool result: name={fn_name} tool_call_id={call.id}")

        return result

    # Per-tool handlers

    async def _handle_select_or_switch_workflow(
        self, fn_name: str, fn_args: dict, mcp_client
    ) -> str:
        """Reset state for the new workflow, check dependencies, then call the MCP tool."""
        workflow_name = fn_args.get("workflow_name", "")

        deps = WORKFLOW_DEPENDENCIES.get(workflow_name, [])
        if deps:
            completed = [
                wf for wf in [
                    "auto_labeling", "class_mapping", "anomaly_detection",
                    "embedding_selection", "auto_labeling_zero_shot", "ensemble_selection"
                ]
                if getattr(self.state, wf, None) is not None
            ]
            ok, msg = self.state.check_workflow_dependencies(workflow_name, completed)
            if not ok:
                return msg

        self.state = self.state.reset_for_workflow(workflow_name)
        return unwrap_tool_output(await mcp_client.call_tool(fn_name, fn_args))

    async def _handle_set_selected_dataset(self, fn_args: dict, mcp_client) -> str:
        ok, msg = self.state.can_confirm_dataset()
        if not ok:
            return msg

        dataset_name = fn_args["dataset_name"]
        raw = await mcp_client.call_tool("set_selected_dataset", {"dataset_name": dataset_name})
        tool_output = unwrap_tool_output(raw)

        if "DATASET_NOT_FOUND" in tool_output:
            # Dataset may not yet be visible in the FiftyOne registry cache after
            # recent ingestion. Retry up to 3 times with increasing backoff.
            logging.warning(f"[PIPELINE] Dataset '{dataset_name}' not found — retrying up to 3 times")
            for attempt, wait in enumerate([2, 4, 6], start=1):
                await asyncio.sleep(wait)
                raw = await mcp_client.call_tool("set_selected_dataset", {"dataset_name": dataset_name})
                tool_output = unwrap_tool_output(raw)
                if "DATASET_NOT_FOUND" not in tool_output:
                    logging.warning(f"[PIPELINE] Dataset '{dataset_name}' found on retry {attempt}")
                    break
                logging.warning(f"[PIPELINE] Dataset '{dataset_name}' still not found after retry {attempt}")

        if "DATASET_NOT_FOUND" not in tool_output:
            self.state.dataset_name = dataset_name
            self.state.dataset_confirmed = True
            self.state.save()
            # Defer CURRENT_DATASET injection to _build_reply (OpenAI ordering rules).
            self._confirmed_dataset = dataset_name
            if self.state.workflow_name == "auto_labeling":
                await self._auto_detect_backend(mcp_client)
        else:
            self.state.dataset_confirmed = False

        return tool_output

    async def _auto_detect_backend(self, mcp_client) -> None:
        """
        Detect the annotation backend from env vars directly (no MCP round-trip).

        Avoids MCP because get_labeling_backend() returns a Python dict which
        unwrap_tool_output renders as a single-quoted repr string that json.loads
        cannot parse, causing silent failure and no backend being written to state.
        """
        try:
            import os as _os
            cvat_ok = bool(_os.getenv("CVAT_ACCESS_TOKEN", "").strip())
            ls_ok   = bool(_os.getenv("LS_TOKEN", "").strip())

            if cvat_ok and ls_ok:
                # Both available — user must choose. Do NOT pre-set a backend in state;
                # writing "cvat" here would make it look like the user already confirmed CVAT,
                # causing the LLM to skip the backend-choice question.
                active  = "both"
                message = (
                    "Both CVAT and Label Studio credentials are configured. "
                    "Which would you prefer to use?"
                )
            elif ls_ok:
                active  = "label_studio"
                message = "Label Studio credentials found. Using Label Studio for annotation."
            elif cvat_ok:
                active  = "cvat"
                message = "CVAT credentials found. Using CVAT for annotation."
            else:
                active  = "none"
                message = (
                    "No annotation backend credentials found in .env. "
                    "Please add CVAT_ACCESS_TOKEN or LS_TOKEN before continuing."
                )

            backend_info = {
                "cvat_available": cvat_ok,
                "ls_available":   ls_ok,
                "active_backend": active,
                "message":        message,
            }

            # Write to state, overriding the default "cvat" set by AutoLabelingState.
            if active in ("cvat", "label_studio"):
                if self.state.auto_labeling is None:
                    self.state.auto_labeling = AutoLabelingState()
                self.state.auto_labeling.labeling_backend = active
                self.state.save()
                logging.warning(f"[PIPELINE] Backend auto-detected and written to state: {active}")
            else:
                logging.warning(f"[PIPELINE] Backend detection: {active} — not writing to state until user confirms")

            self._detected_backend = backend_info
        except Exception as e:
            logging.warning(f"[PIPELINE] _auto_detect_backend failed: {e}")
            self._detected_backend = None

    async def _handle_configure_auto_labeling(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        ok, msg = self.state.auto_labeling.can_configure_auto_labeling()
        if not ok:
            return msg

        if not self.state.auto_labeling.labeling_path:
            self.state.auto_labeling.labeling_path = "auto"
            self.state.save()

        result = unwrap_tool_output(await mcp_client.call_tool("configure_auto_labeling", fn_args))
        self.state.auto_labeling.model_configured = True
        # Hyperparams default to confirmed; set_auto_labeling_hyperparams keeps this True if called.
        self.state.auto_labeling.hyperparams_confirmed = True
        self.state.save()
        return result

    async def _handle_set_auto_labeling_hyperparams(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        for k, v in fn_args.items():
            if v is not None:
                self.hyperparam_cache[k] = v

        result = unwrap_tool_output(
            await mcp_client.call_tool("set_auto_labeling_hyperparams", self.hyperparam_cache.copy())
        )
        self.state.auto_labeling.hyperparams_confirmed = True
        self.state.save()
        return result

    # Class mapping

    async def _handle_configure_class_mapping_model(self, fn_args: dict, mcp_client) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        result = unwrap_tool_output(await mcp_client.call_tool("configure_class_mapping_model", fn_args))
        self.state.class_mapping.model_configured = True
        self.state.save()
        return result

    async def _handle_set_class_mapping_dataset_source(self, fn_args: dict, mcp_client) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        result = unwrap_tool_output(await mcp_client.call_tool("set_class_mapping_dataset_source", fn_args))
        self.state.class_mapping.source_dataset_set = True
        self.state.save()
        return result

    async def _handle_set_class_mapping_dataset_target(self, fn_args: dict, mcp_client) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        result = unwrap_tool_output(await mcp_client.call_tool("set_class_mapping_dataset_target", fn_args))
        self.state.class_mapping.target_dataset_set = True
        self.state.save()
        return result

    async def _handle_set_class_mapping_candidate_labels(self, fn_args: dict, mcp_client) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        result = unwrap_tool_output(await mcp_client.call_tool("set_class_mapping_candidate_labels", fn_args))
        self.state.class_mapping.candidate_labels_set = True
        self.state.save()
        return result

    async def _handle_run_class_mapping(self, mcp_client) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        ok, msg = self.state.class_mapping.can_run_class_mapping(self.state.dataset_confirmed)
        if not ok:
            return msg
        return unwrap_tool_output(await mcp_client.call_tool("run_class_mapping", {}))

    # Anomaly detection

    async def _handle_configure_anomaly_detection_model(self, fn_args: dict, mcp_client) -> str:
        if self.state.anomaly_detection is None:
            self.state.anomaly_detection = AnomalyDetectionState()
        result = unwrap_tool_output(await mcp_client.call_tool("configure_anomaly_detection_model", fn_args))
        self.state.anomaly_detection.model_configured = True
        self.state.save()
        return result

    async def _handle_set_anomaly_detection_data_source(self, fn_args: dict, mcp_client) -> str:
        if self.state.anomaly_detection is None:
            self.state.anomaly_detection = AnomalyDetectionState()
        result = unwrap_tool_output(await mcp_client.call_tool("set_anomaly_detection_data_source", fn_args))
        self.state.anomaly_detection.data_source_set = True
        self.state.save()
        return result

    async def _handle_set_anomaly_detection_hyperparams(self, fn_args: dict, mcp_client) -> str:
        if self.state.anomaly_detection is None:
            self.state.anomaly_detection = AnomalyDetectionState()
        for k, v in fn_args.items():
            if v is not None:
                self.hyperparam_cache_anomaly[k] = v
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_anomaly_detection_hyperparams", self.hyperparam_cache_anomaly.copy())
        )
        self.state.anomaly_detection.hyperparams_confirmed = True
        self.state.save()
        return result

    async def _handle_run_anomaly_detection(self, mcp_client) -> str:
        if self.state.anomaly_detection is None:
            self.state.anomaly_detection = AnomalyDetectionState()
        ok, msg = self.state.anomaly_detection.can_run_anomaly_detection(self.state.dataset_confirmed)
        if not ok:
            return msg
        return unwrap_tool_output(await mcp_client.call_tool("run_anomaly_detection", {}))

    # Embedding selection

    async def _handle_configure_embedding_selection_model(self, fn_args: dict, mcp_client) -> str:
        if self.state.embedding_selection is None:
            self.state.embedding_selection = EmbeddingSelectionState()
        result = unwrap_tool_output(await mcp_client.call_tool("configure_embedding_selection_model", fn_args))
        self.state.embedding_selection.model_configured = True
        self.state.save()
        return result

    async def _handle_set_embedding_selection_params(self, fn_args: dict, mcp_client) -> str:
        if self.state.embedding_selection is None:
            self.state.embedding_selection = EmbeddingSelectionState()
        for k, v in fn_args.items():
            if v is not None:
                self.embedding_selection_cache[k] = v
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_embedding_selection_params", self.embedding_selection_cache.copy())
        )
        self.state.embedding_selection.params_set = True
        self.state.save()
        return result

    async def _handle_run_embedding_selection(self, mcp_client) -> str:
        if self.state.embedding_selection is None:
            self.state.embedding_selection = EmbeddingSelectionState()
        ok, msg = self.state.embedding_selection.can_run_embedding_selection(self.state.dataset_confirmed)
        if not ok:
            return msg
        return unwrap_tool_output(await mcp_client.call_tool("run_embedding_selection", {}))

    # Zero-shot auto-labeling

    async def _handle_configure_zero_shot_models(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling_zero_shot is None:
            self.state.auto_labeling_zero_shot = ZeroShotAutoLabelingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("configure_auto_labeling_zero_shot_models", fn_args)
        )
        self.state.auto_labeling_zero_shot.models_configured = True
        self.state.save()
        return result

    async def _handle_set_zero_shot_threshold(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling_zero_shot is None:
            self.state.auto_labeling_zero_shot = ZeroShotAutoLabelingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_auto_labeling_zero_shot_threshold", fn_args)
        )
        self.state.auto_labeling_zero_shot.threshold_set = True
        self.state.save()
        return result

    async def _handle_set_zero_shot_classes(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling_zero_shot is None:
            self.state.auto_labeling_zero_shot = ZeroShotAutoLabelingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_auto_labeling_zero_shot_classes", fn_args)
        )
        self.state.auto_labeling_zero_shot.classes_set = True
        self.state.save()
        return result

    async def _handle_run_zero_shot(self, mcp_client) -> str:
        if self.state.auto_labeling_zero_shot is None:
            self.state.auto_labeling_zero_shot = ZeroShotAutoLabelingState()
        ok, msg = self.state.auto_labeling_zero_shot.can_run_zero_shot(self.state.dataset_confirmed)
        if not ok:
            return msg
        return unwrap_tool_output(await mcp_client.call_tool("run_zero_shot_auto_labeling", {}))

    # Ensemble selection

    async def _handle_set_ensemble_selection_parameters(self, fn_args: dict, mcp_client) -> str:
        if self.state.ensemble_selection is None:
            self.state.ensemble_selection = EnsembleSelectionState()
        for k, v in fn_args.items():
            if v is not None:
                self.ensemble_selection_cache[k] = v
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_ensemble_selection_parameters", self.ensemble_selection_cache.copy())
        )
        self.state.ensemble_selection.params_set = True
        self.state.save()
        return result

    async def _handle_set_ensemble_classes(self, fn_args: dict, mcp_client) -> str:
        if self.state.ensemble_selection is None:
            self.state.ensemble_selection = EnsembleSelectionState()
        result = unwrap_tool_output(await mcp_client.call_tool("set_ensemble_selection_classes", fn_args))
        self.state.ensemble_selection.classes_set = True
        self.state.save()
        return result

    async def _handle_run_ensemble_selection(self, mcp_client) -> str:
        if self.state.ensemble_selection is None:
            self.state.ensemble_selection = EnsembleSelectionState()
        ok, msg = self.state.ensemble_selection.can_run_ensemble_selection(self.state.dataset_confirmed)
        if not ok:
            return msg
        return unwrap_tool_output(await mcp_client.call_tool("run_ensemble_selection", {}))

    async def _handle_export_to_cvat(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling and not self.state.auto_labeling.labeling_backend:
            return (
                "BACKEND_NOT_SET: The annotation backend has not been determined yet. "
                "Please wait — the system will detect your configured backend first."
            )
        with_predictions = fn_args.get("with_predictions", False)

        if self.state.workflow_name == "auto_labeling":
            if self.state.auto_labeling is None:
                self.state.auto_labeling = AutoLabelingState()

            # On the auto path, export is handled automatically after labeling completes.
            if self.state.auto_labeling.labeling_path == "auto" and not with_predictions:
                return (
                    "Export is handled automatically after auto-labeling completes. "
                    "Please run the auto-labeling workflow first."
                )

            if not self.state.auto_labeling.labeling_path:
                self.state.auto_labeling.labeling_path = "manual" if not with_predictions else "auto"
                self.state.save()

            ok, msg = self.state.auto_labeling.can_export_to_cvat(with_predictions)
            if not ok:
                return msg

        import os as _os
        if not _os.getenv("CVAT_ACCESS_TOKEN", "").strip():
            return (
                "No CVAT credentials found. "
                "Please add CVAT_ACCESS_TOKEN to your .env file and restart the server."
            )

        if not with_predictions and fn_args.get("classes"):
            if self.state.auto_labeling is None:
                self.state.auto_labeling = AutoLabelingState()
            self.state.auto_labeling.manual_classes = fn_args["classes"]
            self.state.save()

        result = unwrap_tool_output(await mcp_client.call_tool("export_to_cvat", fn_args))

        if self.state.auto_labeling and "Task ID:" in result:
            try:
                task_id = int(result.split("Task ID:")[1].split()[0].strip())
                self.state.auto_labeling.cvat_task_id = task_id
                self.state.save()
            except Exception:
                pass

        return result

    async def _handle_run_auto_labeling_streaming(self, fn_args: dict, progress_cb) -> str:
        """
        Streaming variant of run_auto_labeling: runs main.py directly (bypassing MCP)
        and feeds stdout/stderr line-by-line to progress_cb as log + progress events.
        Falls back to the same precondition guard as the non-streaming path.
        """
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        ok, msg = self.state.auto_labeling.can_run_auto_labeling(
            self.state.dataset_confirmed, self.state.dataset_name,
        )
        if not ok:
            return msg

        process = await asyncio.create_subprocess_exec(
            "python", "-u", str(MAIN_PATH),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(MAIN_PATH.parent),
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def _read(stream, buf: list[str]) -> None:
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                text = _ANSI_RE.sub("", raw.decode("utf-8", errors="ignore")).rstrip()
                if not text.strip():
                    continue
                buf.append(text)
                await progress_cb("log", {"line": text})


        await asyncio.gather(_read(process.stdout, stdout_lines), _read(process.stderr, stderr_lines))
        await process.wait()

        output       = "\n".join(stdout_lines)
        error_output = "\n".join(stderr_lines)
        combined     = output + "\n" + error_output

        if "Evaluating detections..." in combined:
            res_lines, capture = [], False
            for line in stdout_lines:
                if "              precision    recall  f1-score   support" in line:
                    capture = True
                    res_lines.append(line)
                elif capture and line.startswith("You have launched a remote App on port 5151"):
                    break
                elif capture:
                    res_lines.append(line)
            report = "\n".join(res_lines).strip() or "No inference results found."
        else:
            report = "Training completed successfully.\nThe model is ready to be tested using inference on the validation set."

        log_path = "output/logs/last_auto_labeling_log.txt"
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(
            f"=== STDOUT ===\n{output}\n\n=== STDERR ===\n{error_output}\n\n=== EXIT CODE ===\n{process.returncode}",
            encoding="utf-8",
        )

        if process.returncode == 0:
            return (
                f"Auto-labeling workflow completed.\n\n"
                f"**Result Summary:**\n```\n{report}\n```\n"
                f"Full logs saved to `{log_path}`"
            )
        return (
            f"Auto-labeling failed with exit code {process.returncode}.\n"
            f"Error details:\n```\n{error_output[-3000:]}\n```\n"
            f"Full logs saved to `{log_path}`"
        )

    async def _handle_run_auto_labeling(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        ok, msg = self.state.auto_labeling.can_run_auto_labeling(
            self.state.dataset_confirmed, self.state.dataset_name,
        )
        if not ok:
            return msg

        return unwrap_tool_output(await mcp_client.call_tool("run_auto_labeling", fn_args))

    async def _handle_import_from_cvat(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling:
            ok, msg = self.state.auto_labeling.can_import_from_cvat()
            if not ok:
                return msg
        return unwrap_tool_output(await mcp_client.call_tool("import_from_cvat", fn_args))

    async def _handle_set_labeling_backend(self, fn_args: dict, mcp_client) -> str:
        """Validate credentials for the chosen backend, update state, and persist."""
        result = unwrap_tool_output(await mcp_client.call_tool("set_labeling_backend", fn_args))
        if "LS_BACKEND_ERROR" not in result:
            backend = fn_args.get("backend", "cvat")
            if self.state.auto_labeling is None:
                self.state.auto_labeling = AutoLabelingState()
            self.state.auto_labeling.labeling_backend = backend
            self.state.save()
        return result

    async def _handle_export_to_label_studio(self, fn_args: dict, mcp_client) -> str:
        """Mirror of _handle_export_to_cvat for Label Studio."""
        if self.state.auto_labeling and not self.state.auto_labeling.labeling_backend:
            return (
                "BACKEND_NOT_SET: The annotation backend has not been determined yet. "
                "Please wait — the system will detect your configured backend first."
            )
        with_predictions = fn_args.get("with_predictions", False)

        if self.state.workflow_name == "auto_labeling":
            if self.state.auto_labeling is None:
                self.state.auto_labeling = AutoLabelingState()

            if self.state.auto_labeling.labeling_path == "auto" and not with_predictions:
                return (
                    "Export is handled automatically after auto-labeling completes. "
                    "Please run the auto-labeling workflow first."
                )

            if not self.state.auto_labeling.labeling_path:
                self.state.auto_labeling.labeling_path = "manual" if not with_predictions else "auto"
                self.state.save()

            ok, msg = self.state.auto_labeling.can_export_to_label_studio(with_predictions)
            if not ok:
                return msg

        import os as _os
        if not _os.getenv("LS_TOKEN", "").strip():
            return (
                "No Label Studio credentials found. "
                "Please add LS_TOKEN to your .env file and restart the server."
            )

        if not with_predictions and fn_args.get("classes"):
            if self.state.auto_labeling is None:
                self.state.auto_labeling = AutoLabelingState()
            self.state.auto_labeling.manual_classes = fn_args["classes"]
            self.state.save()

        result = unwrap_tool_output(await mcp_client.call_tool("export_to_label_studio", fn_args))

        # Read ls_task_ids back from registry and persist to state.
        if self.state.auto_labeling and "Project ID" in result:
            try:
                from pathlib import Path as _Path
                import json as _json
                tasks_file = _Path(__file__).resolve().parents[1] / "output" / "ls_tasks.json"
                if tasks_file.exists():
                    registry = _json.loads(tasks_file.read_text())
                    dataset_name = fn_args.get("dataset_name", "")
                    if dataset_name in registry:
                        self.state.auto_labeling.ls_task_ids = registry[dataset_name].get("task_ids", [])
                        self.state.save()
            except Exception:
                pass

        return result

    async def _handle_import_from_label_studio(self, fn_args: dict, mcp_client) -> str:
        """
        Mirror of _handle_import_from_cvat for Label Studio.
        The finalizer is applied in _build_reply, not here, to avoid double-wrapping.
        """
        if self.state.auto_labeling:
            ok, msg = self.state.auto_labeling.can_import_from_label_studio()
            if not ok:
                return msg
        return unwrap_tool_output(await mcp_client.call_tool("import_from_label_studio", fn_args))

    async def _handle_launch_voxel51(self, fn_args: dict, mcp_client) -> str:
        dataset_name = fn_args.get("dataset_name", "").strip()
        if not dataset_name:
            dataset_name = self.state.labeled_dataset_name.strip() or self.state.dataset_name.strip()
        if not dataset_name:
            return (
                "Cannot launch Voxel51: no dataset name was provided or "
                "found in session state. Please specify the dataset name."
            )
        fn_args["dataset_name"] = dataset_name
        logging.warning(f"[PIPELINE] launch_voxel51_session -> dataset='{dataset_name}'")
        return unwrap_tool_output(await mcp_client.call_tool("launch_voxel51_session", fn_args))

    async def _build_reply(self, tool_results: list, messages: list) -> str | None:
        """
        Inject deferred system messages, then scan tool results for early-return
        tools. Returns a reply string, or None to fall through to the final LLM pass.
        """
        # Inject CURRENT_DATASET after all tool results are in messages.
        # Re-inject on every request where the dataset is confirmed so the
        # model doesn't lose track across a long message history.
        confirmed_name = self._confirmed_dataset or (
            self.state.dataset_name if self.state.dataset_confirmed else None
        )
        if confirmed_name:
            messages.append({"role": "system", "content": f"CURRENT_DATASET: {confirmed_name}"})
            self._confirmed_dataset = None

        # Inject LABELING_BACKEND after CURRENT_DATASET so the model knows which
        # backend to use without needing to call get_labeling_backend.
        if self._detected_backend:
            active   = self._detected_backend.get("active_backend", "")
            cvat_ok  = self._detected_backend.get("cvat_available", False)
            ls_ok    = self._detected_backend.get("ls_available", False)
            msg_text = self._detected_backend.get("message", "")
            both     = cvat_ok and ls_ok

            if both:
                # Both backends available — no backend is confirmed yet.
                # The user MUST choose before any labeling path is presented.
                instruction = (
                    "Backend detection is complete. Both backends are available but NEITHER is confirmed. "
                    "You MUST ask the user which backend they prefer (CVAT or Label Studio) "
                    "and call set_labeling_backend(backend=<choice>) before proceeding. "
                    "Do NOT present labeling paths yet. Do NOT assume CVAT is chosen."
                )
            else:
                # Exactly one backend configured — confirmed, skip straight to Step 3b.
                instruction = (
                    f"Backend detection is complete — active backend is {active}. "
                    f"Skip Step 3a and proceed directly to Step 3b "
                    f"(ask the user Manual vs Auto Labeling)."
                )

            messages.append({
                "role": "system",
                "content": f"LABELING_BACKEND: {active}. {msg_text} {instruction}",
            })
            self._detected_backend = None

        for result in tool_results:
            fn_name = result["name"]
            fn_args = result.get("fn_args", {})
            tool_output = unwrap_tool_output(result.get("result", ""))

            if fn_name == "export_to_cvat" and any(
                s in tool_output for s in [
                    "CVAT_TASK_LIMIT_REACHED", "CVAT_FORBIDDEN",
                    "CVAT_AUTH_ERROR", "CVAT_NOT_FOUND", "CVAT_CONNECTION_ERROR"
                ]
            ):
                return tool_output.split(":", 1)[1].strip() if ":" in tool_output else tool_output

            if fn_name in (
                "export_to_label_studio", "export_to_cvat",
                "set_labeling_backend", "get_labeling_backend"
            ) and any(
                s in tool_output for s in [
                    "LS_AUTH_ERROR", "LS_CONNECTION_ERROR", "LS_BACKEND_ERROR", "BACKEND_NOT_SET"
                ]
            ):
                return tool_output.split(":", 1)[1].strip() if ":" in tool_output else tool_output

            # Model list: return directly so the LLM's subsequent turns are purely advisory
            # (no operational "present the list" framing competing with informational responses).
            if fn_name == "list_model_sources_and_models":
                messages.append({
                    "role": "system",
                    "content": (
                        "MODEL_LIST_SHOWN: The model list has been presented. "
                        "Do NOT call configure_auto_labeling until the user explicitly names a model."
                    ),
                })
                if "DATASET_NOT_CONFIRMED" not in tool_output:
                    return self._format_model_list(tool_output)

            # Hyperparams: return defaults directly so the LLM's subsequent turns are purely
            # advisory — no "present defaults" operational framing competing with source attribution.
            if fn_name == "configure_auto_labeling" and "Invalid model" not in tool_output:
                messages.append({
                    "role": "system",
                    "content": (
                        "HYPERPARAM_STEP: Model configuration is complete and defaults have been shown. "
                        "Do NOT call set_auto_labeling_hyperparams until the user explicitly requests "
                        "changes or confirms they want to apply specific values."
                    ),
                })
                d = self.hyperparam_cache
                return (
                    f"The model has been configured successfully. "
                    f"Here are the default hyperparameters:\n\n"
                    f"- mode: {d['mode']}\n"
                    f"- epochs: {d['epochs']}\n"
                    f"- early_stop_patience: {d['early_stop_patience']}\n"
                    f"- early_stop_threshold: {d['early_stop_threshold']}\n"
                    f"- learning_rate: {d['learning_rate']}\n"
                    f"- weight_decay: {d['weight_decay']}\n"
                    f"- max_grad_norm: {d['max_grad_norm']}\n\n"
                    f"Would you like to modify any of these hyperparameters before we start?"
                )

            if fn_name == "import_from_label_studio" and "LS_NO_ANNOTATIONS" in tool_output:
                return tool_output.split(":", 1)[1].strip() if ":" in tool_output else tool_output

            if fn_name == "import_from_label_studio":
                return self._finalize_import_from_label_studio(fn_args, tool_output)

            if fn_name == "run_auto_labeling":
                return await self._finalize_auto_labeling(tool_output)

            if fn_name == "import_from_cvat":
                return self._finalize_import_from_cvat(fn_args, tool_output)

            if fn_name == "run_class_mapping":
                return await self._format_class_mapping_reply(tool_output)

            if fn_name == "run_anomaly_detection":
                return await self._format_anomaly_detection_reply(tool_output)

            if fn_name == "run_zero_shot_auto_labeling":
                return self._format_zero_shot_reply(tool_output)

            if fn_name == "run_ensemble_selection":
                return self._format_ensemble_reply(tool_output)

            if fn_name == "run_embedding_selection":
                pass  # Falls through to final LLM summarization.

            if fn_name == "set_selected_dataset" and "DATASET_NOT_FOUND" in tool_output:
                return await self._dataset_not_found_reply()

            if fn_name in ("select_workflow", "switch_workflow"):
                all_fn_names = [r["name"] for r in tool_results]
                if "list_datasets" not in all_fn_names:
                    return await self._fetch_and_return_dataset_list()

        return None

    # Reply formatters

    _MODEL_SOURCE_LABELS = {
        "ultralytics":               "Ultralytics",
        "hf_models_objectdetection": "Hugging Face Models for Object Detection",
        "custom_codetr":             "Custom Code Models",
        "roboflow":                  "Roboflow",
    }

    def _format_model_list(self, tool_output: str) -> str:
        try:
            models = json.loads(tool_output)
            lines = []
            counter = 1
            for source_key, model_list in models.items():
                label = self._MODEL_SOURCE_LABELS.get(source_key, source_key)
                lines.append(f"\n**{label}:**")
                for model in model_list:
                    lines.append(f"{counter}. {model}")
                    counter += 1
            return "\n".join(lines) + "\n\nWhich model would you like to use?"
        except Exception:
            return f"{tool_output}\n\nWhich model would you like to use?"

    async def _finalize_auto_labeling(self, tool_output: str) -> str:
        if any(phrase in tool_output for phrase in [
            "No dataset has been confirmed",
            "model source and model must be configured",
            "Hyperparameters must be confirmed",
        ]):
            return tool_output

        if self.state.auto_labeling:
            self.state.auto_labeling.auto_labeling_complete = True
            self.state.save()

        reply = await self._format_auto_labeling_reply(tool_output)
        dataset_name = self.state.dataset_name

        if dataset_name:
            backend = (self.state.auto_labeling.labeling_backend if self.state.auto_labeling else "cvat") or "cvat"

            if self._progress_cb:
                backend_label = "Label Studio" if backend == "label_studio" else "CVAT"
                await self._progress_cb("status", {"message": f"Exporting predictions to {backend_label}..."})

            async with Client(self.transport) as export_client:
                try:
                    if backend == "label_studio":
                        export_result = await export_client.call_tool(
                            "export_to_label_studio",
                            {"dataset_name": dataset_name, "with_predictions": True}
                        )
                        export_msg = unwrap_tool_output(export_result)
                        if self.state.auto_labeling and "Project ID" in export_msg:
                            try:
                                from pathlib import Path as _Path
                                import json as _json
                                tasks_file = _Path(__file__).resolve().parents[1] / "output" / "ls_tasks.json"
                                if tasks_file.exists():
                                    reg = _json.loads(tasks_file.read_text())
                                    if dataset_name in reg:
                                        self.state.auto_labeling.ls_task_ids = reg[dataset_name].get("task_ids", [])
                                        self.state.save()
                            except Exception:
                                pass
                        reply += (
                            f"\n\n{export_msg}"
                            f"\n\nPlease review and correct the predictions in Label Studio. "
                            f"Let me know when you're done and I'll import the labels back."
                        )
                    else:
                        export_result = await export_client.call_tool(
                            "export_to_cvat",
                            {"dataset_name": dataset_name, "with_predictions": True}
                        )
                        export_msg = unwrap_tool_output(export_result)
                        if self.state.auto_labeling and "Task ID:" in export_msg:
                            try:
                                task_id = int(export_msg.split("Task ID:")[1].split()[0].strip())
                                self.state.auto_labeling.cvat_task_id = task_id
                                self.state.save()
                            except Exception:
                                pass
                        reply += (
                            f"\n\n{export_msg}"
                            f"\n\nPlease review and correct the predictions in CVAT. "
                            f"Let me know when you're done and I'll import the labels back."
                        )
                except Exception as e:
                    reply += f"\n\nNote: {backend} export failed: {str(e)}"
        else:
            reply += "\n\nNote: Could not determine dataset name for export."

        return reply

    def _finalize_import_from_cvat(self, fn_args: dict, tool_output: str) -> str:
        base_dataset = fn_args.get("dataset_name", "").removesuffix("_labeled")
        labeled_name = f"{base_dataset}_labeled" if base_dataset else ""
        try:
            self.state.labeled_dataset_name = labeled_name
            if self.state.auto_labeling:
                self.state.auto_labeling.labels_imported = True
            self.state.save()
        except Exception:
            pass
        return f"{tool_output.strip()}\n\nWould you like to visualize the labeled dataset in Voxel51?"

    def _finalize_import_from_label_studio(self, fn_args: dict, tool_output: str) -> str:
        base_dataset = fn_args.get("dataset_name", "").removesuffix("_labeled")
        labeled_name = f"{base_dataset}_labeled" if base_dataset else ""
        try:
            self.state.labeled_dataset_name = labeled_name
            if self.state.auto_labeling:
                self.state.auto_labeling.labels_imported = True
            self.state.save()
        except Exception:
            pass
        return f"{tool_output.strip()}\n\nWould you like to visualize the labeled dataset in Voxel51?"

    async def _format_auto_labeling_reply(self, tool_output: str) -> str:
        if "precision" in tool_output and "recall" in tool_output and "f1-score" in tool_output:
            summary = await self.llm.summarize_classification_report(tool_output)
            return (
                f"{summary}\n[source: tool result — run_auto_labeling]\n\n"
                f"Full Classification Report:\n"
                f"```\n{tool_output.strip()}\n```"
                f"Would you like to launch Voxel51 to explore the results?"
            )
        return tool_output.strip()

    async def _format_class_mapping_reply(self, tool_output: str) -> str:
        summary = await self.llm.summarize_class_mapping_output(tool_output)
        return (
            f"{summary}\n[source: tool result — run_class_mapping]\n\n"
            f"Class Mapping Output:\n```\n{tool_output.strip()}\n```"
        )

    async def _format_anomaly_detection_reply(self, tool_output: str) -> str:
        summary = await self.llm.summarize_anomaly_detection_output(tool_output)
        return (
            f"{summary}\n[source: tool result — run_anomaly_detection]\n\n"
            f"Anomaly Detection Output:\n```\n{tool_output.strip()}\n```"
        )

    def _format_zero_shot_reply(self, tool_output: str) -> str:
        return (
            f"{tool_output.strip()}\n"
            f"You can now use the Ensemble Selection workflow to identify detections "
            f"where multiple models agree.\n"
            f"Would you like to launch Voxel51 to explore the results?"
        )

    def _format_ensemble_reply(self, tool_output: str) -> str:
        return (
            f"{tool_output.strip()}\n\n"
            f"Would you like to launch Voxel51 to explore the results?\n\n"
            f"- In the ENSEMBLE SELECTION section of the left sidebar, use the "
            f"`n_unique_ensemble_selection` field as a filter. "
            f"- It represents the number of overlapping objects retained in each "
            f"sample based on model agreement. "
            f"- Once you select a sample image, use the `detections_overlap` tag "
            f"from the TAGS panel to visualize only those detections that had "
            f"sufficient overlap and were retained by the ensemble logic."
        )

    async def _dataset_not_found_reply(self) -> str:
        async with Client(self.transport) as list_client:
            try:
                list_result = await list_client.call_tool("list_datasets", {})
                list_output = unwrap_tool_output(list_result)
                return (
                    f"That dataset name wasn't recognized. "
                    f"Here are the available datasets:\n\n{list_output}\n\n"
                    f"Please select the correct name or re-ingest if needed."
                )
            except Exception as e:
                return f"Dataset not found and couldn't fetch the list: {e}"

    async def _fetch_and_return_dataset_list(self) -> str:
        """
        Called after select_workflow or switch_workflow when list_datasets was not
        called in the same turn. Fetches and returns the dataset list directly so
        the final LLM pass (tools=None) has the data it needs to show the user.
        """
        import json as _json
        async with Client(self.transport) as mcp_client:
            try:
                raw = unwrap_tool_output(await mcp_client.call_tool("list_datasets", {}))
            except Exception as e:
                return f"Workflow switched. Could not fetch datasets: {e}"
        try:
            datasets = _json.loads(raw)
            if isinstance(datasets, list):
                raw = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(datasets))
        except Exception:
            pass
        return (
            f"Here are the available datasets:\n\n{raw}\n\n"
            "Which dataset would you like to use? If you'd like to use your own "
            "dataset, please use the **data ingestion window** on the right to "
            "upload it first (supported formats: raw images, videos, COCO, YOLO, "
            "CVAT-xml)."
        )