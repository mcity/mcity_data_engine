# mcp_layer/chat_pipeline.py

import ast
import asyncio
import json
import logging
import importlib
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


# Output normalization

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


# ChatPipeline

class ChatPipeline:
    """
    Owns all processing between the HTTP endpoint and the MCP tools:
    - WorkflowState loaded fresh each request as single source of truth
    - Tool dispatch, pre-call precondition enforcement, immediate message appending
    - Post-call reply building

    Design invariant: every tool call is appended to `messages` immediately
    after it executes in `_dispatch`. This makes it structurally impossible
    for OpenAI to receive an assistant message with tool_call_ids that have
    no corresponding tool result messages.
    """

    def __init__(self, mcp_transport: SSETransport, llm):
        self.transport = mcp_transport
        self.llm = llm

        # WorkflowState is loaded fresh at start of run() — single source of truth
        # for all workflow/dataset/step state across requests.
        self.state: WorkflowState = WorkflowState()

        # Per-request hyperparam caches — these are not persisted to config.py
        # by WorkflowState (they live in the WORKFLOWS section of config.py,
        # written directly by the mcptools). They remain as in-request accumulators.
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

        # Set by _handle_set_selected_dataset when dataset is confirmed.
        # Consumed by _build_reply to inject CURRENT_DATASET system message
        # after all tool results are appended (OpenAI ordering requirement).
        self._confirmed_dataset: str | None = None

    # Public entry point

    async def run(self, tool_calls: list, messages: list) -> tuple[list, str | None]:
        """
        Process all tool calls for one request.

        Loads WorkflowState fresh from config.py at the start of every request
        so state changes from previous requests are always visible.

        Returns:
            (tool_results, early_reply)
        """
        # Load fresh state at start of every request
        self.state = WorkflowState.load()

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

                # Validate tool arguments against registered input contract before dispatch
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

                result = await self._dispatch(
                    fn_name, fn_args, call, mcp_client, messages
                )
                tool_results.append({
                    "tool_call_id": call.id,
                    "name": fn_name,
                    "fn_args": fn_args,
                    "result": result,
                })

        # If set_selected_dataset and select/switch_workflow fired in the same
        # batch, the workflow reset in _handle_select_or_switch_workflow wipes
        # dataset_confirmed. Re-apply it now if both succeeded.
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

    # Dispatch — execute tool, enforce preconditions, append to messages

    async def _dispatch(self, fn_name, fn_args, call, mcp_client, messages) -> str:
        """
        Execute a single tool call with precondition checks and state updates.
        Always appends result to messages before returning.
        """
        try:
            if fn_name == "send_reply":
                content = fn_args.get("message", "")
                result = content

            elif fn_name in ("select_workflow", "switch_workflow"):
                result = await self._handle_select_or_switch_workflow(
                    fn_name, fn_args, mcp_client
                )
                content = result

            elif fn_name == "set_selected_dataset":
                result = await self._handle_set_selected_dataset(fn_args, mcp_client)
                content = result

            elif fn_name == "list_model_sources_and_models":
                # Guard: dataset must be confirmed before listing models
                if not self.state.dataset_confirmed or not self.state.dataset_name:
                    result = (
                        "DATASET_NOT_CONFIRMED: A dataset must be confirmed before "
                        "selecting a model. Please call set_selected_dataset first."
                    )
                    content = result
                else:
                    # Listing models implies auto generated labeling path — infer it.
                    if self.state.auto_labeling is None:
                        self.state.auto_labeling = AutoLabelingState()
                    if not self.state.auto_labeling.labeling_path:
                        self.state.auto_labeling.labeling_path = "auto"
                        self.state.save()
                    result = unwrap_tool_output(
                        await mcp_client.call_tool(fn_name, fn_args)
                    )
                    content = result

            elif fn_name == "configure_auto_labeling":
                result = await self._handle_configure_auto_labeling(fn_args, mcp_client)
                content = result

            elif fn_name == "set_auto_labeling_hyperparams":
                result = await self._handle_set_auto_labeling_hyperparams(
                    fn_args, mcp_client
                )
                content = result

            # --- Class mapping ---
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

            # --- Anomaly detection ---
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

            # --- Embedding selection ---
            elif fn_name == "configure_embedding_selection_model":
                result = await self._handle_configure_embedding_selection_model(fn_args, mcp_client)
                content = result

            elif fn_name == "set_embedding_selection_params":
                result = await self._handle_set_embedding_selection_params(fn_args, mcp_client)
                content = result

            elif fn_name == "run_embedding_selection":
                result = await self._handle_run_embedding_selection(mcp_client)
                content = result

            # --- Zero-shot auto-labeling ---
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

            # --- Ensemble selection ---
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
                result = await self._handle_run_auto_labeling(fn_args, mcp_client)
                content = result

            elif fn_name == "import_from_cvat":
                result = await self._handle_import_from_cvat(fn_args, mcp_client)
                content = result

            elif fn_name == "launch_voxel51_session":
                result = await self._handle_launch_voxel51(fn_args, mcp_client)
                content = result

            else:
                logging.warning(f"[PIPELINE] Calling MCP tool: {fn_name} with args: {fn_args}")
                result = unwrap_tool_output(
                    await mcp_client.call_tool(fn_name, fn_args)
                )
                logging.warning(f"[PIPELINE] MCP tool {fn_name} returned")
                content = result

        except Exception as e:
            logging.warning(f"[PIPELINE] Exception in {fn_name}: {e}")
            result = str(e)
            content = result

        # Always append immediately — message history is never left incomplete
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": fn_name,
            "content": content,
        })
        logging.warning(
            f"[PIPELINE] Appended tool result: name={fn_name} "
            f"tool_call_id={call.id}"
        )

        return result

    # Per-tool handlers

    async def _handle_select_or_switch_workflow(
        self, fn_name: str, fn_args: dict, mcp_client
    ) -> str:
        """
        Reset state completely for the new workflow, then call the MCP tool.
        Handles both select_workflow and switch_workflow identically.
        Checks WORKFLOW_DEPENDENCIES before allowing the workflow to start.
        """
        workflow_name = fn_args.get("workflow_name", "")

        # Check workflow dependencies — e.g. ensemble_selection requires zero_shot first
        deps = WORKFLOW_DEPENDENCIES.get(workflow_name, [])
        if deps:
            # Determine which workflows have been completed this session
            # A workflow substate exists and has been used if its substate is non-None
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

        # Full reset — all prior state cleared, new workflow initialised
        self.state = self.state.reset_for_workflow(workflow_name)
        result = unwrap_tool_output(
            await mcp_client.call_tool(fn_name, fn_args)
        )
        return result

    async def _handle_set_selected_dataset(self, fn_args: dict, mcp_client) -> str:
        ok, msg = self.state.can_confirm_dataset()
        if not ok:
            return msg

        dataset_name = fn_args["dataset_name"]

        raw = await mcp_client.call_tool("set_selected_dataset", {
            "dataset_name": dataset_name
        })
        tool_output = unwrap_tool_output(raw)

        if "DATASET_NOT_FOUND" in tool_output:
            # Dataset may have just been ingested and not yet visible in the
            # FiftyOne registry cache. Retry up to 3 times with increasing
            # waits before giving up — handles the race condition silently.
            logging.warning(
                f"[PIPELINE] Dataset '{dataset_name}' not found — retrying up to 3 times"
            )
            for attempt, wait in enumerate([2, 4, 6], start=1):
                await asyncio.sleep(wait)
                raw = await mcp_client.call_tool("set_selected_dataset", {
                    "dataset_name": dataset_name
                })
                tool_output = unwrap_tool_output(raw)
                if "DATASET_NOT_FOUND" not in tool_output:
                    logging.warning(
                        f"[PIPELINE] Dataset '{dataset_name}' found on retry {attempt}"
                    )
                    break
                logging.warning(
                    f"[PIPELINE] Dataset '{dataset_name}' still not found after retry {attempt}"
                )

        if "DATASET_NOT_FOUND" not in tool_output:
            self.state.dataset_name = dataset_name
            self.state.dataset_confirmed = True
            self.state.save()
            # System message injected in _build_reply after all tool results
            # are appended — injecting here would break OpenAI message ordering.
            self._confirmed_dataset = dataset_name
        else:
            self.state.dataset_confirmed = False

        return tool_output

    async def _handle_configure_auto_labeling(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        # Configuring a model implies auto generated labeling path — infer it.
        if not self.state.auto_labeling.labeling_path:
            self.state.auto_labeling.labeling_path = "auto"
            self.state.save()

        result = unwrap_tool_output(
            await mcp_client.call_tool("configure_auto_labeling", fn_args)
        )
        self.state.auto_labeling.model_configured = True
        # Hyperparams are confirmed by default after model configuration.
        # If the user changes values, set_auto_labeling_hyperparams will
        # be called and hyperparams_confirmed stays True. If the user
        # accepts defaults, no tool is called — but confirmation is implicit.
        self.state.auto_labeling.hyperparams_confirmed = True
        self.state.save()
        return result

    async def _handle_set_auto_labeling_hyperparams(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        for k, v in fn_args.items():
            if v is not None:
                self.hyperparam_cache[k] = v

        result = unwrap_tool_output(
            await mcp_client.call_tool(
                "set_auto_labeling_hyperparams", self.hyperparam_cache.copy()
            )
        )
        self.state.auto_labeling.hyperparams_confirmed = True
        self.state.save()
        return result

    # --- Class mapping handlers ---

    async def _handle_configure_class_mapping_model(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("configure_class_mapping_model", fn_args)
        )
        self.state.class_mapping.model_configured = True
        self.state.save()
        return result

    async def _handle_set_class_mapping_dataset_source(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_class_mapping_dataset_source", fn_args)
        )
        self.state.class_mapping.source_dataset_set = True
        self.state.save()
        return result

    async def _handle_set_class_mapping_dataset_target(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_class_mapping_dataset_target", fn_args)
        )
        self.state.class_mapping.target_dataset_set = True
        self.state.save()
        return result

    async def _handle_set_class_mapping_candidate_labels(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_class_mapping_candidate_labels", fn_args)
        )
        self.state.class_mapping.candidate_labels_set = True
        self.state.save()
        return result

    async def _handle_run_class_mapping(self, mcp_client) -> str:
        if self.state.class_mapping is None:
            self.state.class_mapping = ClassMappingState()
        ok, msg = self.state.class_mapping.can_run_class_mapping(
            self.state.dataset_confirmed
        )
        if not ok:
            return msg
        return unwrap_tool_output(await mcp_client.call_tool("run_class_mapping", {}))

    # --- Anomaly detection handlers ---

    async def _handle_configure_anomaly_detection_model(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.anomaly_detection is None:
            self.state.anomaly_detection = AnomalyDetectionState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("configure_anomaly_detection_model", fn_args)
        )
        self.state.anomaly_detection.model_configured = True
        self.state.save()
        return result

    async def _handle_set_anomaly_detection_data_source(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.anomaly_detection is None:
            self.state.anomaly_detection = AnomalyDetectionState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_anomaly_detection_data_source", fn_args)
        )
        self.state.anomaly_detection.data_source_set = True
        self.state.save()
        return result

    async def _handle_set_anomaly_detection_hyperparams(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.anomaly_detection is None:
            self.state.anomaly_detection = AnomalyDetectionState()
        for k, v in fn_args.items():
            if v is not None:
                self.hyperparam_cache_anomaly[k] = v
        result = unwrap_tool_output(
            await mcp_client.call_tool(
                "set_anomaly_detection_hyperparams",
                self.hyperparam_cache_anomaly.copy()
            )
        )
        self.state.anomaly_detection.hyperparams_confirmed = True
        self.state.save()
        return result

    async def _handle_run_anomaly_detection(self, mcp_client) -> str:
        if self.state.anomaly_detection is None:
            self.state.anomaly_detection = AnomalyDetectionState()
        ok, msg = self.state.anomaly_detection.can_run_anomaly_detection(
            self.state.dataset_confirmed
        )
        if not ok:
            return msg
        return unwrap_tool_output(
            await mcp_client.call_tool("run_anomaly_detection", {})
        )

    # --- Embedding selection handlers ---

    async def _handle_configure_embedding_selection_model(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.embedding_selection is None:
            self.state.embedding_selection = EmbeddingSelectionState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("configure_embedding_selection_model", fn_args)
        )
        self.state.embedding_selection.model_configured = True
        self.state.save()
        return result

    async def _handle_set_embedding_selection_params(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.embedding_selection is None:
            self.state.embedding_selection = EmbeddingSelectionState()
        for k, v in fn_args.items():
            if v is not None:
                self.embedding_selection_cache[k] = v
        result = unwrap_tool_output(
            await mcp_client.call_tool(
                "set_embedding_selection_params",
                self.embedding_selection_cache.copy()
            )
        )
        self.state.embedding_selection.params_set = True
        self.state.save()
        return result

    async def _handle_run_embedding_selection(self, mcp_client) -> str:
        if self.state.embedding_selection is None:
            self.state.embedding_selection = EmbeddingSelectionState()
        ok, msg = self.state.embedding_selection.can_run_embedding_selection(
            self.state.dataset_confirmed
        )
        if not ok:
            return msg
        return unwrap_tool_output(
            await mcp_client.call_tool("run_embedding_selection", {})
        )

    # --- Zero-shot auto-labeling handlers ---

    async def _handle_configure_zero_shot_models(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.auto_labeling_zero_shot is None:
            self.state.auto_labeling_zero_shot = ZeroShotAutoLabelingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool(
                "configure_auto_labeling_zero_shot_models", fn_args
            )
        )
        self.state.auto_labeling_zero_shot.models_configured = True
        self.state.save()
        return result

    async def _handle_set_zero_shot_threshold(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.auto_labeling_zero_shot is None:
            self.state.auto_labeling_zero_shot = ZeroShotAutoLabelingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool(
                "set_auto_labeling_zero_shot_threshold", fn_args
            )
        )
        self.state.auto_labeling_zero_shot.threshold_set = True
        self.state.save()
        return result

    async def _handle_set_zero_shot_classes(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.auto_labeling_zero_shot is None:
            self.state.auto_labeling_zero_shot = ZeroShotAutoLabelingState()
        result = unwrap_tool_output(
            await mcp_client.call_tool(
                "set_auto_labeling_zero_shot_classes", fn_args
            )
        )
        self.state.auto_labeling_zero_shot.classes_set = True
        self.state.save()
        return result

    async def _handle_run_zero_shot(self, mcp_client) -> str:
        if self.state.auto_labeling_zero_shot is None:
            self.state.auto_labeling_zero_shot = ZeroShotAutoLabelingState()
        ok, msg = self.state.auto_labeling_zero_shot.can_run_zero_shot(
            self.state.dataset_confirmed
        )
        if not ok:
            return msg
        return unwrap_tool_output(
            await mcp_client.call_tool("run_zero_shot_auto_labeling", {})
        )

    # --- Ensemble selection handlers ---

    async def _handle_set_ensemble_selection_parameters(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.ensemble_selection is None:
            self.state.ensemble_selection = EnsembleSelectionState()
        for k, v in fn_args.items():
            if v is not None:
                self.ensemble_selection_cache[k] = v
        result = unwrap_tool_output(
            await mcp_client.call_tool(
                "set_ensemble_selection_parameters",
                self.ensemble_selection_cache.copy()
            )
        )
        self.state.ensemble_selection.params_set = True
        self.state.save()
        return result

    async def _handle_set_ensemble_classes(
        self, fn_args: dict, mcp_client
    ) -> str:
        if self.state.ensemble_selection is None:
            self.state.ensemble_selection = EnsembleSelectionState()
        result = unwrap_tool_output(
            await mcp_client.call_tool("set_ensemble_selection_classes", fn_args)
        )
        self.state.ensemble_selection.classes_set = True
        self.state.save()
        return result

    async def _handle_run_ensemble_selection(self, mcp_client) -> str:
        if self.state.ensemble_selection is None:
            self.state.ensemble_selection = EnsembleSelectionState()
        ok, msg = self.state.ensemble_selection.can_run_ensemble_selection(
            self.state.dataset_confirmed
        )
        if not ok:
            return msg
        return unwrap_tool_output(
            await mcp_client.call_tool("run_ensemble_selection", {})
        )

    async def _handle_export_to_cvat(self, fn_args: dict, mcp_client) -> str:
        with_predictions = fn_args.get("with_predictions", False)

        if self.state.workflow_name == "auto_labeling":
            if self.state.auto_labeling is None:
                self.state.auto_labeling = AutoLabelingState()

            # If labeling path not yet set, infer it from with_predictions
            if not self.state.auto_labeling.labeling_path:
                inferred = "manual" if not with_predictions else "auto"
                self.state.auto_labeling.labeling_path = inferred
                self.state.save()

            ok, msg = self.state.auto_labeling.can_export_to_cvat(with_predictions)
            if not ok:
                return msg

        result = unwrap_tool_output(
            await mcp_client.call_tool("export_to_cvat", fn_args)
        )

        # Extract and persist task_id from the result string
        if self.state.auto_labeling and "Task ID:" in result:
            try:
                task_id = int(result.split("Task ID:")[1].split()[0].strip())
                self.state.auto_labeling.cvat_task_id = task_id
                self.state.save()
            except Exception:
                pass

        return result

    async def _handle_run_auto_labeling(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling is None:
            self.state.auto_labeling = AutoLabelingState()

        ok, msg = self.state.auto_labeling.can_run_auto_labeling(
            self.state.dataset_confirmed,
            self.state.dataset_name,
        )
        if not ok:
            return msg

        return unwrap_tool_output(
            await mcp_client.call_tool("run_auto_labeling", fn_args)
        )

    async def _handle_import_from_cvat(self, fn_args: dict, mcp_client) -> str:
        if self.state.auto_labeling:
            ok, msg = self.state.auto_labeling.can_import_from_cvat()
            if not ok:
                return msg
        return unwrap_tool_output(
            await mcp_client.call_tool("import_from_cvat", fn_args)
        )

    async def _handle_launch_voxel51(self, fn_args: dict, mcp_client) -> str:
        dataset_name = fn_args.get("dataset_name", "").strip()
        if not dataset_name:
            dataset_name = (
                self.state.labeled_dataset_name.strip()
                or self.state.dataset_name.strip()
            )
        if not dataset_name:
            return (
                "Cannot launch Voxel51: no dataset name was provided or "
                "found in session state. Please specify the dataset name."
            )
        fn_args["dataset_name"] = dataset_name
        logging.warning(f"[PIPELINE] launch_voxel51_session -> dataset='{dataset_name}'")
        return unwrap_tool_output(
            await mcp_client.call_tool("launch_voxel51_session", fn_args)
        )

    # Reply building — no message mutation except CURRENT_DATASET injection

    async def _build_reply(self, tool_results: list, messages: list) -> str | None:
        """
        Inject any deferred system messages, then scan tool results for
        early-return tools. Returns a reply string or None for final LLM pass.
        """
        # Inject CURRENT_DATASET after all tool results are in messages —
        # injecting earlier would violate OpenAI's message ordering rules.
        # Also re-inject on every request where dataset is confirmed so the
        # LLM doesn't lose track of it as message history grows.
        confirmed_name = self._confirmed_dataset or (
            self.state.dataset_name if self.state.dataset_confirmed else None
        )
        if confirmed_name:
            messages.append({
                "role": "system",
                "content": f"CURRENT_DATASET: {confirmed_name}"
            })
            self._confirmed_dataset = None

        for result in tool_results:
            fn_name = result["name"]
            fn_args = result.get("fn_args", {})
            tool_output = unwrap_tool_output(result.get("result", ""))

            if fn_name == "export_to_cvat" and any(
                sentinel in tool_output for sentinel in [
                    "CVAT_TASK_LIMIT_REACHED", "CVAT_FORBIDDEN",
                    "CVAT_AUTH_ERROR", "CVAT_NOT_FOUND", "CVAT_CONNECTION_ERROR"
                ]
            ):
                # Strip the sentinel prefix and return clean message
                clean = tool_output.split(":", 1)[1].strip() if ":" in tool_output else tool_output
                return clean

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
                # Embedding selection has no special formatter — falls through
                # to final LLM summarization
                pass

            if fn_name == "set_selected_dataset" and "DATASET_NOT_FOUND" in tool_output:
                return await self._dataset_not_found_reply()

        return None

    # Reply formatters

    async def _finalize_auto_labeling(self, tool_output: str) -> str:
        # Block if precondition was not met (guard returned an error string)
        if any(phrase in tool_output for phrase in [
            "No dataset has been confirmed",
            "model source and model must be configured",
            "Hyperparameters must be confirmed",
        ]):
            return tool_output

        # Mark auto-labeling complete and persist
        if self.state.auto_labeling:
            self.state.auto_labeling.auto_labeling_complete = True
            self.state.save()

        reply = await self._format_auto_labeling_reply(tool_output)
        dataset_name = self.state.dataset_name

        if dataset_name:
            async with Client(self.transport) as export_client:
                try:
                    export_result = await export_client.call_tool(
                        "export_to_cvat",
                        {"dataset_name": dataset_name, "with_predictions": True}
                    )
                    export_msg = unwrap_tool_output(export_result)
                    # Extract and persist task_id
                    if self.state.auto_labeling and "Task ID:" in export_msg:
                        try:
                            task_id = int(
                                export_msg.split("Task ID:")[1].split()[0].strip()
                            )
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
                    reply += f"\n\nNote: CVAT export failed: {str(e)}"
        else:
            reply += "\n\nNote: Could not determine dataset name for CVAT export."

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
        return (
            f"{tool_output.strip()}\n\n"
            f"Would you like to visualize the labeled dataset in Voxel51?"
        )

    async def _format_auto_labeling_reply(self, tool_output: str) -> str:
        if (
            "precision" in tool_output
            and "recall" in tool_output
            and "f1-score" in tool_output
        ):
            summary = await self.llm.summarize_classification_report(tool_output)
            return (
                f"{summary}\n\n"
                f"Full Classification Report:\n"
                f"```\n{tool_output.strip()}\n```"
                f"Would you like to launch Voxel51 to explore the results?"
            )
        return tool_output.strip()

    async def _format_class_mapping_reply(self, tool_output: str) -> str:
        summary = await self.llm.summarize_class_mapping_output(tool_output)
        return (
            f"{summary}\n\n"
            f"Class Mapping Output:\n"
            f"```\n{tool_output.strip()}\n```"
        )

    async def _format_anomaly_detection_reply(self, tool_output: str) -> str:
        summary = await self.llm.summarize_anomaly_detection_output(tool_output)
        return (
            f"{summary}\n\n"
            f"Anomaly Detection Output:\n"
            f"```\n{tool_output.strip()}\n```"
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