# mcp_layer/chat_pipeline.py

import ast
import json
import logging
import importlib
import re
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import SSETransport

import config.config as _cc
from config.config import WORKFLOW_STATE_DEFAULT

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.py"

# Workflow state — persisted to config.py so it survives across requests

def read_workflow_state() -> dict:
    try:
        importlib.reload(_cc)
        return dict(_cc.WORKFLOW_STATE)
    except Exception:
        return dict(WORKFLOW_STATE_DEFAULT)


def write_workflow_state(state: dict) -> None:
    try:
        src = CONFIG_PATH.read_text()
        tree = ast.parse(src)
        lines = src.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "WORKFLOW_STATE":
                        start = node.lineno - 1
                        end = node.end_lineno
                        lines[start:end] = [f"WORKFLOW_STATE = {repr(state)}"]
                        CONFIG_PATH.write_text("\n".join(lines) + "\n")
                        return
    except Exception as e:
        logging.warning(f"[STATE] Failed to write WORKFLOW_STATE: {e}")


def update_workflow_state(**kwargs) -> dict:
    state = read_workflow_state()
    state.update(kwargs)
    write_workflow_state(state)
    return state


def reset_workflow_state() -> dict:
    try:
        defaults = dict(WORKFLOW_STATE_DEFAULT)
        write_workflow_state(defaults)
        return defaults
    except Exception as e:
        logging.warning(f"[STATE] Failed to reset WORKFLOW_STATE: {e}")
        return {}


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
    - per-request state and cache management
    - tool dispatch, pre-call guards, and immediate message appending
    - post-call reply building
    - workflow state persistence

    Design invariant: every tool call is appended to `messages` immediately
    after it executes in `_dispatch`, before any reply logic runs. This makes
    it structurally impossible for OpenAI to receive an assistant message with
    tool_call_ids that have no corresponding tool result messages.
    """

    def __init__(self, mcp_transport: SSETransport, llm):
        self.transport = mcp_transport
        self.llm = llm

        # Per-request caches — fresh instance created per request in chat_server
        self.conversation_state = {
            "workflow_name": None,
            "dataset_selected": False,
            "auto_labeling_complete": False,
        }
        self.selected_dataset_cache = {"dataset_name": "fisheye8k_mini", "n_samples": None}
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
        # Set by _handle_set_selected_dataset when dataset is confirmed;
        # consumed by _build_reply to inject CURRENT_DATASET system message
        # after all tool results are appended (OpenAI ordering requirement).
        self._confirmed_dataset: str | None = None


    # Public entry point

    async def run(self, tool_calls: list, messages: list) -> tuple[list, str | None]:
        """
        Process all tool calls for one request.

        Each tool is executed and its result is appended to `messages`
        immediately in `_dispatch`. Once all calls are complete, `_build_reply`
        scans the results and returns an early reply string if appropriate,
        or None to let chat_server do a final LLM summarization pass.

        Returns:
            (tool_results, early_reply)
        """
        tool_results = []

        logging.warning(f"[PIPELINE] Processing {len(tool_calls)} tool call(s): {[c.function.name for c in tool_calls]}")

        async with Client(self.transport) as mcp_client:
            for call in tool_calls:
                fn_name = call.function.name

                try:
                    fn_args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                result = await self._dispatch(
                    fn_name, fn_args, call, mcp_client, messages
                )
                tool_results.append({
                    "tool_call_id": call.id,
                    "name": fn_name,
                    "fn_args": fn_args,
                    "result": result,
                })

        early_reply = await self._build_reply(tool_results, messages)
        return tool_results, early_reply


    # Dispatch — execute tool, update state, append to messages immediately


    async def _dispatch(self, fn_name, fn_args, call, mcp_client, messages) -> str:
        """
        Execute a single tool call, apply any pre/post state updates,
        and immediately append the result to `messages`.

        Returns the raw tool output string.
        """
        try:
            if fn_name == "send_reply":
                # No MCP call — capture the reply text directly
                content = fn_args.get("message", "")
                result = content

            elif fn_name in ("select_workflow", "switch_workflow"):
                self.conversation_state.update({
                    "workflow_name": fn_args.get("workflow_name"),
                    "dataset_selected": False,
                    "auto_labeling_complete": False,
                })
                reset_workflow_state()
                result = unwrap_tool_output(
                    await mcp_client.call_tool(fn_name, fn_args)
                )
                content = result

            elif fn_name == "set_selected_dataset":
                result = await self._handle_set_selected_dataset(
                    fn_args, mcp_client
                )
                content = result

            elif fn_name == "set_auto_labeling_hyperparams":
                for k, v in fn_args.items():
                    if v is not None:
                        self.hyperparam_cache[k] = v
                result = unwrap_tool_output(
                    await mcp_client.call_tool(fn_name, self.hyperparam_cache.copy())
                )
                content = result

            elif fn_name == "set_anomaly_detection_hyperparams":
                for k, v in fn_args.items():
                    if v is not None:
                        self.hyperparam_cache_anomaly[k] = v
                result = unwrap_tool_output(
                    await mcp_client.call_tool(fn_name, self.hyperparam_cache_anomaly.copy())
                )
                content = result

            elif fn_name == "set_embedding_selection_params":
                for k, v in fn_args.items():
                    if v is not None:
                        self.embedding_selection_cache[k] = v
                result = unwrap_tool_output(
                    await mcp_client.call_tool(fn_name, self.embedding_selection_cache.copy())
                )
                content = result

            elif fn_name == "set_ensemble_selection_parameters":
                if not fn_args.get("agreement_threshold"):
                    result = "Please provide the required `agreement_threshold` parameter."
                    content = result
                else:
                    for k, v in fn_args.items():
                        if v is not None:
                            self.ensemble_selection_cache[k] = v
                    result = unwrap_tool_output(
                        await mcp_client.call_tool(fn_name, self.ensemble_selection_cache.copy())
                    )
                    content = result

            elif fn_name == "export_to_cvat":
                result = unwrap_tool_output(
                    await self._handle_export_to_cvat(fn_args, mcp_client)
                )
                content = result

            elif fn_name == "run_auto_labeling":
                result = unwrap_tool_output(
                    await self._handle_run_auto_labeling(fn_args, mcp_client)
                )
                content = result

            elif fn_name == "import_from_cvat":
                result = unwrap_tool_output(
                    await mcp_client.call_tool(fn_name, fn_args)
                )
                content = result

            elif fn_name == "launch_voxel51_session":
                result = unwrap_tool_output(
                    await self._handle_launch_voxel51(fn_args, mcp_client)
                )
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
        logging.warning(f"[PIPELINE] Appended tool result: name={fn_name} tool_call_id={call.id}")

        return result


    # Per-tool handlers

    async def _handle_set_selected_dataset(self, fn_args, mcp_client) -> str:
        self.conversation_state["dataset_selected"] = True
        self.selected_dataset_cache["dataset_name"] = fn_args["dataset_name"]
        self.selected_dataset_cache["n_samples"] = None
        raw = await mcp_client.call_tool("set_selected_dataset", {
            "dataset_name": fn_args["dataset_name"]
        })
        tool_output = unwrap_tool_output(raw)
        if "DATASET_NOT_FOUND" not in tool_output:
            update_workflow_state(
                dataset_confirmed=True,
                dataset_name=fn_args["dataset_name"],
            )
            # Store confirmed name — system message injected after all tool
            # results are appended, to avoid breaking OpenAI message ordering.
            self._confirmed_dataset = fn_args["dataset_name"]
        else:
            self.conversation_state["dataset_selected"] = False
        return tool_output

    async def _handle_export_to_cvat(self, fn_args, mcp_client):
        is_premature = (
            fn_args.get("with_predictions", False)
            and self.conversation_state.get("workflow_name") == "auto_labeling"
            and not self.conversation_state.get("auto_labeling_complete")
        )
        if is_premature:
            return (
                "export_to_cvat cannot be called yet. "
                "The auto-labeling model must be configured and run_auto_labeling "
                "must complete first. Please continue with model configuration."
            )
        return await mcp_client.call_tool("export_to_cvat", fn_args)

    async def _handle_run_auto_labeling(self, fn_args, mcp_client):
        state = read_workflow_state()
        if not state.get("dataset_confirmed", False):
            config_dataset = self._read_dataset_from_config()
            return (
                f"Cannot run auto-labeling: no dataset was confirmed this session. "
                f"Config currently points to '{config_dataset}'. "
                f"Please confirm with the user: is '{config_dataset}' the correct dataset? "
                f"If not, ask them for the correct name and call set_selected_dataset first."
            )
        return await mcp_client.call_tool("run_auto_labeling", fn_args)

    async def _handle_launch_voxel51(self, fn_args, mcp_client):
        dataset_name = fn_args.get("dataset_name", "").strip()
        if not dataset_name:
            state = read_workflow_state()
            dataset_name = (
                state.get("labeled_dataset_name", "").strip()
                or state.get("dataset_name", "").strip()
            )
        if not dataset_name:
            return (
                "Cannot launch Voxel51: no dataset name was provided or "
                "found in session state. Please specify the dataset name."
            )
        fn_args["dataset_name"] = dataset_name
        logging.warning(f"[PIPELINE] launch_voxel51_session -> dataset='{dataset_name}'")
        return await mcp_client.call_tool("launch_voxel51_session", fn_args)


    # Reply building — pure, no message mutation

    async def _build_reply(self, tool_results: list, messages: list) -> str | None:
        """
        Scan completed tool results and return an early reply if any tool
        requires one, otherwise return None for final LLM summarization.

        All tool results are already appended to messages by _dispatch.
        Any extra system messages (e.g. CURRENT_DATASET) are injected here,
        after the tool results, so OpenAI message ordering is never violated.
        """
        # Inject CURRENT_DATASET system message now that all tool results
        # are in messages — injecting it earlier would break OpenAI ordering.
        if self._confirmed_dataset:
            messages.append({
                "role": "system",
                "content": f"CURRENT_DATASET: {self._confirmed_dataset}"
            })
            self._confirmed_dataset = None

        for result in tool_results:
            fn_name = result["name"]
            fn_args = result.get("fn_args", {})
            tool_output = unwrap_tool_output(result.get("result", ""))

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

            if fn_name == "set_selected_dataset" and "DATASET_NOT_FOUND" in tool_output:
                return await self._dataset_not_found_reply()

        return None


    # Reply formatters

    async def _finalize_auto_labeling(self, tool_output: str) -> str:
        self.conversation_state["auto_labeling_complete"] = True

        if "Cannot run auto-labeling" in tool_output:
            return tool_output

        reply = await self._format_auto_labeling_reply(tool_output)
        dataset_name = (
            self._read_dataset_from_config()
            or self.selected_dataset_cache.get("dataset_name", "")
        )

        if dataset_name:
            async with Client(self.transport) as export_client:
                try:
                    export_result = await export_client.call_tool(
                        "export_to_cvat",
                        {"dataset_name": dataset_name, "with_predictions": True}
                    )
                    export_msg = unwrap_tool_output(export_result)
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
        try:
            base_dataset = fn_args.get("dataset_name", "").removesuffix("_labeled")
            labeled_name = f"{base_dataset}_labeled" if base_dataset else ""
            update_workflow_state(labeled_dataset_name=labeled_name)
        except Exception:
            pass
        return f"{tool_output.strip()}\n\nWould you like to visualize the labeled dataset in Voxel51?"

    async def _format_auto_labeling_reply(self, tool_output: str) -> str:
        if "precision" in tool_output and "recall" in tool_output and "f1-score" in tool_output:
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
            f"- It represents the number of overlapping objects retained in each sample "
            f"based on model agreement. "
            f"- Once you select a sample image, use the `detections_overlap` tag from "
            f"the TAGS panel to visualize only those detections that had sufficient "
            f"overlap and were retained by the ensemble logic."
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


    # Utilities

    def _read_dataset_from_config(self) -> str:
        try:
            config_text = open("config/config.py").read()
            m = re.search(
                r'SELECTED_DATASET\s*=\s*\{[^}]*"name":\s*"([^"]+)"',
                config_text
            )
            return m.group(1) if m else ""
        except Exception:
            return ""