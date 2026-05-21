# mcp_layer/chat_server.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Client
from fastmcp.client.transports import SSETransport

import os
from dotenv import load_dotenv
import json
import sys
import os
sys.path.append(os.path.dirname(__file__))
from llm_clients import OpenAIClient, GroqClient, GeminiClient
from tool_schema import tools
import uuid, shutil, tempfile, logging, asyncio, json
from pathlib import Path
from fastapi import UploadFile, File, Form, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
from mcptools.data_ingest import _run_data_ingest_streaming_core
import requests

import ast
import importlib
import sys
from config.config import WORKFLOW_STATE_DEFAULT
import importlib, config.config as _cc


load_dotenv()

llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
llm_map = {
    "openai": OpenAIClient,
    "groq": GroqClient,
    "gemini": GeminiClient
}
llm = llm_map[llm_provider]()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
host = os.getenv("PUBLIC_IP", "localhost")

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.py"

def _read_workflow_state() -> dict:
    try:
        importlib.reload(_cc)
        return dict(_cc.WORKFLOW_STATE)
    except Exception:
        return dict(WORKFLOW_STATE_DEFAULT)

def _write_workflow_state(state: dict):
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
                        new_block = f"WORKFLOW_STATE = {repr(state)}"
                        lines[start:end] = [new_block]
                        CONFIG_PATH.write_text("\n".join(lines) + "\n")
                        return
    except Exception as e:
        logging.warning(f"[STATE] Failed to write WORKFLOW_STATE: {e}")

def _update_workflow_state(**kwargs) -> dict:
    state = _read_workflow_state()
    state.update(kwargs)
    _write_workflow_state(state)
    return state

def _reset_workflow_state() -> dict:
    try:
        defaults = dict(WORKFLOW_STATE_DEFAULT)
        _write_workflow_state(defaults)
        return defaults
    except Exception as e:
        logging.warning(f"[STATE] Failed to reset WORKFLOW_STATE: {e}")
        return {}


def get_imds_token():
    token_url = "http://169.254.169.254/latest/api/token"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
    try:
        response = requests.put(token_url, headers=headers, timeout=2)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error getting token: {e}")
        return None


def get_metadata_with_token(path, token):
    url = f"http://169.254.169.254/latest/meta-data/{path}"
    headers = {"X-aws-ec2-metadata-token": token}
    try:
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching metadata for {path}: {e}")
        return None

token = get_imds_token()
if token:
    host = get_metadata_with_token("public-ipv4", token)
else:
    host = "localhost"
    print("Could not obtain IMDSv2 token.")

url = f"http://{host}:8000/sse"
MCP_TRANSPORT = SSETransport(url=url)

SYSTEM_PROMPT = (Path(__file__).resolve().parent / "prompts" / "system_prompt.txt").read_text()

def unwrap_tool_output(raw):
    """Normalize LLM/MCP outputs to a plain string."""
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

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    # Always require a tool call — the model must call either a real tool or send_reply.
    # This eliminates all intent-detection logic and makes routing fully deterministic.
    assistant_message = await llm.chat(messages, tools=tools, tool_choice="required")

    selected_dataset_cache = {
        "dataset_name": "fisheye8k_mini",
        "n_samples": None
    }

    conversation_state = {
        "workflow_name": None,
        "dataset_selected": False,
        "auto_labeling_complete": False
    }

    hyperparam_cache = {
        "mode": ["train", "inference"],
        "epochs": 10,
        "early_stop_patience": 5,
        "early_stop_threshold": 0,
        "learning_rate": 5e-5,
        "weight_decay": 0.0001,
        "max_grad_norm": 0.01,
    }

    hyperparam_cache_anomaly = {
        "mode": ["train", "inference"],
        "epochs": 12,
        "early_stop_patience": 5,
    }

    embedding_selection_cache = {
        "compute_representativeness": 0.99,
        "compute_unique_images_greedy": 0.01,
        "compute_unique_images_deterministic": 0.99,
        "compute_similar_images": 0.03,
        "neighbour_count": 3
    }

    ensemble_selection_cache = {
        "iou_threshold": 0.5,
        "max_bbox_size": 0.1,
    }

    if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
        tool_calls = assistant_message.tool_calls
        tool_results = []

        # FIX 1 (edge case 2): send_reply fast-path with robust fallback
        if len(tool_calls) == 1 and tool_calls[0].function.name == "send_reply":
            try:
                args = json.loads(tool_calls[0].function.arguments)
                return {"reply": args.get("message", "")}
            except Exception:
                try:
                    # Raw arguments string is better than nothing
                    return {"reply": tool_calls[0].function.arguments}
                except Exception:
                    return {"reply": "Something went wrong. Please try again."}

        async with Client(MCP_TRANSPORT) as mcp_client:
            for call in tool_calls:
                fn_name = call.function.name

                # FIX 1 (edge case 1): send_reply mixed with real tools —
                # append a tool result so the message history stays valid for
                # the final llm.chat call, then skip MCP execution.
                if fn_name == "send_reply":
                    try:
                        args = json.loads(call.function.arguments)
                        reply_text = args.get("message", "")
                    except Exception:
                        reply_text = ""
                    tool_results.append({
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "fn_args": {"message": reply_text},
                        "result": reply_text
                    })
                    continue

                try:
                    fn_args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                try:
                    if fn_name == "select_workflow":
                        conversation_state["workflow_name"] = fn_args["workflow_name"]
                        conversation_state["dataset_selected"] = False
                        conversation_state["auto_labeling_complete"] = False
                        _reset_workflow_state()
                        result = await mcp_client.call_tool(fn_name, fn_args)

                    elif fn_name == "switch_workflow":
                        conversation_state["workflow_name"] = fn_args["workflow_name"]
                        conversation_state["dataset_selected"] = False
                        conversation_state["auto_labeling_complete"] = False
                        _reset_workflow_state()
                        result = await mcp_client.call_tool(fn_name, fn_args)

                    elif fn_name == "set_auto_labeling_hyperparams":
                        for k, v in fn_args.items():
                            if v is not None:
                                hyperparam_cache[k] = v
                        result = await mcp_client.call_tool(fn_name, hyperparam_cache.copy())

                    elif fn_name == "set_selected_dataset":
                        conversation_state["dataset_selected"] = True
                        selected_dataset_cache["dataset_name"] = fn_args["dataset_name"]
                        selected_dataset_cache["n_samples"] = None
                        result = await mcp_client.call_tool(fn_name, {
                            "dataset_name": fn_args["dataset_name"]
                        })
                        tool_output = unwrap_tool_output(result)
                        if "DATASET_NOT_FOUND" not in tool_output:
                            _update_workflow_state(
                                dataset_confirmed=True,
                                dataset_name=fn_args["dataset_name"],
                            )
                            messages.append({
                                "role": "system",
                                "content": f"CURRENT_DATASET: {fn_args['dataset_name']}"
                            })
                        else:
                            conversation_state["dataset_selected"] = False

                    elif fn_name == "set_anomaly_detection_hyperparams":
                        for k, v in fn_args.items():
                            if v is not None:
                                hyperparam_cache_anomaly[k] = v
                        result = await mcp_client.call_tool(fn_name, hyperparam_cache_anomaly.copy())

                    elif fn_name == "set_embedding_selection_params":
                        for k, v in fn_args.items():
                            if v is not None:
                                embedding_selection_cache[k] = v
                        result = await mcp_client.call_tool(fn_name, embedding_selection_cache.copy())

                    elif fn_name == "set_ensemble_selection_parameters":
                        if "agreement_threshold" not in fn_args or fn_args["agreement_threshold"] is None:
                            return {"reply": "Please provide the required `agreement_threshold` parameter."}
                        for k, v in fn_args.items():
                            if v is not None:
                                ensemble_selection_cache[k] = v
                        result = await mcp_client.call_tool(fn_name, ensemble_selection_cache.copy())

                    elif fn_name == "export_to_cvat":
                        is_auto_labeling_premature = (
                            fn_args.get("with_predictions", False) == True
                            and conversation_state.get("workflow_name") == "auto_labeling"
                            and not conversation_state.get("auto_labeling_complete")
                        )
                        if is_auto_labeling_premature:
                            tool_results.append({
                                "tool_call_id": call.id,
                                "name": fn_name,
                                "fn_args": fn_args,
                                "result": (
                                    "export_to_cvat cannot be called yet. "
                                    "The auto-labeling model must be configured and run_auto_labeling "
                                    "must complete first. Please continue with model configuration."
                                )
                            })
                        else:
                            result = await mcp_client.call_tool(fn_name, fn_args)
                            tool_results.append({
                                "tool_call_id": call.id,
                                "name": fn_name,
                                "result": result,
                                "fn_args": fn_args,
                            })
                        continue

                    elif fn_name == "run_auto_labeling":
                        state = _read_workflow_state()
                        dataset_confirmed = state.get("dataset_confirmed", False)

                        if not dataset_confirmed:
                            config_dataset = ""
                            try:
                                import re as _re
                                config_text = open("config/config.py").read()
                                m_cfg = _re.search(
                                    r'SELECTED_DATASET\s*=\s*\{[^}]*"name":\s*"([^"]+)"',
                                    config_text
                                )
                                if m_cfg:
                                    config_dataset = m_cfg.group(1)
                            except Exception:
                                pass

                            tool_results.append({
                                "tool_call_id": call.id,
                                "name": fn_name,
                                "fn_args": fn_args,
                                "result": (
                                    f"Cannot run auto-labeling: no dataset was confirmed this session. "
                                    f"Config currently points to '{config_dataset}'. "
                                    f"Please confirm with the user: is '{config_dataset}' the correct dataset? "
                                    f"If not, ask them for the correct name and call set_selected_dataset first."
                                )
                            })
                            continue

                        result = await mcp_client.call_tool(fn_name, fn_args)
                        tool_results.append({
                            "tool_call_id": call.id,
                            "name": fn_name,
                            "fn_args": fn_args,
                            "result": result
                        })
                        continue

                    # FIX 2 (edge case 5): guard launch_voxel51_session against
                    # empty dataset_name before hitting the MCP tool.
                    elif fn_name == "launch_voxel51_session":
                        dataset_name = fn_args.get("dataset_name", "").strip()
                        if not dataset_name:
                            state = _read_workflow_state()
                            dataset_name = (
                                state.get("labeled_dataset_name", "").strip()
                                or state.get("dataset_name", "").strip()
                            )
                        if not dataset_name:
                            tool_results.append({
                                "tool_call_id": call.id,
                                "name": fn_name,
                                "fn_args": fn_args,
                                "result": (
                                    "Cannot launch Voxel51: no dataset name was provided or "
                                    "found in session state. Please specify the dataset name."
                                )
                            })
                            continue
                        fn_args["dataset_name"] = dataset_name
                        logging.warning(f"[CHAT] Calling MCP tool: launch_voxel51_session with dataset='{dataset_name}'")
                        result = await mcp_client.call_tool(fn_name, fn_args)
                        logging.warning(f"[CHAT] MCP tool launch_voxel51_session returned")

                    else:
                        logging.warning(f"[CHAT] Calling MCP tool: {fn_name} with args: {fn_args}")
                        result = await mcp_client.call_tool(fn_name, fn_args)
                        logging.warning(f"[CHAT] MCP tool {fn_name} returned")

                    tool_results.append({
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "fn_args": fn_args,
                        "result": result
                    })
                except Exception as e:
                    tool_results.append({
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "fn_args": fn_args,
                        "error": str(e)
                    })

        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                } for call in tool_calls
            ]
        })

        for result in tool_results:
            fn_name = result["name"]
            fn_args = result.get("fn_args", {})
            tool_output = str(result.get("result", result.get("error", "Tool error.")))

            # FIX 1 (edge case 1): send_reply tool result must be appended to
            # keep the message history valid when mixed with real tool calls.
            if fn_name == "send_reply":
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": fn_name,
                    "content": fn_args.get("message", "")
                })
                continue

            if fn_name == "run_auto_labeling":
                conversation_state["auto_labeling_complete"] = True
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                tool_output = unwrap_tool_output(tool_output_raw)

                if "Cannot run auto-labeling" in tool_output:
                    return {"reply": tool_output}

                if "precision" in tool_output and "recall" in tool_output and "f1-score" in tool_output:
                    summary = await llm.summarize_classification_report(tool_output)
                    reply = (
                        f"{summary}\n\n"
                        f"Full Classification Report:\n"
                        f"```\n{tool_output.strip()}\n```"
                        f"Would you like to launch Voxel51 to explore the results?"
                    )
                else:
                    reply = f"{tool_output.strip()}"

                dataset_name = ""
                try:
                    import re
                    config_text = open("config/config.py").read()
                    m = re.search(r'SELECTED_DATASET\s*=\s*\{[^}]*"name":\s*"([^"]+)"', config_text)
                    if m:
                        dataset_name = m.group(1)
                except Exception as e:
                    print(f"DEBUG: could not read config.py: {e}")

                if not dataset_name:
                    dataset_name = selected_dataset_cache.get("dataset_name", "")

                if dataset_name:
                    async with Client(MCP_TRANSPORT) as export_client:
                        try:
                            export_result = await export_client.call_tool(
                                "export_to_cvat",
                                {"dataset_name": dataset_name, "with_predictions": True}
                            )
                            export_msg = unwrap_tool_output(export_result)
                            reply += f"\n\n{export_msg}"
                            reply += f"\n\nPlease review and correct the predictions in CVAT. Let me know when you're done and I'll import the labels back."
                        except Exception as e:
                            reply += f"\n\nNote: CVAT export failed: {str(e)}"
                else:
                    reply += f"\n\nNote: Could not determine dataset name for CVAT export."

                return {"reply": reply}

            elif fn_name == "import_from_cvat":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                tool_output = unwrap_tool_output(tool_output_raw)

                # FIX 2 (edge case 4): strip _labeled suffix before constructing
                # labeled name to avoid double-suffixing if LLM passes wrong name.
                try:
                    base_dataset = fn_args.get("dataset_name", "").removesuffix("_labeled")
                    labeled_name = f"{base_dataset}_labeled" if base_dataset else ""
                    _update_workflow_state(labeled_dataset_name=labeled_name)
                except Exception:
                    pass

                return {"reply": f"{tool_output.strip()}\n\nWould you like to visualize the labeled dataset in Voxel51?"}

            elif fn_name == "run_class_mapping":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                tool_output = unwrap_tool_output(tool_output_raw)
                summary = await llm.summarize_class_mapping_output(tool_output)
                reply = (
                    f"{summary}\n\n"
                    f"Class Mapping Output:\n"
                    f"```\n{tool_output.strip()}\n```"
                )
                return {"reply": reply}

            elif fn_name == "run_anomaly_detection":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                tool_output = unwrap_tool_output(tool_output_raw)
                summary = await llm.summarize_anomaly_detection_output(tool_output)
                reply = (
                    f"{summary}\n\n"
                    f"Anomaly Detection Output:\n"
                    f"```\n{tool_output.strip()}\n```"
                )
                return {"reply": reply}

            elif fn_name == "run_zero_shot_auto_labeling":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                tool_output = unwrap_tool_output(tool_output_raw)
                reply = (
                    f"{tool_output.strip()}\n"
                    f"You can now use the Ensemble Selection workflow to identify detections where multiple models agree.\n"
                    f"Would you like to launch Voxel51 to explore the results?"
                )
                return {"reply": reply}

            elif fn_name == "run_ensemble_selection":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                tool_output = unwrap_tool_output(tool_output_raw)
                reply = (
                    f"{tool_output.strip()}\n\n"
                    f"launch Voxel51 to explore the results?\n\n"
                    f"- In the ENSEMBLE SELECTION section of the left sidebar, use the `n_unique_ensemble_selection` field as a filter. "
                    f"- It represents the number of overlapping objects retained in each sample based on model agreement. "
                    f"- Once you select a sample image, use the `detections_overlap` tag from the TAGS panel to visualize only those detections that had sufficient overlap and were retained by the ensemble logic."
                )
                return {"reply": reply}

            elif fn_name == "set_selected_dataset":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                tool_output = unwrap_tool_output(tool_output_raw)
                if "DATASET_NOT_FOUND" in tool_output:
                    async with Client(MCP_TRANSPORT) as list_client:
                        try:
                            list_result = await list_client.call_tool("list_datasets", {})
                            list_output = unwrap_tool_output(list_result)
                            return {
                                "reply": (
                                    f"That dataset name wasn't recognized. "
                                    f"Here are the available datasets:\n\n{list_output}\n\n"
                                    f"Please select the correct name or re-ingest if needed."
                                )
                            }
                        except Exception as e:
                            return {"reply": f"Dataset not found and couldn't fetch the list: {e}"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": fn_name,
                    "content": tool_output
                })
                continue

            # For other tools, keep old flow
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "name": fn_name,
                "content": tool_output
            })

        # Skip reminder if datasets were just fetched
        tools_called_this_round = [r["name"] for r in tool_results]
        if conversation_state["workflow_name"] and not conversation_state["dataset_selected"] \
                and "list_datasets" not in tools_called_this_round:
            messages.append({
                "role": "system",
                "content": (
                    "Reminder: the user has selected a workflow but has not yet selected a dataset. "
                    "Show the user the dataset list returned by list_datasets above. "
                    "Do NOT list datasets from memory."
                )
            })

        logging.warning(f"[CHAT] Calling final llm.chat, message count={len(messages)}")
        # Final summarization call — plain text only, no tools needed
        final_response_msg = await llm.chat(messages, tools=None, tool_choice=None)
        logging.warning(f"[CHAT] Final response received: '{(getattr(final_response_msg, 'content', '') or '')[:100]}'")
        try:
            reply_content = getattr(final_response_msg, "content", final_response_msg)
            reply = unwrap_tool_output(reply_content)
            if not reply:
                reply = "I've completed the action. What would you like to do next?"
            logging.warning(f"[CHAT] Full reply length={len(reply)}: '{reply}'")
            return {"reply": reply}
        except Exception as e:
            logging.warning(f"[CHAT] Exception after final llm.chat: {e}")
            import traceback
            logging.warning(traceback.format_exc())
            return {"reply": "I've completed the action. What would you like to do next?"}

    else:
        # tool_choice="required" means this branch should never be reached,
        # but handle it gracefully just in case
        reply = assistant_message.content or ""
        logging.warning(f"[CHAT] Unexpected: no tool_calls despite tool_choice=required. Content: '{reply[:100]}'")
        return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)