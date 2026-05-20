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

import os
from dotenv import load_dotenv

load_dotenv()
host = os.getenv("PUBLIC_IP", "localhost")

INTENT_PHRASES = [
    "i'll fetch", "i'll get", "i'll check", "i'll import", "i'll load",
    "i'll show", "i'll list", "let me fetch", "let me get", "let me check",
    "let me import", "let me load", "let me show", "let me list",
    "one moment", "i will fetch", "i will get", "i will check",
    "i will import", "i will load", "i will show", "fetching",
    "pulling up", "loading up", "grabbing", "retrieving",
]

MAX_INTENT_RETRIES = 2

import json as _json

WORKFLOW_STATE_FILE = Path(__file__).resolve().parent.parent / "output" / "workflow_state.json"

def _read_workflow_state() -> dict:
    try:
        if WORKFLOW_STATE_FILE.exists():
            return _json.loads(WORKFLOW_STATE_FILE.read_text())
    except Exception:
        pass
    return {"dataset_confirmed": False, "dataset_name": "", "labeled_dataset_name": ""}

# FIX: added labeled_dataset_name parameter
def _write_workflow_state(
    dataset_confirmed: bool,
    dataset_name: str = "",
    labeled_dataset_name: str = ""
):
    try:
        WORKFLOW_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        WORKFLOW_STATE_FILE.write_text(_json.dumps({
            "dataset_confirmed": dataset_confirmed,
            "dataset_name": dataset_name,
            "labeled_dataset_name": labeled_dataset_name
        }))
    except Exception:
        pass

def _has_intent_without_action(message) -> bool:
    """Returns True if the LLM announced an action but did not call any tool."""
    if hasattr(message, "tool_calls") and message.tool_calls:
        return False
    content = (getattr(message, "content", "") or "").lower()
    return any(phrase in content for phrase in INTENT_PHRASES)

async def _retry_if_intent(messages: list, assistant_message, llm, tools) -> tuple:
    for attempt in range(MAX_INTENT_RETRIES):
        if not _has_intent_without_action(assistant_message):
            break
        messages.append({
            "role": "assistant",
            "content": getattr(assistant_message, "content", "") or ""
        })
        messages.append({
            "role": "system",
            "content": (
                "You said you would perform an action but did not call any tool. "
                "You MUST call the appropriate tool RIGHT NOW. "
                "Do NOT answer from memory. Do NOT list anything as text. "
                "ONLY call the tool. No explanation."
            )
        })
        assistant_message = await llm.chat(messages, tools=tools, tool_choice="required")
        if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
            break
    return assistant_message, messages


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

    assistant_message = await llm.chat(messages, tools=tools)
    assistant_message, messages = await _retry_if_intent(messages, assistant_message, llm, tools)

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

        async with Client(MCP_TRANSPORT) as mcp_client:
            for call in tool_calls:
                fn_name = call.function.name
                try:
                    fn_args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                try:
                    if fn_name == "select_workflow":
                        conversation_state["workflow_name"] = fn_args["workflow_name"]
                        conversation_state["dataset_selected"] = False
                        conversation_state["auto_labeling_complete"] = False
                        _write_workflow_state(False)
                        result = await mcp_client.call_tool(fn_name, fn_args)

                    elif fn_name == "switch_workflow":
                        conversation_state["workflow_name"] = fn_args["workflow_name"]
                        conversation_state["dataset_selected"] = False
                        conversation_state["auto_labeling_complete"] = False
                        _write_workflow_state(False)
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
                        # FIX: preserve labeled_dataset_name from previous state
                        existing_state = _read_workflow_state()
                        _write_workflow_state(
                            True,
                            fn_args["dataset_name"],
                            existing_state.get("labeled_dataset_name", "")
                        )
                        result = await mcp_client.call_tool(fn_name, {
                            "dataset_name": selected_dataset_cache["dataset_name"]
                        })
                        messages.append({
                            "role": "system",
                            "content": f"CURRENT_DATASET: {fn_args['dataset_name']}"
                        })

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
                                "result": result
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
                            "result": result
                        })
                        continue

                    else:
                        result = await mcp_client.call_tool(fn_name, fn_args)

                    tool_results.append({
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "result": result
                    })
                except Exception as e:
                    tool_results.append({
                        "tool_call_id": call.id,
                        "name": fn_name,
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
            tool_output = str(result.get("result", result.get("error", "Tool error.")))

            if fn_name == "run_auto_labeling":
                conversation_state["auto_labeling_complete"] = True
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                tool_output = unwrap_tool_output(tool_output_raw)

                # If the guard blocked execution, just return the message — don't export
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

                # Read dataset name from config.py
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

                # FIX: persist the labeled dataset name so launch_voxel51_session loads the right one
                try:
                    base_dataset = fn_args.get("dataset_name", "")
                    labeled_name = f"{base_dataset}_labeled" if base_dataset else ""
                    state = _read_workflow_state()
                    _write_workflow_state(
                        state.get("dataset_confirmed", False),
                        state.get("dataset_name", ""),
                        labeled_name
                    )
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

        # Continue with normal summarization for other tools
        final_response_msg = await llm.chat(messages, tools=None, tool_choice=None)
        reply_content = getattr(final_response_msg, "content", final_response_msg)
        reply = unwrap_tool_output(reply_content)
        if not reply:
            reply = "I've completed the action. What would you like to do next?"

    else:
        reply = assistant_message.content
    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)