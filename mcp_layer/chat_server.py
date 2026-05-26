# mcp_layer/chat_server.py

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastmcp.client.transports import SSETransport

sys.path.append(os.path.dirname(__file__))

from chat_pipeline import ChatPipeline, unwrap_tool_output
from host_utils import resolve_host
from llm_clients import GeminiClient, GroqClient, OpenAIClient
from tool_schema import tools

load_dotenv()

llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
llm = {"openai": OpenAIClient, "groq": GroqClient, "gemini": GeminiClient}[llm_provider]()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

host = resolve_host()
MCP_TRANSPORT = SSETransport(url=f"http://{host}:8000/sse")

SYSTEM_PROMPT = (
    Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"
).read_text()


def _clear_persisted_backend() -> None:
    """
    Clear labeling_backend from persisted state on startup.

    The backend is re-detected from .env at dataset confirmation each session.
    Without this, a stale backend written by a previous session would be used
    if credentials change between server restarts.
    """
    try:
        import ast as _ast
        import importlib
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import config.config as _cc
        importlib.reload(_cc)
        raw = dict(_cc.WORKFLOW_STATE)
        al = raw.get("auto_labeling")
        if isinstance(al, dict) and al.get("labeling_backend"):
            al["labeling_backend"] = ""
            config_path = Path(__file__).resolve().parents[1] / "config" / "config.py"
            src = config_path.read_text()
            tree = _ast.parse(src)
            lines = src.splitlines()
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Assign):
                    for t in node.targets:
                        if isinstance(t, _ast.Name) and t.id == "WORKFLOW_STATE":
                            lines[node.lineno - 1:node.end_lineno] = [
                                f"WORKFLOW_STATE = {repr(raw)}"
                            ]
                            config_path.write_text("\n".join(lines) + "\n")
                            logging.warning(
                                "[STARTUP] Cleared persisted labeling_backend "
                                "from WORKFLOW_STATE — will re-detect from .env "
                                "on next session."
                            )
                            return
    except Exception as e:
        logging.warning(f"[STARTUP] Could not clear labeling_backend: {e}")

_clear_persisted_backend()


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # tool_choice="required" forces the model to call a real tool or send_reply.
    try:
        assistant_message = await llm.chat(messages, tools=tools, tool_choice="required")
    except Exception as e:
        err = str(e).lower()
        if "timeout" in err or "connecttimeout" in err or "apitimeout" in err:
            logging.warning(f"[CHAT] LLM request timed out: {e}")
            return {"reply": "The request timed out reaching the AI service. Please try again in a moment."}
        logging.warning(f"[CHAT] LLM request failed: {e}")
        return {"reply": "Something went wrong connecting to the AI service. Please try again."}

    if not (hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls):
        reply = assistant_message.content or ""
        logging.warning(f"[CHAT] No tool_calls despite tool_choice=required: '{reply[:100]}'")
        return {"reply": reply}

    tool_calls = assistant_message.tool_calls

    # Single send_reply — return immediately without going through the pipeline.
    if len(tool_calls) == 1 and tool_calls[0].function.name == "send_reply":
        try:
            args = json.loads(tool_calls[0].function.arguments)
            return {"reply": args.get("message", "")}
        except Exception:
            return {"reply": "Something went wrong. Please try again."}

    # Append assistant turn before tool results to satisfy OpenAI message ordering.
    messages.append({
        "role": "assistant",
        "content": assistant_message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    })

    pipeline = ChatPipeline(mcp_transport=MCP_TRANSPORT, llm=llm)
    tool_results, early_reply = await pipeline.run(tool_calls, messages)

    if early_reply is not None:
        return {"reply": early_reply}

    # If a workflow was selected but no dataset confirmed, remind the model to
    # show the dataset list from tool results rather than recalling from memory.
    tools_called = [r["name"] for r in tool_results]
    if (
        pipeline.state.workflow_name
        and not pipeline.state.dataset_confirmed
        and "list_datasets" not in tools_called
    ):
        messages.append({
            "role": "system",
            "content": (
                "Reminder: the user has selected a workflow but has not yet "
                "confirmed a dataset. Show the user the dataset list returned "
                "by list_datasets above. Do NOT list datasets from memory."
            ),
        })

    logging.warning(f"[CHAT] Final llm.chat, message count={len(messages)}")
    final_msg = await llm.chat(messages, tools=None, tool_choice=None)
    logging.warning(
        f"[CHAT] Final response: '{(getattr(final_msg, 'content', '') or '')[:100]}'"
    )

    try:
        reply = unwrap_tool_output(getattr(final_msg, "content", final_msg))
        if not reply:
            reply = "I've completed the action. What would you like to do next?"
        logging.warning(f"[CHAT] Reply length={len(reply)}")
        return {"reply": reply}
    except Exception as e:
        logging.warning(f"[CHAT] Exception building final reply: {e}")
        return {"reply": "I've completed the action. What would you like to do next?"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)