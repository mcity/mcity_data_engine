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
from llm_clients import ClaudeClient, GeminiClient, GroqClient, OpenAIClient
from tool_schema import tools

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

llm = _LLM_PROVIDERS[llm_provider]()

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


def filter_tools_for_state(all_tools: list, state) -> list:
    """
    Return only the tools valid for the current workflow step.

    Derives the valid set from WorkflowState.valid_tool_names(), which encodes
    step sequencing using the same Pydantic state previously used only for
    post-call validation. This makes wrong-sequence tool calls structurally
    impossible rather than textually discouraged.

    Fails open: if state is None or valid_tool_names() returns None, the full
    tool list is returned unchanged.
    """
    if state is None:
        return all_tools
    valid_names = state.valid_tool_names()
    if valid_names is None:
        return all_tools
    return [t for t in all_tools if t["function"]["name"] in valid_names]


def _build_state_hint() -> str:
    """
    Return a compact SESSION_STATE string injected before every user message.

    Only tells the LLM where it is in the flow so it can phrase responses
    correctly. With tool filtering handling step sequencing structurally,
    this no longer needs next_action clauses or do-not-call lists.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from validate_workflow_state import WorkflowState
        state = WorkflowState.load()
        if not state.workflow_name:
            return ""
        parts = [f"workflow={state.workflow_name}"]
        if state.dataset_confirmed and state.dataset_name:
            parts.append(f"dataset={state.dataset_name}")
        else:
            parts.append("dataset=not confirmed")
        al = state.auto_labeling
        if al:
            if al.labeling_backend:
                parts.append(f"backend={al.labeling_backend}")
            if al.labeling_path:
                parts.append(f"labeling_path={al.labeling_path}")
            if al.model_configured:
                parts.append("model=configured")
            if al.auto_labeling_complete:
                parts.append("auto_labeling=complete")
        return "SESSION_STATE: " + " | ".join(parts)
    except Exception:
        return ""


def _reset_state_on_startup() -> None:
    """
    Reset WORKFLOW_STATE to defaults on every server start.

    config.py persists state across restarts, but each server start is a new
    session. Without this, the agent greets returning users with stale context
    from the previous session. Mid-session resets are handled by switch_workflow.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from validate_workflow_state import WorkflowState
        WorkflowState().save()
        logging.warning("[STARTUP] Reset WORKFLOW_STATE to defaults.")
    except Exception as e:
        logging.warning(f"[STARTUP] Could not reset WORKFLOW_STATE: {e}")

_reset_state_on_startup()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])

    # Trim to the most recent 8 turns. SESSION_STATE carries workflow position,
    # so long history adds tokens without adding useful context. 8 turns covers
    # the full manual-labeling flow with one turn of headroom.
    MAX_HISTORY_TURNS = 8
    if len(history) > MAX_HISTORY_TURNS:
        history = history[-MAX_HISTORY_TURNS:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # Inject workflow position immediately before the user's message. The
    # conversation history only carries final text replies, so without this the
    # LLM may re-call tools that already ran (e.g. select_workflow when the user
    # is replying to a dataset list prompt with a dataset name).
    state_hint = _build_state_hint()
    if state_hint:
        messages.append({"role": "system", "content": state_hint})

    messages.append({"role": "user", "content": message})

    # Filter to tools valid for the current workflow step. The LLM structurally
    # cannot call out-of-sequence tools because they aren't in its tool list.
    try:
        from validate_workflow_state import WorkflowState as _WS
        _state = _WS.load()
    except Exception:
        _state = None
    active_tools = filter_tools_for_state(tools, _state)

    logging.warning(
        f"[CHAT] Active tools ({len(active_tools)}): "
        f"{[t['function']['name'] for t in active_tools]}"
    )

    # tool_choice="required" forces the model to call a real tool or send_reply.
    try:
        assistant_message = await llm.chat(messages, tools=active_tools, tool_choice="required")
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