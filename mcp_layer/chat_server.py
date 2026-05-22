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

# LLM client

llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
llm = {"openai": OpenAIClient, "groq": GroqClient, "gemini": GeminiClient}[llm_provider]()

# FastAPI app

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP transport

host = resolve_host()
MCP_TRANSPORT = SSETransport(url=f"http://{host}:8000/sse")

# System prompt

SYSTEM_PROMPT = (
    Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"
).read_text()

# Chat endpoint

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])

    # Build message history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # Always require a tool call — model must call a real tool or send_reply
    assistant_message = await llm.chat(messages, tools=tools, tool_choice="required")

    if not (hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls):
        reply = assistant_message.content or ""
        logging.warning(f"[CHAT] No tool_calls despite tool_choice=required: '{reply[:100]}'")
        return {"reply": reply}

    tool_calls = assistant_message.tool_calls

    # send_reply only — return immediately, no MCP needed
    if len(tool_calls) == 1 and tool_calls[0].function.name == "send_reply":
        try:
            args = json.loads(tool_calls[0].function.arguments)
            return {"reply": args.get("message", "")}
        except Exception:
            return {"reply": "Something went wrong. Please try again."}

    # Append assistant turn before tool results
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

    # Run the pipeline
    pipeline = ChatPipeline(mcp_transport=MCP_TRANSPORT, llm=llm)
    tool_results, early_reply = await pipeline.run(tool_calls, messages)

    # Early return — pipeline produced a ready reply
    if early_reply is not None:
        return {"reply": early_reply}

    # Remind model to show dataset list if workflow selected but no dataset yet
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

    # Final LLM call to summarize tool results into a user-facing reply
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