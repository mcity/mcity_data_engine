"""Append one JSON record per LLM call to logs/agent/prompts.jsonl.

The server logs only derived data ([STREAM STATE], [STREAM HINT], [STREAM
TOOLS], [STREAM DECISION]), so the prompt that produced a decision is lost as
soon as the request ends. That makes a wrong tool call impossible to examine
afterwards: the history the UI sent, the injected banners, and the state hint
must all be guessed. This module keeps the full record instead.

One line per provider call, all iterations of the agentic loop included. Every
function fails silently: a logging problem must never break a chat request.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "agent" / "prompts.jsonl"


def _tool_calls_to_json(assistant_message) -> list:
    """Extract the model's tool calls in the OpenAI shape used across providers."""
    out = []
    for call in getattr(assistant_message, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        out.append({
            "id":        getattr(call, "id", ""),
            "name":      getattr(fn, "name", ""),
            "arguments": getattr(fn, "arguments", ""),
        })
    return out


def log_llm_call(
    *,
    request_id: str,
    iteration: int,
    provider: str,
    model: str,
    tool_choice: str,
    active_tools: list,
    messages: list,
    duration_ms: float,
    user_message: str,
    history_turns: int,
    assistant_message=None,
    error: str = "",
) -> None:
    """Write one record. `messages` is the exact list handed to the provider."""
    record = {
        "ts":            datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "request_id":    request_id,
        "iteration":     iteration,
        "provider":      provider,
        "model":         model,
        "tool_choice":   tool_choice,
        "duration_ms":   round(duration_ms, 1),
        "user_message":  user_message,
        "history_turns": history_turns,
        "active_tools":  [t.get("function", {}).get("name", "") for t in active_tools],
        "messages":      messages,
        "response": {
            "content":    getattr(assistant_message, "content", None),
            "tool_calls": _tool_calls_to_json(assistant_message),
        } if assistant_message is not None else None,
        "error": error,
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str, ensure_ascii=False)
        # One write per record keeps concurrent requests from interleaving lines.
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logging.warning(f"[PROMPTLOG] Could not write {LOG_PATH}: {e}")


def provider_and_model(llm_client) -> tuple[str, str]:
    """Name the client that served a call. GeminiClient uses `model_name`."""
    provider = type(llm_client).__name__.replace("Client", "").lower()
    model = getattr(llm_client, "model", None) or getattr(llm_client, "model_name", "")
    return provider, str(model)
