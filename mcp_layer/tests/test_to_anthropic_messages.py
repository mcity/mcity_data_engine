"""
Priority 6: ClaudeClient._to_anthropic_messages() format conversion.

_to_anthropic_messages is a @staticmethod — callable without instantiating ClaudeClient
(the __init__ import of AsyncAnthropic is deferred and never triggered here).

Tests cover:
  - System message accumulation into system_parts
  - Tool result attachment to preceding user message
  - Tool result with no preceding user message (new user block created)
  - Empty tool_calls list on assistant message → placeholder " " block
  - Consecutive user messages merged
  - Leading non-user message trimmed from result list
  - Empty/whitespace system content filtered
  - Empty user content filtered out at end
"""
import pytest
from llm_clients import ClaudeClient

tam = ClaudeClient._to_anthropic_messages


# ---------------------------------------------------------------------------
# System messages
# ---------------------------------------------------------------------------

def test_system_messages_accumulated_into_system_parts():
    msgs = [
        {"role": "system", "content": "part1"},
        {"role": "system", "content": "part2"},
        {"role": "user", "content": "hello"},
    ]
    system_parts, result = tam(msgs)
    assert system_parts == ["part1", "part2"]
    assert result == [{"role": "user", "content": "hello"}]


def test_system_message_whitespace_only_is_filtered():
    msgs = [
        {"role": "system", "content": "   "},
        {"role": "user", "content": "hi"},
    ]
    system_parts, result = tam(msgs)
    assert system_parts == []


def test_system_message_empty_string_is_filtered():
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "hi"},
    ]
    system_parts, result = tam(msgs)
    assert system_parts == []


def test_system_message_none_content_is_filtered():
    msgs = [
        {"role": "system", "content": None},
        {"role": "user", "content": "hi"},
    ]
    system_parts, result = tam(msgs)
    assert system_parts == []


# ---------------------------------------------------------------------------
# User messages
# ---------------------------------------------------------------------------

def test_single_user_message():
    msgs = [{"role": "user", "content": "hello"}]
    system_parts, result = tam(msgs)
    assert system_parts == []
    assert result == [{"role": "user", "content": "hello"}]


def test_consecutive_user_messages_merged_as_string():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    _, result = tam(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "first" in result[0]["content"]
    assert "second" in result[0]["content"]


def test_empty_user_content_filtered_from_result():
    """User message with empty content: appended then filtered by the end-of-function filter."""
    msgs = [{"role": "user", "content": ""}]
    _, result = tam(msgs)
    # Empty string is in the filter list → should be removed
    assert result == []


# ---------------------------------------------------------------------------
# Tool result messages
# ---------------------------------------------------------------------------

def test_tool_result_appended_to_preceding_user_message():
    msgs = [
        {"role": "user", "content": "question"},
        {"role": "tool", "tool_call_id": "tc1", "content": "tool output"},
    ]
    _, result = tam(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    content = result[0]["content"]
    assert isinstance(content, list)
    tool_block = next((b for b in content if b.get("type") == "tool_result"), None)
    assert tool_block is not None
    assert tool_block["tool_use_id"] == "tc1"
    assert tool_block["content"] == "tool output"


def test_tool_result_with_no_preceding_user_message_creates_new_user_block():
    msgs = [
        {"role": "tool", "tool_call_id": "tc1", "content": "result"},
    ]
    _, result = tam(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert result[0]["content"][0]["type"] == "tool_result"


# ---------------------------------------------------------------------------
# Assistant messages
# ---------------------------------------------------------------------------

def test_assistant_with_text_content():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello back", "tool_calls": []},
    ]
    _, result = tam(msgs)
    asst = next(m for m in result if m["role"] == "assistant")
    text_block = next(b for b in asst["content"] if b["type"] == "text")
    assert text_block["text"] == "hello back"


def test_assistant_with_empty_tool_calls_list_gets_placeholder_block():
    """
    Empty tool_calls + empty content → blocks is empty → fallback placeholder ' '.
    This placeholder is intentional and must not be filtered by the content filter.
    """
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": []},
    ]
    _, result = tam(msgs)
    asst = next(m for m in result if m["role"] == "assistant")
    assert isinstance(asst["content"], list)
    assert any(b.get("type") == "text" and b.get("text") == " " for b in asst["content"])


def test_assistant_with_dict_tool_call():
    msgs = [
        {"role": "user", "content": "use tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc1", "function": {"name": "do_thing", "arguments": '{"x": 1}'}}
            ],
        },
    ]
    _, result = tam(msgs)
    asst = next(m for m in result if m["role"] == "assistant")
    tool_block = next(b for b in asst["content"] if b["type"] == "tool_use")
    assert tool_block["id"] == "tc1"
    assert tool_block["name"] == "do_thing"
    assert tool_block["input"] == {"x": 1}


def test_assistant_with_object_style_tool_call():
    """tool_calls can be objects with .id and .function attributes (OpenAI SDK style)."""
    class _Fn:
        name = "send_reply"
        arguments = '{"message": "hi"}'

    class _TC:
        id = "tc2"
        function = _Fn()

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [_TC()]},
    ]
    _, result = tam(msgs)
    asst = next(m for m in result if m["role"] == "assistant")
    tool_block = next(b for b in asst["content"] if b["type"] == "tool_use")
    assert tool_block["id"] == "tc2"
    assert tool_block["name"] == "send_reply"
    assert tool_block["input"] == {"message": "hi"}


# ---------------------------------------------------------------------------
# Leading role trim
# ---------------------------------------------------------------------------

def test_leading_assistant_message_is_trimmed():
    msgs = [
        {"role": "assistant", "content": "some preamble", "tool_calls": []},
        {"role": "user", "content": "actual start"},
    ]
    _, result = tam(msgs)
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "actual start"


def test_leading_system_messages_do_not_appear_in_result_list():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "start"},
    ]
    system_parts, result = tam(msgs)
    assert result[0]["role"] == "user"
    assert len([m for m in result if m["role"] == "system"]) == 0


# ---------------------------------------------------------------------------
# Empty messages list
# ---------------------------------------------------------------------------

def test_empty_messages_list_returns_empty_system_parts_and_empty_result():
    system_parts, result = tam([])
    assert system_parts == []
    assert result == []
