"""
Priority 5: unwrap_tool_output() — all input shapes handled by the function.

Tests cover every distinct branch in the implementation:
  None, str, object-with-text, list, dict-with-text, dict-with-content-list,
  dict-with-data.msg, dict-with-message, dict-with-detail, and the str() fallback.
"""
import pytest
from chat_pipeline import unwrap_tool_output


# ---------------------------------------------------------------------------
# None input
# ---------------------------------------------------------------------------

def test_unwrap_none_returns_empty_string():
    assert unwrap_tool_output(None) == ""


# ---------------------------------------------------------------------------
# String input (returned as-is; no stripping of the raw string)
# ---------------------------------------------------------------------------

def test_unwrap_plain_string_returned_as_is():
    assert unwrap_tool_output("hello world") == "hello world"


def test_unwrap_empty_string_returned_as_is():
    assert unwrap_tool_output("") == ""


def test_unwrap_string_with_escaped_newlines_not_modified():
    # Raw strings are returned verbatim — no \\n→\n conversion for the str branch
    assert unwrap_tool_output("line1\\nline2") == "line1\\nline2"


# ---------------------------------------------------------------------------
# Object with .text attribute (e.g. MCP ContentBlock)
# ---------------------------------------------------------------------------

class _TextObj:
    def __init__(self, text):
        self.text = text


def test_unwrap_object_with_text_returns_stripped_text():
    obj = _TextObj("  hello  ")
    assert unwrap_tool_output(obj) == "hello"


def test_unwrap_object_with_text_converts_escaped_newlines():
    obj = _TextObj("line1\\nline2")
    assert unwrap_tool_output(obj) == "line1\nline2"


def test_unwrap_object_with_empty_text_returns_empty_string():
    obj = _TextObj("")
    assert unwrap_tool_output(obj) == ""


def test_unwrap_object_with_none_text_returns_empty_string():
    obj = _TextObj(None)
    assert unwrap_tool_output(obj) == ""


# ---------------------------------------------------------------------------
# List input — parts joined with \n, empty parts filtered
# ---------------------------------------------------------------------------

def test_unwrap_list_of_strings_joined():
    result = unwrap_tool_output(["part1", "part2"])
    assert result == "part1\npart2"


def test_unwrap_list_with_none_elements_filtered():
    result = unwrap_tool_output([None, "part1", None])
    assert result == "part1"


def test_unwrap_list_with_text_objects():
    result = unwrap_tool_output([_TextObj("a"), _TextObj("b")])
    assert result == "a\nb"


def test_unwrap_empty_list_returns_empty_string():
    result = unwrap_tool_output([])
    assert result == ""


def test_unwrap_list_all_empty_parts_returns_empty_string():
    result = unwrap_tool_output([None, "", None])
    assert result == ""


# ---------------------------------------------------------------------------
# Dict with "text" key
# ---------------------------------------------------------------------------

def test_unwrap_dict_with_text_key_returns_stripped_value():
    assert unwrap_tool_output({"text": "  hello  "}) == "hello"


def test_unwrap_dict_with_text_key_converts_escaped_newlines():
    assert unwrap_tool_output({"text": "a\\nb"}) == "a\nb"


# ---------------------------------------------------------------------------
# Dict with "content" list key (recursive)
# ---------------------------------------------------------------------------

def test_unwrap_dict_with_content_list_delegates_to_list_handler():
    d = {"content": ["part1", "part2"]}
    result = unwrap_tool_output(d)
    assert result == "part1\npart2"


def test_unwrap_dict_with_content_text_objects():
    d = {"content": [_TextObj("x"), _TextObj("y")]}
    result = unwrap_tool_output(d)
    assert result == "x\ny"


# ---------------------------------------------------------------------------
# Dict with "data" → {"msg": ...}
# ---------------------------------------------------------------------------

def test_unwrap_dict_data_msg_returns_string_of_msg():
    d = {"data": {"msg": "operation succeeded"}}
    result = unwrap_tool_output(d)
    assert result == "operation succeeded"


def test_unwrap_dict_data_msg_converts_escaped_newlines():
    d = {"data": {"msg": "line1\\nline2"}}
    result = unwrap_tool_output(d)
    assert result == "line1\nline2"


# ---------------------------------------------------------------------------
# Dict with "message" key
# ---------------------------------------------------------------------------

def test_unwrap_dict_message_key_returned():
    d = {"message": "error occurred"}
    assert unwrap_tool_output(d) == "error occurred"


def test_unwrap_dict_message_key_converts_escaped_newlines():
    d = {"message": "a\\nb"}
    assert unwrap_tool_output(d) == "a\nb"


# ---------------------------------------------------------------------------
# Dict with "detail" key (checked after "message")
# ---------------------------------------------------------------------------

def test_unwrap_dict_detail_key_returned():
    d = {"detail": "not found"}
    assert unwrap_tool_output(d) == "not found"


def test_unwrap_dict_detail_key_converts_escaped_newlines():
    d = {"detail": "a\\nb"}
    assert unwrap_tool_output(d) == "a\nb"


# ---------------------------------------------------------------------------
# Dict — "message" takes priority over "detail"
# ---------------------------------------------------------------------------

def test_unwrap_dict_message_takes_priority_over_detail():
    d = {"message": "from message", "detail": "from detail"}
    result = unwrap_tool_output(d)
    assert result == "from message"


# ---------------------------------------------------------------------------
# Fallback: str(raw).strip()
# ---------------------------------------------------------------------------

def test_unwrap_arbitrary_object_falls_back_to_str():
    class _Weird:
        def __str__(self):
            return "  weird  "
    result = unwrap_tool_output(_Weird())
    assert result == "weird"


def test_unwrap_integer_falls_back_to_str():
    assert unwrap_tool_output(42) == "42"


def test_unwrap_dict_with_no_recognized_keys_falls_back_to_str():
    d = {"unknown_key": "some value"}
    result = unwrap_tool_output(d)
    # No recognized key → str(d).strip()
    assert result == str(d).strip()
