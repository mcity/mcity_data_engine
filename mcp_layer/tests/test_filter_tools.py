"""
Priority 7: filter_tools_for_state() from chat_server.py.

Fails-open behavior:
  - state is None                → all tools returned
  - valid_tool_names() raises    → all tools returned (fail-open)
  - valid_tool_names() returns None → all tools returned
  - valid_tool_names() returns a set → filtered list returned

The current fail-open behavior (returning all tools on exception) is documented
here as the CURRENT behavior, not necessarily the desired one. See the report for
the security note about this choice.
"""
from unittest.mock import MagicMock

from chat_server import filter_tools_for_state
from validate_workflow_state import WorkflowState, AutoLabelingState


def _tool(name: str) -> dict:
    return {"function": {"name": name}}


ALL_TOOLS = [_tool("send_reply"), _tool("select_workflow"), _tool("set_selected_dataset")]


# ---------------------------------------------------------------------------
# state=None → fails open (all tools)
# ---------------------------------------------------------------------------

def test_filter_tools_none_state_returns_all_tools():
    result = filter_tools_for_state(ALL_TOOLS, None)
    assert result == ALL_TOOLS


# ---------------------------------------------------------------------------
# valid_tool_names() returns None → fails open (all tools)
# ---------------------------------------------------------------------------

def test_filter_tools_valid_tool_names_returns_none_returns_all_tools():
    state = MagicMock()
    state.valid_tool_names.return_value = None
    result = filter_tools_for_state(ALL_TOOLS, state)
    assert result == ALL_TOOLS


# ---------------------------------------------------------------------------
# valid_tool_names() raises → fails open (all tools)
# ---------------------------------------------------------------------------

def test_filter_tools_valid_tool_names_raises_returns_all_tools():
    state = MagicMock()
    state.valid_tool_names.side_effect = RuntimeError("state corruption")
    result = filter_tools_for_state(ALL_TOOLS, state)
    assert result == ALL_TOOLS


def test_filter_tools_valid_tool_names_raises_attribute_error_returns_all_tools():
    state = MagicMock()
    state.valid_tool_names.side_effect = AttributeError("missing attr")
    result = filter_tools_for_state(ALL_TOOLS, state)
    assert result == ALL_TOOLS


# ---------------------------------------------------------------------------
# valid_tool_names() returns a set → filtered list
# ---------------------------------------------------------------------------

def test_filter_tools_filters_to_valid_set():
    state = MagicMock()
    state.valid_tool_names.return_value = {"send_reply", "select_workflow"}
    result = filter_tools_for_state(ALL_TOOLS, state)
    names = [t["function"]["name"] for t in result]
    assert "send_reply" in names
    assert "select_workflow" in names
    assert "set_selected_dataset" not in names


def test_filter_tools_empty_valid_set_returns_empty_list():
    state = MagicMock()
    state.valid_tool_names.return_value = set()
    result = filter_tools_for_state(ALL_TOOLS, state)
    assert result == []


def test_filter_tools_preserves_order_of_all_tools():
    """Filtered list should maintain the relative order from all_tools."""
    tools = [_tool("a"), _tool("b"), _tool("c"), _tool("d")]
    state = MagicMock()
    state.valid_tool_names.return_value = {"a", "c"}
    result = filter_tools_for_state(tools, state)
    names = [t["function"]["name"] for t in result]
    assert names == ["a", "c"]


# ---------------------------------------------------------------------------
# Real WorkflowState integration
# ---------------------------------------------------------------------------

def test_filter_tools_with_real_state_no_workflow():
    """WorkflowState with no workflow set → only select_workflow + always tools visible."""
    state = WorkflowState()
    tools = [
        _tool("send_reply"), _tool("switch_workflow"), _tool("select_workflow"),
        _tool("set_selected_dataset"), _tool("run_auto_labeling"),
    ]
    result = filter_tools_for_state(tools, state)
    names = {t["function"]["name"] for t in result}
    assert "send_reply" in names
    assert "switch_workflow" in names
    assert "select_workflow" in names
    assert "set_selected_dataset" not in names
    assert "run_auto_labeling" not in names


def test_filter_tools_with_real_state_dataset_not_confirmed():
    state = WorkflowState(workflow_name="auto_labeling")
    tools = [
        _tool("send_reply"), _tool("switch_workflow"),
        _tool("set_selected_dataset"), _tool("list_datasets"),
        _tool("run_auto_labeling"),
    ]
    result = filter_tools_for_state(tools, state)
    names = {t["function"]["name"] for t in result}
    assert "set_selected_dataset" in names
    assert "list_datasets" in names
    assert "run_auto_labeling" not in names
