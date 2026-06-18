"""
Priority 4: ChatPipeline._orchestrate() two-pass routing logic.

_orchestrate is synchronous. Tests verify:
  - Single Injection + single HardStop: Injection appended to messages, HardStop becomes early_reply
  - Multiple Injections, no HardStop: all appended in order, no early_reply returned
  - Multiple HardStops: the LAST one wins (documented behavior)
  - FallThrough items: no effect on messages or early_reply

state.save() is mocked to prevent disk writes; the MCP transport and LLM
are mocked since _orchestrate never calls them.
"""
import pytest
from unittest.mock import MagicMock, patch

from chat_pipeline import ChatPipeline, Injection, HardStop, FallThrough
from validate_workflow_state import WorkflowState


def make_pipeline(dataset_confirmed=False):
    """Create a ChatPipeline with mocked dependencies and a fresh state."""
    pipeline = ChatPipeline(mcp_transport=MagicMock(), llm=MagicMock())
    pipeline.state = WorkflowState()
    if dataset_confirmed:
        pipeline.state.dataset_name = "test_ds"
        pipeline.state.dataset_confirmed = True
    return pipeline


# ---------------------------------------------------------------------------
# Single Injection + single HardStop
# ---------------------------------------------------------------------------

def test_orchestrate_injection_appended_and_hardstop_returned():
    pipeline = make_pipeline()
    messages = []
    all_routings = [[Injection("ctx msg"), HardStop("stop reply")]]
    result = pipeline._orchestrate(all_routings, messages)

    assert result == "stop reply"
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert any(m["content"] == "ctx msg" for m in system_msgs)


def test_orchestrate_hardstop_alone_returns_reply_with_no_injection():
    pipeline = make_pipeline()
    messages = []
    all_routings = [[HardStop("only stop")]]
    result = pipeline._orchestrate(all_routings, messages)

    assert result == "only stop"
    # No Injection was in the list → no extra system messages (beyond CURRENT_DATASET if applicable)
    system_msgs = [m for m in messages if m["role"] == "system" and m["content"] != "CURRENT_DATASET: test_ds"]
    assert len(system_msgs) == 0


# ---------------------------------------------------------------------------
# Multiple Injections, no HardStop
# ---------------------------------------------------------------------------

def test_orchestrate_multiple_injections_all_appended_in_order():
    pipeline = make_pipeline()
    messages = []
    all_routings = [
        [Injection("first"), Injection("second")],
        [Injection("third")],
    ]
    result = pipeline._orchestrate(all_routings, messages)

    assert result is None
    system_contents = [m["content"] for m in messages if m["role"] == "system"]
    assert "first" in system_contents
    assert "second" in system_contents
    assert "third" in system_contents
    assert system_contents.index("first") < system_contents.index("second")
    assert system_contents.index("second") < system_contents.index("third")


def test_orchestrate_no_routings_returns_none():
    pipeline = make_pipeline()
    messages = []
    result = pipeline._orchestrate([], messages)
    assert result is None


# ---------------------------------------------------------------------------
# Multiple HardStops — last one wins
# ---------------------------------------------------------------------------

def test_orchestrate_last_hardstop_wins_within_single_list():
    pipeline = make_pipeline()
    messages = []
    all_routings = [[HardStop("first stop"), HardStop("second stop")]]
    result = pipeline._orchestrate(all_routings, messages)
    assert result == "second stop"


def test_orchestrate_last_hardstop_wins_across_multiple_routing_lists():
    pipeline = make_pipeline()
    messages = []
    all_routings = [
        [HardStop("earlier stop")],
        [HardStop("later stop")],
    ]
    result = pipeline._orchestrate(all_routings, messages)
    assert result == "later stop"


def test_orchestrate_last_hardstop_wins_interleaved_with_injections():
    pipeline = make_pipeline()
    messages = []
    all_routings = [
        [HardStop("first"), Injection("ctx1")],
        [Injection("ctx2"), HardStop("last")],
    ]
    result = pipeline._orchestrate(all_routings, messages)
    assert result == "last"


# ---------------------------------------------------------------------------
# FallThrough items — no effect
# ---------------------------------------------------------------------------

def test_orchestrate_fallthrough_items_do_not_affect_messages():
    pipeline = make_pipeline()
    messages = []
    all_routings = [[FallThrough()]]
    result = pipeline._orchestrate(all_routings, messages)
    assert result is None
    assert messages == []


def test_orchestrate_fallthrough_mixed_with_injection_and_hardstop():
    pipeline = make_pipeline()
    messages = []
    all_routings = [[FallThrough(), Injection("ctx"), FallThrough(), HardStop("stop")]]
    result = pipeline._orchestrate(all_routings, messages)
    assert result == "stop"
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert any(m["content"] == "ctx" for m in system_msgs)


# ---------------------------------------------------------------------------
# CURRENT_DATASET injection when dataset is confirmed
# ---------------------------------------------------------------------------

def test_orchestrate_injects_current_dataset_message_when_confirmed():
    pipeline = make_pipeline(dataset_confirmed=True)
    messages = []
    pipeline._orchestrate([], messages)
    assert any(
        m["role"] == "system" and "test_ds" in m["content"]
        for m in messages
    )


def test_orchestrate_does_not_inject_current_dataset_when_not_confirmed():
    pipeline = make_pipeline(dataset_confirmed=False)
    messages = []
    pipeline._orchestrate([], messages)
    assert not any(
        m["role"] == "system" and "CURRENT_DATASET" in m.get("content", "")
        for m in messages
    )
