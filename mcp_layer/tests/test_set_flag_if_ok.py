"""
Priority 8: ChatPipeline._set_flag_if_ok() helper.

Tests verify:
  - Error sentinel present in result → flag NOT set, HardStop returned with message
  - Error sentinel absent → flag IS set via setter, state.save() called, returns None
  - Empty sentinel list (pass-through) → flag always set
  - Error message extraction: if result contains ":", err uses the text after ":"
  - Error message extraction: if result has no ":", err is the full result string

WorkflowState.save is patched at class level (Pydantic extra="forbid" prevents
assigning a MagicMock directly to an instance attribute).
"""
import pytest
from unittest.mock import MagicMock, patch

from chat_pipeline import ChatPipeline, HardStop
from validate_workflow_state import WorkflowState, AutoLabelingState


def make_pipeline():
    pipeline = ChatPipeline(mcp_transport=MagicMock(), llm=MagicMock())
    pipeline.state = WorkflowState(
        workflow_name="auto_labeling",
        auto_labeling=AutoLabelingState(),
    )
    return pipeline


# ---------------------------------------------------------------------------
# Error sentinel present → HardStop returned, setter NOT called
# ---------------------------------------------------------------------------

def test_set_flag_if_ok_returns_hardstop_when_sentinel_present():
    pipeline = make_pipeline()
    flag_set = []
    with patch.object(WorkflowState, "save"):
        result = pipeline._set_flag_if_ok(
            result="DATASET_NOT_FOUND: dataset 'foo' not found",
            error_sentinels=["DATASET_NOT_FOUND"],
            state_setter=lambda: flag_set.append(True),
        )
    assert isinstance(result, HardStop)
    assert flag_set == []


def test_set_flag_if_ok_does_not_call_save_on_error():
    pipeline = make_pipeline()
    with patch.object(WorkflowState, "save") as mock_save:
        pipeline._set_flag_if_ok(
            result="DATASET_NOT_FOUND: not found",
            error_sentinels=["DATASET_NOT_FOUND"],
            state_setter=lambda: None,
        )
    mock_save.assert_not_called()


def test_set_flag_if_ok_sentinel_not_in_result_passes():
    pipeline = make_pipeline()
    flag_set = []
    with patch.object(WorkflowState, "save"):
        result = pipeline._set_flag_if_ok(
            result="Success: model configured",
            error_sentinels=["DATASET_NOT_FOUND"],
            state_setter=lambda: flag_set.append(True),
        )
    assert result is None
    assert flag_set == [True]


# ---------------------------------------------------------------------------
# Error sentinel absent → setter called, save() called, returns None
# ---------------------------------------------------------------------------

def test_set_flag_if_ok_calls_setter_when_no_sentinel():
    pipeline = make_pipeline()
    flag_set = []
    with patch.object(WorkflowState, "save") as mock_save:
        result = pipeline._set_flag_if_ok(
            result="Model configured successfully",
            error_sentinels=["FAILED"],
            state_setter=lambda: flag_set.append(True),
        )
    assert result is None
    assert flag_set == [True]
    mock_save.assert_called_once()


def test_set_flag_if_ok_returns_none_when_no_error():
    pipeline = make_pipeline()
    with patch.object(WorkflowState, "save"):
        result = pipeline._set_flag_if_ok(
            result="ok",
            error_sentinels=["ERROR"],
            state_setter=lambda: None,
        )
    assert result is None


# ---------------------------------------------------------------------------
# Empty sentinel list → always succeeds (no sentinel to match)
# ---------------------------------------------------------------------------

def test_set_flag_if_ok_empty_sentinel_list_always_passes():
    pipeline = make_pipeline()
    flag_set = []
    with patch.object(WorkflowState, "save") as mock_save:
        result = pipeline._set_flag_if_ok(
            result="DATASET_NOT_FOUND: something failed",
            error_sentinels=[],
            state_setter=lambda: flag_set.append(True),
        )
    assert result is None
    assert flag_set == [True]
    mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# Error message extraction from result string
# ---------------------------------------------------------------------------

def test_set_flag_if_ok_extracts_message_after_colon():
    """When result contains ':', the HardStop message is the text after the first ':'."""
    pipeline = make_pipeline()
    with patch.object(WorkflowState, "save"):
        stop = pipeline._set_flag_if_ok(
            result="Failed to locate: candidate labels not found in dataset",
            error_sentinels=["Failed to locate"],
            state_setter=lambda: None,
        )
    assert isinstance(stop, HardStop)
    assert stop.reply == "candidate labels not found in dataset"


def test_set_flag_if_ok_uses_full_result_when_no_colon():
    """When result has no ':', the full result string becomes the HardStop message."""
    pipeline = make_pipeline()
    with patch.object(WorkflowState, "save"):
        stop = pipeline._set_flag_if_ok(
            result="SENTINEL_ONLY",
            error_sentinels=["SENTINEL_ONLY"],
            state_setter=lambda: None,
        )
    assert isinstance(stop, HardStop)
    assert stop.reply == "SENTINEL_ONLY"


# ---------------------------------------------------------------------------
# Multiple sentinels — any match triggers the stop
# ---------------------------------------------------------------------------

def test_set_flag_if_ok_first_sentinel_match_triggers_stop():
    pipeline = make_pipeline()
    flag_set = []
    with patch.object(WorkflowState, "save"):
        result = pipeline._set_flag_if_ok(
            result="not found in dataset: some detail",
            error_sentinels=["other error", "not found in"],
            state_setter=lambda: flag_set.append(True),
        )
    assert isinstance(result, HardStop)
    assert flag_set == []


def test_set_flag_if_ok_no_sentinel_matches_still_passes():
    pipeline = make_pipeline()
    flag_set = []
    with patch.object(WorkflowState, "save"):
        result = pipeline._set_flag_if_ok(
            result="Success",
            error_sentinels=["FAILED", "ERROR", "NOT_FOUND"],
            state_setter=lambda: flag_set.append(True),
        )
    assert result is None
    assert flag_set == [True]


# ---------------------------------------------------------------------------
# Real-world usage: state flag mutation via setter
# ---------------------------------------------------------------------------

def test_set_flag_if_ok_actually_mutates_state_via_setter():
    """Verify the setter actually mutates state (not just called)."""
    pipeline = make_pipeline()
    assert pipeline.state.auto_labeling.models_listed is False

    with patch.object(WorkflowState, "save"):
        pipeline._set_flag_if_ok(
            result="Listed 5 models",
            error_sentinels=[],
            state_setter=lambda: setattr(pipeline.state.auto_labeling, "models_listed", True),
        )
    assert pipeline.state.auto_labeling.models_listed is True
