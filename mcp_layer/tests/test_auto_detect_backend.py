"""
Tests for ChatPipeline._auto_detect_backend.

Regression guard: prior cleanup removed `import os as _os` from the function
body without adding `import os` at module level, causing a silent NameError
caught by the broad except-handler and returning None instead of a backend_info
dict.  These tests confirm os.getenv is reachable and that the returned dict
has the expected shape under each credential combination.
"""
import asyncio
from unittest.mock import MagicMock, patch

from chat_pipeline import ChatPipeline
from validate_workflow_state import (
    AutoLabelingState, LabelingBackend, WorkflowState,
)


def make_pipeline():
    pipeline = ChatPipeline(mcp_transport=MagicMock(), llm=MagicMock())
    pipeline.state = WorkflowState(
        workflow_name="auto_labeling",
        auto_labeling=AutoLabelingState(),
    )
    return pipeline


def detect(cvat_token: str = "", ls_token: str = "") -> dict | None:
    """Run _auto_detect_backend with controlled env vars, return the result."""
    pipeline = make_pipeline()
    env = {"CVAT_ACCESS_TOKEN": cvat_token, "LS_TOKEN": ls_token}
    with patch("chat_pipeline.os.getenv", side_effect=lambda key, default="": env.get(key, default)):
        with patch.object(WorkflowState, "save"):
            return pipeline, asyncio.run(pipeline._auto_detect_backend(mcp_client=MagicMock()))


# ---------------------------------------------------------------------------
# Regression: os.getenv must not raise NameError → result must not be None
# ---------------------------------------------------------------------------

def test_auto_detect_backend_no_nameerror():
    """os.getenv must be reachable — no NameError from missing 'import os'."""
    try:
        _, result = detect(cvat_token="", ls_token="")
    except AttributeError as exc:
        raise AssertionError(
            "chat_pipeline.os is not defined — check that 'import os' exists at module level"
        ) from exc
    assert result is not None, (
        "_auto_detect_backend returned None — "
        "likely NameError from missing 'import os' at module level"
    )


# ---------------------------------------------------------------------------
# No credentials → backend = "none"
# ---------------------------------------------------------------------------

def test_auto_detect_backend_no_credentials():
    _, result = detect(cvat_token="", ls_token="")
    assert result is not None
    assert result["active_backend"] == LabelingBackend.NONE
    assert result["cvat_available"] is False
    assert result["ls_available"] is False


# ---------------------------------------------------------------------------
# Only LS_TOKEN set → backend = "label_studio"
# ---------------------------------------------------------------------------

def test_auto_detect_backend_ls_only():
    _, result = detect(cvat_token="", ls_token="tok123")
    assert result is not None
    assert result["active_backend"] == LabelingBackend.LABEL_STUDIO
    assert result["ls_available"] is True
    assert result["cvat_available"] is False


# ---------------------------------------------------------------------------
# Only CVAT_ACCESS_TOKEN set → backend = "cvat"
# ---------------------------------------------------------------------------

def test_auto_detect_backend_cvat_only():
    _, result = detect(cvat_token="cvattok", ls_token="")
    assert result is not None
    assert result["active_backend"] == LabelingBackend.CVAT
    assert result["cvat_available"] is True
    assert result["ls_available"] is False


# ---------------------------------------------------------------------------
# Both credentials set → backend = "both"
# ---------------------------------------------------------------------------

def test_auto_detect_backend_both():
    _, result = detect(cvat_token="cvattok", ls_token="lstok")
    assert result is not None
    assert result["active_backend"] == LabelingBackend.BOTH
    assert result["cvat_available"] is True
    assert result["ls_available"] is True


# ---------------------------------------------------------------------------
# State is written for single-backend cases, not for NONE
# ---------------------------------------------------------------------------

def test_auto_detect_backend_writes_state_for_ls():
    pipeline = make_pipeline()
    env = {"CVAT_ACCESS_TOKEN": "", "LS_TOKEN": "tok"}
    with patch("chat_pipeline.os.getenv", side_effect=lambda key, default="": env.get(key, default)):
        with patch.object(WorkflowState, "save") as mock_save:
            asyncio.run(pipeline._auto_detect_backend(mcp_client=MagicMock()))
    assert pipeline.state.auto_labeling.labeling_backend == LabelingBackend.LABEL_STUDIO
    mock_save.assert_called_once()


def test_auto_detect_backend_does_not_write_state_for_none():
    pipeline = make_pipeline()
    env = {"CVAT_ACCESS_TOKEN": "", "LS_TOKEN": ""}
    with patch("chat_pipeline.os.getenv", side_effect=lambda key, default="": env.get(key, default)):
        with patch.object(WorkflowState, "save") as mock_save:
            asyncio.run(pipeline._auto_detect_backend(mcp_client=MagicMock()))
    assert pipeline.state.auto_labeling.labeling_backend == ""
    mock_save.assert_not_called()
