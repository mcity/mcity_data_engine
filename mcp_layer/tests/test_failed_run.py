"""
Failed-run recovery: a run_* tool that reports a non-zero exit code must leave
the workflow editable and retryable.

Tests verify:
  - run_failed() detects the RUN_FAILED sentinel and the exit-code fallback
  - parse_run_failure() pulls the error line and log path from both result shapes
  - _finalize_auto_labeling() on failure keeps phase/auto_labeling_complete as
    they were, clears the run consent, and never runs the post-run export
  - valid_tool_names() still exposes every configuration tool after a failure
  - WorkflowState.record_run() / failed_run() bookkeeping
  - a non-auto-labeling run handler reports the failure without the LLM summarizer

WorkflowState.save is patched throughout: these tests must not write config.py.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from chat_pipeline import ChatPipeline, HardStop, run_failed, parse_run_failure
from validate_workflow_state import (
    AnomalyDetectionState, AutoLabelingPhase, AutoLabelingState,
    LabelingBackend, LabelingPath, LastRun, WorkflowState,
)

# Real shape returned by the five non-streaming run tools.
FAILED_RESULT = (
    "RUN_FAILED: Anomaly detection failed with exit code 1.\n"
    "Last error line: RuntimeError: CUDA out of memory\n"
    "Full logs saved to `output/logs/last_anomaly_detection_log.txt`"
)

# Real shape returned by the streaming auto-labeling path in chat_pipeline.
FAILED_STREAMING_RESULT = (
    "RUN_FAILED: Auto-labeling failed with exit code 1.\n"
    "Error details:\n```\n"
    "Traceback (most recent call last):\n"
    "ValueError: no samples in dataset\n"
    "```\n"
    "Full logs saved to `output/logs/last_auto_labeling_log.txt`"
)


def make_pipeline(state: WorkflowState) -> ChatPipeline:
    pipeline = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    pipeline.state = state
    return pipeline


def _al_state(**kwargs) -> WorkflowState:
    """auto_labeling state on the auto path, model configured, ready to run."""
    defaults = dict(
        labeling_path=LabelingPath.AUTO,
        labeling_backend=LabelingBackend.CVAT,
        models_listed=True,
        model_configured=True,
        hyperparams_confirmed=True,
        run_confirmed=True,
        model_source="ultralytics",
        model_name="yolo11n",
    )
    defaults.update(kwargs)
    return WorkflowState(
        workflow_name="auto_labeling",
        dataset_name="test_ds",
        dataset_confirmed=True,
        auto_labeling=AutoLabelingState(**defaults),
    )


# ---------------------------------------------------------------------------
# run_failed() detection
# ---------------------------------------------------------------------------

def test_run_failed_detects_sentinel():
    assert run_failed(FAILED_RESULT) is True


def test_run_failed_detects_streaming_sentinel():
    assert run_failed(FAILED_STREAMING_RESULT) is True


def test_run_failed_fallback_without_sentinel_prefix():
    """A run tool that never got the RUN_FAILED prefix is still detected."""
    assert run_failed("Ensemble selection failed with exit code 2.\n") is True


def test_run_failed_false_on_success():
    assert run_failed("Auto-labeling workflow completed.\n\n**Result Summary:**") is False


def test_run_failed_ignores_phrase_inside_a_successful_report():
    """Captured logs can mention an exit code; only the headline decides."""
    result = (
        "Auto-labeling workflow completed.\n\n**Result Summary:**\n```\n"
        "retry 1: subprocess failed with exit code 1, retried and recovered\n"
        "```"
    )
    assert run_failed(result) is False


def test_run_failed_false_on_empty():
    assert run_failed("") is False


# ---------------------------------------------------------------------------
# parse_run_failure()
# ---------------------------------------------------------------------------

def test_parse_run_failure_uses_last_error_line():
    error, log_path = parse_run_failure(FAILED_RESULT)
    assert error == "RuntimeError: CUDA out of memory"
    assert log_path == "output/logs/last_anomaly_detection_log.txt"


def test_parse_run_failure_streaming_uses_last_stderr_line():
    error, log_path = parse_run_failure(FAILED_STREAMING_RESULT)
    assert error == "ValueError: no samples in dataset"
    assert log_path == "output/logs/last_auto_labeling_log.txt"


def test_parse_run_failure_without_log_path():
    error, log_path = parse_run_failure("RUN_FAILED: Error executing auto_labeling: boom")
    assert log_path == ""
    assert "boom" in error


def test_parse_run_failure_empty_error_block():
    result = (
        "RUN_FAILED: Auto-labeling failed with exit code 1.\n"
        "Error details:\n```\n```\n"
        "Full logs saved to `output/logs/x.txt`"
    )
    error, log_path = parse_run_failure(result)
    assert error == "no error output captured"
    assert log_path == "output/logs/x.txt"


# ---------------------------------------------------------------------------
# _finalize_auto_labeling() — the lock must not close on failure
# ---------------------------------------------------------------------------

def test_finalize_failure_does_not_set_complete_or_phase():
    pipeline = make_pipeline(_al_state())
    with patch.object(WorkflowState, "save"):
        asyncio.run(pipeline._finalize_auto_labeling(FAILED_STREAMING_RESULT, MagicMock()))
    al = pipeline.state.auto_labeling
    assert al.auto_labeling_complete is False
    assert al.phase == AutoLabelingPhase.PENDING


def test_finalize_failure_clears_run_consent():
    """A retry must show a new pre-run summary with the changed values."""
    pipeline = make_pipeline(_al_state(run_confirmed=True, run_awaiting_confirmation=True))
    with patch.object(WorkflowState, "save"):
        asyncio.run(pipeline._finalize_auto_labeling(FAILED_STREAMING_RESULT, MagicMock()))
    al = pipeline.state.auto_labeling
    assert al.run_confirmed is False
    assert al.run_awaiting_confirmation is False


def test_finalize_failure_skips_post_run_export():
    pipeline = make_pipeline(_al_state())
    mcp_client = MagicMock()
    mcp_client.call_tool = AsyncMock()
    with patch.object(WorkflowState, "save"):
        asyncio.run(pipeline._finalize_auto_labeling(FAILED_STREAMING_RESULT, mcp_client))
    mcp_client.call_tool.assert_not_called()


def test_finalize_failure_records_last_run():
    pipeline = make_pipeline(_al_state())
    with patch.object(WorkflowState, "save"):
        asyncio.run(pipeline._finalize_auto_labeling(FAILED_STREAMING_RESULT, MagicMock()))
    lr = pipeline.state.last_run
    assert lr.workflow == "auto_labeling"
    assert lr.status == "failed"
    assert lr.error == "ValueError: no samples in dataset"
    assert lr.attempts == 1


def test_finalize_failure_reply_mentions_error_and_retry():
    pipeline = make_pipeline(_al_state())
    with patch.object(WorkflowState, "save"):
        reply = asyncio.run(pipeline._finalize_auto_labeling(FAILED_STREAMING_RESULT, MagicMock()))
    assert "no samples in dataset" in reply
    assert "output/logs/last_auto_labeling_log.txt" in reply
    assert "run it again" in reply
    assert "hyperparameters" in reply


def test_finalize_success_still_locks_and_exports():
    """The success path is unchanged."""
    pipeline = make_pipeline(_al_state())
    success = "Auto-labeling workflow completed.\n\n**Result Summary:**\n```\nok\n```"
    with patch.object(WorkflowState, "save"), \
         patch.object(ChatPipeline, "_do_post_run_export", AsyncMock(return_value=("", True))):
        asyncio.run(pipeline._finalize_auto_labeling(success, MagicMock()))
    al = pipeline.state.auto_labeling
    assert al.auto_labeling_complete is True
    assert al.phase == AutoLabelingPhase.TRAINING
    assert pipeline.state.last_run.status == "success"


# ---------------------------------------------------------------------------
# The tool list stays open after a failure
# ---------------------------------------------------------------------------

def test_tools_after_failure_allow_reconfigure_and_retry():
    pipeline = make_pipeline(_al_state())
    with patch.object(WorkflowState, "save"):
        asyncio.run(pipeline._finalize_auto_labeling(FAILED_STREAMING_RESULT, MagicMock()))
    names = pipeline.state.valid_tool_names()
    for tool in (
        "configure_auto_labeling",
        "set_auto_labeling_hyperparams",
        "list_model_sources_and_models",
        "set_selected_dataset",
        "run_auto_labeling",
    ):
        assert tool in names


def test_tools_after_success_are_locked_to_import():
    """Contrast case: a successful run does still lock the workflow."""
    pipeline = make_pipeline(_al_state())
    success = "Auto-labeling workflow completed."
    with patch.object(WorkflowState, "save"), \
         patch.object(ChatPipeline, "_do_post_run_export", AsyncMock(return_value=("", True))):
        asyncio.run(pipeline._finalize_auto_labeling(success, MagicMock()))
    names = pipeline.state.valid_tool_names()
    assert "set_auto_labeling_hyperparams" not in names
    assert "import_from_cvat" in names


# ---------------------------------------------------------------------------
# WorkflowState.record_run() / failed_run()
# ---------------------------------------------------------------------------

def test_record_run_counts_consecutive_attempts():
    state = WorkflowState(workflow_name="auto_labeling")
    with patch.object(WorkflowState, "save"):
        state.record_run("auto_labeling", failed=True, error="a")
        state.record_run("auto_labeling", failed=True, error="b")
    assert state.last_run.attempts == 2
    assert state.last_run.error == "b"


def test_record_run_restarts_count_for_another_workflow():
    state = WorkflowState(workflow_name="auto_labeling")
    with patch.object(WorkflowState, "save"):
        state.record_run("auto_labeling", failed=True, error="a")
        state.record_run("class_mapping", failed=True, error="b")
    assert state.last_run.attempts == 1


def test_record_run_truncates_long_errors():
    state = WorkflowState(workflow_name="auto_labeling")
    with patch.object(WorkflowState, "save"):
        state.record_run("auto_labeling", failed=True, error="x" * 900)
    assert len(state.last_run.error) == 500


def test_failed_run_returns_record_for_active_workflow():
    state = WorkflowState(
        workflow_name="auto_labeling",
        last_run=LastRun(workflow="auto_labeling", status="failed", error="e", attempts=1),
    )
    assert state.failed_run().error == "e"


def test_failed_run_none_after_workflow_switch():
    state = WorkflowState(
        workflow_name="class_mapping",
        last_run=LastRun(workflow="auto_labeling", status="failed", error="e", attempts=1),
    )
    assert state.failed_run() is None


def test_failed_run_none_on_success():
    state = WorkflowState(
        workflow_name="auto_labeling",
        last_run=LastRun(workflow="auto_labeling", status="success", attempts=1),
    )
    assert state.failed_run() is None


def test_failed_run_none_without_record():
    assert WorkflowState(workflow_name="auto_labeling").failed_run() is None


# ---------------------------------------------------------------------------
# Migration keeps last_run loadable
# ---------------------------------------------------------------------------

def test_migrate_keeps_last_run():
    raw = WorkflowState._migrate({
        "workflow_name": "auto_labeling",
        "last_run": {"workflow": "auto_labeling", "status": "failed",
                     "error": "e", "log_path": "p", "attempts": 3},
    })
    assert WorkflowState.model_validate(raw).last_run.attempts == 3


def test_migrate_drops_unknown_last_run_key():
    """extra='forbid' would otherwise reject the whole state and lose the session."""
    raw = WorkflowState._migrate({
        "workflow_name": "auto_labeling",
        "last_run": {"status": "failed", "stale_key": 1},
    })
    state = WorkflowState.model_validate(raw)
    assert state.last_run.status == "failed"


def test_state_without_last_run_still_loads():
    raw = WorkflowState._migrate({"workflow_name": "auto_labeling"})
    assert WorkflowState.model_validate(raw).last_run is None


# ---------------------------------------------------------------------------
# Other workflows: failure reply instead of an LLM summary of the traceback
# ---------------------------------------------------------------------------

def test_anomaly_detection_failure_skips_summarizer():
    state = WorkflowState(
        workflow_name="anomaly_detection",
        dataset_name="test_ds",
        dataset_confirmed=True,
        anomaly_detection=AnomalyDetectionState(model_configured=True, data_source_set=True),
    )
    pipeline = make_pipeline(state)
    mcp_client = MagicMock()
    mcp_client.call_tool = AsyncMock(return_value=FAILED_RESULT)

    with patch.object(WorkflowState, "save"):
        result, routings = asyncio.run(pipeline._handle_run_anomaly_detection(mcp_client))

    pipeline.llm.summarize_anomaly_detection_output.assert_not_called()
    assert isinstance(routings[0], HardStop)
    assert "CUDA out of memory" in routings[0].reply
    assert pipeline.state.last_run.status == "failed"
    # Flags stay set, so the parameters can be changed and the run repeated.
    assert pipeline.state.anomaly_detection.model_configured is True
    assert "set_anomaly_detection_hyperparams" in state.valid_tool_names()


def test_anomaly_detection_success_records_success():
    state = WorkflowState(
        workflow_name="anomaly_detection",
        dataset_name="test_ds",
        dataset_confirmed=True,
        anomaly_detection=AnomalyDetectionState(model_configured=True, data_source_set=True),
        last_run=LastRun(workflow="anomaly_detection", status="failed", error="e", attempts=1),
    )
    pipeline = make_pipeline(state)
    pipeline.llm.summarize_anomaly_detection_output = AsyncMock(return_value="summary")
    mcp_client = MagicMock()
    mcp_client.call_tool = AsyncMock(return_value="Anomaly Detection completed successfully.")

    with patch.object(WorkflowState, "save"):
        asyncio.run(pipeline._handle_run_anomaly_detection(mcp_client))

    assert pipeline.state.last_run.status == "success"
    assert pipeline.state.failed_run() is None
