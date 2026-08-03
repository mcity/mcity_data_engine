"""
An export that delivers 0 images fails the run.

An empty export leaves the user with nothing to annotate and nothing to import,
so it must not lock the workflow. Two entry points are covered:

  - the manual path (export_to_cvat / export_to_label_studio called directly):
    no task ids are recorded and the phase does not advance
  - the auto path (the export that runs after auto-labeling): the run is
    recorded as failed and the workflow stays reconfigurable

WorkflowState.save is patched throughout: these tests must not write config.py.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from chat_pipeline import ChatPipeline, HardStop, export_empty
from validate_workflow_state import (
    AutoLabelingPhase, AutoLabelingState, LabelingBackend, LabelingPath, WorkflowState,
)

EMPTY_CVAT = (
    "EXPORT_NO_IMAGES: Dataset 'custom_dataset2' contains 0 images, "
    "so there is nothing to export to CVAT."
)
EMPTY_LS = (
    "EXPORT_NO_IMAGES: Label Studio export uploaded 0 of 30 images: "
    "connection reset by peer"
)
GOOD_CVAT = (
    "Dataset 'custom_dataset2' uploaded to CVAT successfully (1 task).\n"
    "Task IDs: 2474952\n"
    "Images: 30\n"
    "  - Task 2474952: https://app.cvat.ai/tasks/2474952"
)


def make_pipeline(state: WorkflowState) -> ChatPipeline:
    pipeline = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    pipeline.state = state
    return pipeline


def _al_state(**kwargs) -> WorkflowState:
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
        dataset_name="custom_dataset2",
        dataset_confirmed=True,
        auto_labeling=AutoLabelingState(**defaults),
    )


# ---------------------------------------------------------------------------
# export_empty() detection
# ---------------------------------------------------------------------------

def test_export_empty_detects_cvat_sentinel():
    assert export_empty(EMPTY_CVAT) is True


def test_export_empty_detects_label_studio_sentinel():
    assert export_empty(EMPTY_LS) is True


def test_export_empty_detects_zero_image_count():
    """Backstop for a tool that reports a task with no images in it."""
    assert export_empty("Task IDs: 5\nImages: 0\n  - Task 5: url") is True


def test_export_empty_false_on_good_export():
    assert export_empty(GOOD_CVAT) is False


def test_export_empty_does_not_match_a_count_ending_in_zero():
    assert export_empty("Task IDs: 5\nImages: 30\n") is False


# ---------------------------------------------------------------------------
# Manual path: the export tool is called directly
# ---------------------------------------------------------------------------

def _run_manual_export(export_result: str, backend: str = LabelingBackend.CVAT):
    state = _al_state(
        labeling_path=LabelingPath.MANUAL,
        labeling_backend=backend,
        manual_classes=["Car"],
        export_confirmed=True,
        model_configured=False,
        hyperparams_confirmed=False,
        run_confirmed=False,
    )
    pipeline = make_pipeline(state)
    mcp_client = MagicMock()
    mcp_client.call_tool = AsyncMock(return_value=export_result)
    env = {"CVAT_ACCESS_TOKEN": "t", "LS_TOKEN": "t"}
    with patch.object(WorkflowState, "save"), patch.dict("os.environ", env):
        result, routings = asyncio.run(
            pipeline._handle_export_generic(
                backend, {"dataset_name": "custom_dataset2", "classes": ["Car"]}, mcp_client
            )
        )
    return pipeline, routings


def test_empty_cvat_export_stops_with_a_clear_reply():
    _, routings = _run_manual_export(EMPTY_CVAT)
    assert isinstance(routings[0], HardStop)
    assert "0 images" in routings[0].reply
    assert "nothing to annotate" in routings[0].reply


def test_empty_cvat_export_records_no_task_ids():
    pipeline, _ = _run_manual_export(EMPTY_CVAT)
    assert pipeline.state.auto_labeling.cvat_task_ids == []


def test_empty_cvat_export_does_not_advance_phase():
    pipeline, _ = _run_manual_export(EMPTY_CVAT)
    assert pipeline.state.auto_labeling.phase == AutoLabelingPhase.PENDING


def test_empty_cvat_export_keeps_the_export_tool_available():
    pipeline, _ = _run_manual_export(EMPTY_CVAT)
    assert "export_to_cvat" in pipeline.state.valid_tool_names()


def test_empty_label_studio_export_stops_too():
    _, routings = _run_manual_export(EMPTY_LS, backend=LabelingBackend.LABEL_STUDIO)
    assert isinstance(routings[0], HardStop)
    assert "uploaded 0 of 30 images" in routings[0].reply


def test_good_cvat_export_still_records_tasks_and_locks():
    pipeline, routings = _run_manual_export(GOOD_CVAT)
    assert pipeline.state.auto_labeling.cvat_task_ids == [2474952]
    assert pipeline.state.auto_labeling.phase == AutoLabelingPhase.ANNOTATING


# ---------------------------------------------------------------------------
# Auto path: the export that runs after auto-labeling
# ---------------------------------------------------------------------------

RUN_OK = "Auto-labeling workflow completed.\n\n**Result Summary:**\n```\nok\n```"


def _finalize_with_export(export_result: str):
    pipeline = make_pipeline(_al_state())
    mcp_client = MagicMock()
    mcp_client.call_tool = AsyncMock(return_value=export_result)
    with patch.object(WorkflowState, "save"):
        reply = asyncio.run(pipeline._finalize_auto_labeling(RUN_OK, mcp_client))
    return pipeline, reply


def test_empty_export_after_run_marks_the_run_failed():
    pipeline, _ = _finalize_with_export(EMPTY_CVAT)
    lr = pipeline.state.last_run
    assert lr.status == "failed"
    assert "0 images" in lr.error


def test_empty_export_after_run_does_not_lock():
    pipeline, _ = _finalize_with_export(EMPTY_CVAT)
    al = pipeline.state.auto_labeling
    assert al.phase == AutoLabelingPhase.PENDING
    assert al.auto_labeling_complete is False


def test_empty_export_after_run_clears_run_consent():
    pipeline, _ = _finalize_with_export(EMPTY_CVAT)
    assert pipeline.state.auto_labeling.run_confirmed is False


def test_empty_export_after_run_keeps_parameters_editable():
    pipeline, _ = _finalize_with_export(EMPTY_CVAT)
    names = pipeline.state.valid_tool_names()
    for tool in (
        "configure_auto_labeling",
        "set_auto_labeling_hyperparams",
        "set_selected_dataset",
        "run_auto_labeling",
    ):
        assert tool in names, tool


def test_empty_export_after_run_explains_both_the_run_and_the_export():
    pipeline, reply = _finalize_with_export(EMPTY_CVAT)
    assert "0 images" in reply
    assert "Nothing was reset" in reply
    assert "run it again" in reply


def test_export_exception_after_run_also_fails_the_run():
    """A crashing export leaves nothing to import either."""
    pipeline = make_pipeline(_al_state())
    mcp_client = MagicMock()
    mcp_client.call_tool = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch.object(WorkflowState, "save"):
        reply = asyncio.run(pipeline._finalize_auto_labeling(RUN_OK, mcp_client))
    assert pipeline.state.last_run.status == "failed"
    assert pipeline.state.auto_labeling.phase == AutoLabelingPhase.PENDING
    assert "connection refused" in reply


def test_good_export_after_run_locks_and_records_success():
    pipeline, reply = _finalize_with_export(GOOD_CVAT)
    al = pipeline.state.auto_labeling
    assert al.phase == AutoLabelingPhase.TRAINING
    assert al.auto_labeling_complete is True
    assert al.cvat_task_ids == [2474952]
    assert pipeline.state.last_run.status == "success"
    assert "import the labels back" in reply
    assert "import_from_cvat" in pipeline.state.valid_tool_names()
