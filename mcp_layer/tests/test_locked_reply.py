"""
The reply the agent sends when the workflow is locked mid-run.

A HardStop reply becomes an assistant turn in the browser's history and comes
straight back to the model on the next request. This particular reply used to
carry the instruction meant for the model — "…then call reset_workflow_state()"
— so that order stayed in context for four turns. It fired three turns later,
while the user was choosing a dataset, and wiped the session.

The reply must therefore name no tool. It must also not be a yes/no question: a
"?" invites a bare "yes" that would read as consent to discard everything.

WorkflowState.save is patched: these tests must not write config.py.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from chat_pipeline import ChatPipeline, HardStop
from validate_workflow_state import AutoLabelingPhase, AutoLabelingState, WorkflowState


def _locked_state(phase: str) -> WorkflowState:
    """A session that has exported or run, and is waiting for the import step."""
    return WorkflowState(
        workflow_name="auto_labeling",
        dataset_name="custom_dataset5",
        dataset_confirmed=True,
        auto_labeling=AutoLabelingState(phase=phase),
    )


def _switch(state: WorkflowState) -> tuple[str, list]:
    pipeline = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    pipeline.state = state
    client = MagicMock()
    client.call_tool = AsyncMock(return_value="")
    with patch.object(WorkflowState, "save"):
        return asyncio.run(pipeline._handle_select_or_switch_workflow(
            "switch_workflow", {"workflow_name": "auto_labeling"}, client, []
        ))


def test_locked_reply_names_no_tool():
    result, routings = _switch(_locked_state(AutoLabelingPhase.TRAINING))

    assert isinstance(routings[0], HardStop)
    assert "reset_workflow_state" not in routings[0].reply
    assert "(" not in routings[0].reply
    assert "locked" in routings[0].reply
    # The model still receives the full instruction in the tool result.
    assert "reset_workflow_state()" in result


def test_locked_reply_is_not_a_yes_no_question():
    _, routings = _switch(_locked_state(AutoLabelingPhase.ANNOTATING))
    assert "?" not in routings[0].reply


def test_both_locked_phases_are_reported_by_name():
    _, training = _switch(_locked_state(AutoLabelingPhase.TRAINING))
    _, annotating = _switch(_locked_state(AutoLabelingPhase.ANNOTATING))
    assert "auto-labeling run" in training[0].reply
    assert "annotation export" in annotating[0].reply
