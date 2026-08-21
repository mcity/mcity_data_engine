"""
The chat history lives in the browser, not on the server.

The page keeps every turn it has displayed and posts them all back with each
request, so reset_workflow_state cannot make the client forget anything. The
server therefore decides how much of that history the model may read:
turns_since_reset goes back to 0 with the reset, and chat_server trims the
posted history to it. Without the trim the model kept reading the session the
user had already discarded — that is how a leftover "call reset_workflow_state()"
line wiped a live session three turns after the reset it belonged to.

WorkflowState.save is patched wherever a test would otherwise write config.py.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from chat_pipeline import ChatPipeline
from chat_server import MAX_HISTORY_TURNS, history_for_session
from validate_workflow_state import WorkflowState

# What the browser posts: [user, assistant] pairs, oldest first.
HISTORY = [
    ["auto labeling", "Cannot reconfigure mid-run. ... call reset_workflow_state()"],
    ["reset workflow", "Are you sure? This cannot be undone."],
    ["yes", "Workflow, dataset, and session state have been reset."],
    ["auto labeling", "Here are the available datasets: ..."],
]


def _state(turns: int) -> WorkflowState:
    return WorkflowState(workflow_name="auto_labeling", turns_since_reset=turns)


# ---------------------------------------------------------------------------
# The trim itself
# ---------------------------------------------------------------------------

def test_first_request_after_a_reset_carries_no_history():
    assert history_for_session(HISTORY, _state(0)) == []


def test_window_grows_one_turn_per_request():
    assert history_for_session(HISTORY, _state(1)) == HISTORY[-1:]
    assert history_for_session(HISTORY, _state(2)) == HISTORY[-2:]
    assert history_for_session(HISTORY, _state(3)) == HISTORY[-3:]


def test_window_never_exceeds_the_four_turn_maximum():
    long_history = HISTORY + HISTORY  # eight turns
    assert history_for_session(long_history, _state(99)) == long_history[-MAX_HISTORY_TURNS:]


def test_no_state_falls_open_to_the_plain_window():
    long_history = HISTORY + HISTORY
    assert history_for_session(long_history, None) == long_history[-MAX_HISTORY_TURNS:]


def test_state_without_the_field_falls_open():
    """An old config.py has no turns_since_reset; that must not blank the history."""
    assert history_for_session(HISTORY, object()) == HISTORY


def test_empty_history_stays_empty():
    assert history_for_session([], _state(0)) == []
    assert history_for_session([], _state(4)) == []


# ---------------------------------------------------------------------------
# The counter that drives it
# ---------------------------------------------------------------------------

def test_reset_sets_the_counter_to_zero():
    with patch.object(WorkflowState, "save"):
        assert WorkflowState.reset().turns_since_reset == 0


def test_pipeline_reset_leaves_the_counter_at_zero():
    pipeline = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    pipeline.state = _state(4)
    pipeline._user_texts = ["reset workflow"]
    client = MagicMock()
    client.call_tool = AsyncMock(return_value="Workflow, dataset, and session state have been reset.")

    with patch.object(WorkflowState, "save"):
        asyncio.run(pipeline._handle_reset_workflow_state(client))

    assert pipeline.state.turns_since_reset == 0


def test_switching_workflow_keeps_the_counter():
    """A switch clears the parameters, not the conversation."""
    with patch.object(WorkflowState, "save"):
        fresh = _state(3).reset_for_workflow("anomaly_detection")
    assert fresh.turns_since_reset == 3
    assert fresh.workflow_name == "anomaly_detection"


def test_counter_survives_a_load():
    """_migrate drops unknown keys, so the field has to be on the known list."""
    migrated = WorkflowState._migrate(
        {"workflow_name": "auto_labeling", "turns_since_reset": 2}
    )
    assert migrated["turns_since_reset"] == 2


def test_config_without_the_counter_still_loads():
    migrated = WorkflowState._migrate({"workflow_name": "auto_labeling"})
    assert WorkflowState.model_validate(migrated).turns_since_reset == 0


# ---------------------------------------------------------------------------
# The logged failure, replayed as a sequence of requests
# ---------------------------------------------------------------------------

def test_dead_session_never_reaches_the_prompt_again():
    """Request by request, after the user resets and starts a new workflow.

    The leaked "call reset_workflow_state()" line is HISTORY[0]. It must not come
    back into the prompt on any later request.
    """
    posted = list(HISTORY)          # what the browser holds at the reset
    state = _state(4)

    with patch.object(WorkflowState, "save"):
        state = WorkflowState.reset()

    # Request: "auto labeling" — the whole discarded session is dropped.
    kept = history_for_session(posted, state)
    assert kept == []
    state.turns_since_reset += 1
    posted.append(["auto labeling", "Here are the available datasets: ..."])

    # Request: "custom_dataset5" — the turn that used to wipe the session.
    kept = history_for_session(posted, state)
    assert kept == posted[-1:]
    assert all("reset_workflow_state()" not in turn[1] for turn in kept)
