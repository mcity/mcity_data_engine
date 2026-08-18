"""
Deleting a dataset is irreversible, so the pipeline never does it in one step.

The first delete_dataset call only produces a summary and arms the consent gate;
confirm_delete_dataset records the user's answer; a second delete_dataset call
does the erasing. These tests cover every way that sequence can be short-circuited:
a name the user never typed, a built-in dataset, a name that does not exist,
consent that belongs to a different dataset, and a replayed confirm.

WorkflowState.save is patched throughout: these tests must not write config.py.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from chat_pipeline import ChatPipeline, FallThrough, HardStop, Injection, Sentinels
from validate_workflow_state import WorkflowState

AVAILABLE = ["fisheye8k", "fisheye8k_mini", "custom_dataset2", "custom_dataset8"]
DELETED_OK = "Dataset `custom_dataset8` deleted (removed from FiftyOne; removed from datasets.yaml)."


def make_pipeline(
    state: WorkflowState, tool_result=DELETED_OK, user_texts=None
) -> tuple[ChatPipeline, MagicMock]:
    pipeline = ChatPipeline(mcp_client=MagicMock(), llm=MagicMock())
    pipeline.state = state
    # The provenance guard reads this; by default the user named the dataset.
    pipeline._user_texts = (
        ["delete custom_dataset8"] if user_texts is None else user_texts
    )
    client = MagicMock()

    async def _call(name, args=None):
        return json.dumps(AVAILABLE) if name == "list_datasets" else tool_result

    client.call_tool = AsyncMock(side_effect=_call)
    return pipeline, client


def _state(**kwargs) -> WorkflowState:
    """Session at the dataset-selection step, where deletion is allowed."""
    return WorkflowState(workflow_name="auto_labeling", **kwargs)


def _armed(name="custom_dataset8", **kwargs) -> WorkflowState:
    """Session in which the user has already confirmed deleting `name`."""
    return _state(
        delete_awaiting_confirmation=False, delete_confirmed=True,
        delete_pending_name=name, **kwargs
    )


def _delete(pipeline, client, **args):
    args.setdefault("dataset_name", "custom_dataset8")
    with patch.object(WorkflowState, "save"):
        return asyncio.run(pipeline._handle_delete_dataset(args, client))


def _confirm(pipeline):
    with patch.object(WorkflowState, "save"):
        return asyncio.run(pipeline._handle_confirm_delete_dataset())


# ---------------------------------------------------------------------------
# Step 1 — the first call must never delete
# ---------------------------------------------------------------------------

def test_first_call_asks_before_deleting_anything():
    pipeline, client = make_pipeline(_state())
    result, routings = _delete(pipeline, client)

    assert Sentinels.DELETE_NEEDS_CONFIRMATION in result
    assert isinstance(routings[0], HardStop)
    assert "cannot be undone" in routings[0].reply
    assert "custom_dataset8" in routings[0].reply
    # The MCP delete tool was never reached.
    assert [c.args[0] for c in client.call_tool.call_args_list] == ["list_datasets"]


def test_first_call_arms_the_consent_gate():
    pipeline, client = make_pipeline(_state())
    _delete(pipeline, client)

    assert pipeline.state.delete_awaiting_confirmation is True
    assert pipeline.state.delete_pending_name == "custom_dataset8"
    assert pipeline.state.delete_confirmed is False


def test_summary_says_files_are_kept_by_default():
    pipeline, client = make_pipeline(_state())
    _, routings = _delete(pipeline, client)
    assert "kept" in routings[0].reply


def test_summary_warns_when_files_will_be_erased():
    pipeline, client = make_pipeline(_state())
    result, routings = _delete(pipeline, client, delete_files=True)
    assert "erased" in routings[0].reply
    # The flag is echoed so the model can repeat it on the confirmed call.
    assert "delete_files=True" in result


# ---------------------------------------------------------------------------
# Step 1 — refusals that never reach the confirmation prompt
# ---------------------------------------------------------------------------

def test_name_the_user_never_typed_is_refused():
    """tool_choice='required' pressures the model into inventing arguments."""
    pipeline, client = make_pipeline(_state(), user_texts=["ok", "yes go ahead"])
    result, routings = _delete(pipeline, client)

    assert result == Sentinels.DATASET_NOT_NAMED
    assert isinstance(routings[0], HardStop)
    assert "exact name" in routings[0].reply
    assert pipeline.state.delete_awaiting_confirmation is False
    client.call_tool.assert_not_called()


def test_default_dataset_is_refused_without_asking():
    pipeline, client = make_pipeline(_state(), user_texts=["delete fisheye8k"])
    with patch.object(ChatPipeline, "_protected_dataset_names", return_value={"fisheye8k"}):
        result, routings = _delete(pipeline, client, dataset_name="fisheye8k")

    assert Sentinels.PROTECTED_DATASET in result
    assert isinstance(routings[0], HardStop)
    assert pipeline.state.delete_awaiting_confirmation is False
    client.call_tool.assert_not_called()


def test_protected_names_come_from_the_loader_fct_in_datasets_yaml(tmp_path):
    """Repo datasets are protected; ingested ones (load_custom_dataset) are not."""
    yaml_file = tmp_path / "datasets.yaml"
    yaml_file.write_text(
        "datasets:\n"
        "  - name: fisheye8k\n"
        "    loader_fct: load_fisheye_8k\n"
        "  - name: custom_dataset2\n"
        "    loader_fct: load_custom_dataset\n"
    )
    pipeline, _ = make_pipeline(_state())
    with patch("chat_pipeline.DATASETS_YAML", yaml_file):
        assert pipeline._protected_dataset_names() == {"fisheye8k"}


def test_unreadable_datasets_yaml_protects_nothing(tmp_path):
    """Fails open — the MCP tool still refuses a default dataset on its own."""
    pipeline, _ = make_pipeline(_state())
    with patch("chat_pipeline.DATASETS_YAML", tmp_path / "missing.yaml"):
        assert pipeline._protected_dataset_names() == set()


def test_real_datasets_yaml_classifies_shipped_and_ingested_entries():
    """Guards against the rule drifting from the file it reads."""
    pipeline, _ = make_pipeline(_state())
    protected = pipeline._protected_dataset_names()
    assert "fisheye8k" in protected
    assert "custom_dataset1" not in protected


def test_unknown_dataset_is_refused_with_the_real_list():
    pipeline, client = make_pipeline(_state(), user_texts=["delete typo_ds"])
    result, routings = _delete(pipeline, client, dataset_name="typo_ds")

    assert Sentinels.DATASET_NOT_FOUND in result
    assert "custom_dataset2" in routings[0].reply
    assert pipeline.state.delete_awaiting_confirmation is False


def test_unreachable_dataset_list_does_not_block_the_flow():
    """An empty list means 'unknown', not 'no datasets' — still ask the user."""
    pipeline, client = make_pipeline(_state())
    client.call_tool = AsyncMock(side_effect=RuntimeError("mcp down"))
    result, routings = _delete(pipeline, client)

    assert Sentinels.DELETE_NEEDS_CONFIRMATION in result
    assert isinstance(routings[0], HardStop)


# ---------------------------------------------------------------------------
# confirm_delete_dataset
# ---------------------------------------------------------------------------

def test_confirm_records_consent_for_the_pending_dataset():
    pipeline, _ = make_pipeline(
        _state(delete_awaiting_confirmation=True, delete_pending_name="custom_dataset8")
    )
    result, routings = _confirm(pipeline)

    assert pipeline.state.delete_confirmed is True
    assert pipeline.state.delete_awaiting_confirmation is False
    assert isinstance(routings[0], Injection)
    assert "custom_dataset8" in result


def test_confirm_without_a_pending_summary_is_blocked():
    """A replayed confirm must not pre-approve a deletion the user never saw."""
    pipeline, _ = make_pipeline(_state())
    result, routings = _confirm(pipeline)

    assert result == Sentinels.CONFIRM_NOT_PENDING
    assert pipeline.state.delete_confirmed is False
    assert isinstance(routings[0], Injection)


# ---------------------------------------------------------------------------
# Step 2 — the confirmed call
# ---------------------------------------------------------------------------

def test_confirmed_call_deletes_and_clears_the_gate():
    pipeline, client = make_pipeline(_armed())
    result, routings = _delete(pipeline, client)

    client.call_tool.assert_awaited_once_with(
        "delete_dataset", {"dataset_name": "custom_dataset8", "delete_files": False}
    )
    assert result == DELETED_OK
    assert isinstance(routings[0], Injection)
    assert "list_datasets" in routings[0].message
    assert pipeline.state.delete_confirmed is False
    assert pipeline.state.delete_pending_name == ""


def test_consent_for_one_dataset_cannot_delete_another():
    """The gate is keyed by name, so a swapped argument only re-prompts."""
    pipeline, client = make_pipeline(_armed(name="custom_dataset8"),
                                     user_texts=["delete custom_dataset2"])
    result, routings = _delete(pipeline, client, dataset_name="custom_dataset2")

    assert Sentinels.DELETE_NEEDS_CONFIRMATION in result
    assert isinstance(routings[0], HardStop)
    assert "delete_dataset" not in [c.args[0] for c in client.call_tool.call_args_list]


def test_deleting_the_selected_dataset_clears_the_session_pointer():
    state = _armed()
    state.dataset_name = "custom_dataset8"
    pipeline, client = make_pipeline(state)
    _delete(pipeline, client)

    assert pipeline.state.dataset_name == ""
    assert pipeline.state.dataset_confirmed is False


def test_tool_failure_spends_the_consent():
    """A failed delete must not leave a live approval behind for the next call."""
    pipeline, client = make_pipeline(
        _armed(), tool_result="DATASET_NOT_FOUND: 'custom_dataset8' does not exist."
    )
    result, routings = _delete(pipeline, client)

    assert Sentinels.DATASET_NOT_FOUND in result
    assert isinstance(routings[0], FallThrough)
    assert pipeline.state.delete_confirmed is False
    assert pipeline.state.delete_pending_name == ""


def test_transport_error_spends_the_consent():
    pipeline, client = make_pipeline(_armed())
    client.call_tool = AsyncMock(side_effect=RuntimeError("connection reset"))
    result, routings = _delete(pipeline, client)

    assert Sentinels.DELETE_FAILED in result
    assert isinstance(routings[0], FallThrough)
    assert pipeline.state.delete_confirmed is False


def test_delete_files_is_passed_through_on_the_confirmed_call():
    pipeline, client = make_pipeline(_armed())
    _delete(pipeline, client, delete_files=True)

    client.call_tool.assert_awaited_once_with(
        "delete_dataset", {"dataset_name": "custom_dataset8", "delete_files": True}
    )


# ---------------------------------------------------------------------------
# Dataset-list freshness after a deletion (chat_server.dataset_was_deleted)
# ---------------------------------------------------------------------------

def _tool_result(name: str, result: str) -> dict:
    return {"tool_call_id": "c1", "name": name, "fn_args": {}, "result": result}


def test_completed_deletion_marks_the_list_stale():
    from chat_server import dataset_was_deleted

    assert dataset_was_deleted([_tool_result("delete_dataset", DELETED_OK)]) is True


def test_stale_list_detected_even_when_listed_first_in_the_same_batch():
    """The list was fetched BEFORE the delete, so it still names the dataset."""
    from chat_server import dataset_was_deleted

    batch = [
        _tool_result("list_datasets", json.dumps(AVAILABLE)),
        _tool_result("delete_dataset", DELETED_OK),
    ]
    assert dataset_was_deleted(batch) is True


def test_delete_that_erased_nothing_leaves_the_list_valid():
    from chat_server import dataset_was_deleted

    for result in (
        f"{Sentinels.DELETE_NEEDS_CONFIRMATION} dataset='x' delete_files=False",
        Sentinels.DATASET_NOT_NAMED,
        f"{Sentinels.DATASET_NOT_FOUND}: 'x' does not exist.",
        f"{Sentinels.PROTECTED_DATASET}: 'fisheye8k' is a built-in dataset.",
        f"{Sentinels.DELETE_FAILED}: connection reset",
    ):
        assert dataset_was_deleted([_tool_result("delete_dataset", result)]) is False, result


def test_batches_without_a_delete_are_not_stale():
    from chat_server import dataset_was_deleted

    assert dataset_was_deleted([]) is False
    assert dataset_was_deleted([_tool_result("list_datasets", "[]")]) is False
    assert dataset_was_deleted([_tool_result("confirm_delete_dataset", "consent")]) is False


# ---------------------------------------------------------------------------
# SESSION_STATE hint — what the model is told on the confirmation turn
# ---------------------------------------------------------------------------

def _hint(state: WorkflowState) -> str:
    from chat_server import _build_state_hint
    return _build_state_hint(state)


def test_hint_tells_the_model_to_confirm_not_to_re_ask():
    """Without this the model re-calls delete_dataset and loops on the summary."""
    hint = _hint(_state(
        delete_awaiting_confirmation=True, delete_pending_name="custom_dataset6"
    ))
    assert "confirm_delete_dataset" in hint
    assert "custom_dataset6" in hint
    # The dataset-selection instruction would otherwise capture the user's "yes".
    assert "wait for the user to name a dataset" not in hint


def test_hint_pushes_a_confirmed_delete_to_completion():
    hint = _hint(_armed(name="custom_dataset6"))
    assert "delete_confirmed=custom_dataset6" in hint
    assert "delete_dataset(dataset_name='custom_dataset6')" in hint


def test_hint_is_unchanged_when_no_deletion_is_pending():
    hint = _hint(_state())
    assert "delete_pending" not in hint
    assert "wait for the user to name a dataset" in hint


# ---------------------------------------------------------------------------
# The consent window lasts exactly one turn
# ---------------------------------------------------------------------------

def _run_batch(state: WorkflowState, tool_names: list[str]):
    """Drive run() with a batch of already-dispatched tools."""
    pipeline, _ = make_pipeline(state)

    async def _fake_dispatch(fn_name, fn_args, call, mcp_client, messages, progress_cb=None):
        return "ok", []

    pipeline._dispatch = _fake_dispatch
    calls = [MagicMock(id=f"c{i}") for i, _ in enumerate(tool_names)]
    for c, n in zip(calls, tool_names):
        c.function.name = n
        c.function.arguments = "{}"
    with patch.object(WorkflowState, "save"), patch.object(
        WorkflowState, "load", return_value=state
    ):
        asyncio.run(pipeline.run(calls, [], None))
    return pipeline


def test_pending_delete_is_dropped_when_the_user_moves_on():
    state = _state(delete_awaiting_confirmation=True, delete_pending_name="custom_dataset6")
    pipeline = _run_batch(state, ["list_datasets"])

    assert pipeline.state.delete_awaiting_confirmation is False
    assert pipeline.state.delete_pending_name == ""


def test_pending_delete_survives_a_turn_that_acts_on_it():
    state = _state(delete_awaiting_confirmation=True, delete_pending_name="custom_dataset6")
    pipeline = _run_batch(state, ["confirm_delete_dataset"])

    assert pipeline.state.delete_pending_name == "custom_dataset6"


def test_recorded_consent_does_not_outlive_an_unrelated_turn():
    """A later 'yes' about something else must not revive an abandoned delete."""
    pipeline = _run_batch(_armed(name="custom_dataset6"), ["set_selected_dataset"])

    assert pipeline.state.delete_confirmed is False
    assert pipeline.state.delete_pending_name == ""
