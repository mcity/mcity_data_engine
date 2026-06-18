"""
Priority 3: WorkflowState._migrate() backward-compatibility logic.

_migrate is a classmethod that normalizes old-shape WORKFLOW_STATE dicts to the
current schema. Tests verify that representative old shapes migrate without data
loss or silently dropped fields, and that the migrated dict is accepted by
WorkflowState.model_validate().
"""
import pytest
from validate_workflow_state import WorkflowState


def migrate(raw: dict) -> dict:
    """Convenience wrapper that calls _migrate in-place and returns the result."""
    return WorkflowState._migrate(dict(raw))


# ---------------------------------------------------------------------------
# workflow_name normalization
# ---------------------------------------------------------------------------

def test_migrate_workflow_name_none_becomes_empty_string():
    result = migrate({"workflow_name": None})
    assert result["workflow_name"] == ""


def test_migrate_workflow_name_already_present_unchanged():
    result = migrate({"workflow_name": "auto_labeling"})
    assert result["workflow_name"] == "auto_labeling"


# ---------------------------------------------------------------------------
# Old top-level auto_labeling fields lifted into subdict
# ---------------------------------------------------------------------------

def test_migrate_top_level_auto_labeling_complete_lifted_to_subdict():
    """Old shape had auto_labeling_complete at top level with workflow=auto_labeling."""
    raw = {
        "workflow_name": "auto_labeling",
        "dataset_name": "ds",
        "dataset_confirmed": True,
        "labeled_dataset_name": "",
        "auto_labeling_complete": True,
    }
    result = migrate(raw)
    assert "auto_labeling_complete" not in result
    assert result["auto_labeling"]["auto_labeling_complete"] is True


def test_migrate_top_level_cvat_task_id_lifted_to_subdict():
    raw = {
        "workflow_name": "auto_labeling",
        "auto_labeling_complete": False,
        "cvat_task_id": 42,
    }
    result = migrate(raw)
    assert "cvat_task_id" not in result
    assert result["auto_labeling"]["cvat_task_id"] == 42


def test_migrate_top_level_cvat_task_id_none_becomes_zero():
    """Old shape could have cvat_task_id=None; _migrate converts to 0."""
    raw = {
        "workflow_name": "auto_labeling",
        "cvat_task_id": None,
    }
    result = migrate(raw)
    assert result["auto_labeling"]["cvat_task_id"] == 0


def test_migrate_top_level_auto_labeling_fields_dropped_for_non_auto_labeling_workflow():
    """Top-level old fields are dropped when the workflow is not auto_labeling."""
    raw = {
        "workflow_name": "class_mapping",
        "auto_labeling_complete": True,
        "cvat_task_id": 5,
    }
    result = migrate(raw)
    assert "auto_labeling_complete" not in result
    assert "cvat_task_id" not in result


# ---------------------------------------------------------------------------
# auto_labeling subdict — missing optional fields get defaults
# ---------------------------------------------------------------------------

def test_migrate_auto_labeling_subdict_gets_missing_defaults():
    """An existing auto_labeling subdict that's missing new fields gets defaults added."""
    raw = {
        "workflow_name": "auto_labeling",
        "auto_labeling": {
            "auto_labeling_complete": False,
            "cvat_task_id": 0,
        },
    }
    result = migrate(raw)
    al = result["auto_labeling"]
    assert al.get("labeling_backend") == ""
    assert al.get("ls_task_ids") == []
    assert al.get("manual_classes") == []
    assert al.get("models_listed") is False
    assert al.get("export_confirmed") is False
    assert al.get("run_confirmed") is False
    assert al.get("run_awaiting_confirmation") is False
    assert al.get("model_source") == ""
    assert al.get("model_name") == ""
    assert al.get("phase") == ""


def test_migrate_auto_labeling_subdict_preserves_existing_values():
    raw = {
        "workflow_name": "auto_labeling",
        "auto_labeling": {
            "auto_labeling_complete": True,
            "cvat_task_id": 99,
            "labeling_backend": "cvat",
            "ls_task_ids": [1, 2],
            "manual_classes": ["car"],
            "models_listed": True,
            "model_source": "ultralytics",
            "model_name": "yolov8n",
            "export_confirmed": True,
            "run_confirmed": True,
            "run_awaiting_confirmation": False,
            "phase": "complete",
            "labeling_path": "auto",
            "model_configured": True,
            "hyperparams_confirmed": True,
            "labels_imported": True,
        },
    }
    result = migrate(raw)
    al = result["auto_labeling"]
    assert al["auto_labeling_complete"] is True
    assert al["cvat_task_id"] == 99
    assert al["labeling_backend"] == "cvat"
    assert al["ls_task_ids"] == [1, 2]
    assert al["manual_classes"] == ["car"]
    assert al["models_listed"] is True
    assert al["model_source"] == "ultralytics"
    assert al["model_name"] == "yolov8n"
    assert al["phase"] == "complete"


def test_migrate_auto_labeling_subdict_removes_pending_dataset_change():
    """pending_dataset_change is a removed field that should be stripped."""
    raw = {
        "workflow_name": "auto_labeling",
        "auto_labeling": {
            "pending_dataset_change": True,
            "auto_labeling_complete": False,
        },
    }
    result = migrate(raw)
    assert "pending_dataset_change" not in result["auto_labeling"]


# ---------------------------------------------------------------------------
# Unknown top-level keys are stripped
# ---------------------------------------------------------------------------

def test_migrate_unknown_top_level_keys_are_removed():
    raw = {
        "workflow_name": "",
        "some_old_flag": True,
        "another_legacy_field": "foo",
    }
    result = migrate(raw)
    assert "some_old_flag" not in result
    assert "another_legacy_field" not in result


def test_migrate_known_top_level_keys_are_preserved():
    raw = {
        "workflow_name": "auto_labeling",
        "dataset_name": "my_dataset",
        "dataset_confirmed": True,
        "labeled_dataset_name": "my_dataset_labeled",
    }
    result = migrate(raw)
    assert result["dataset_name"] == "my_dataset"
    assert result["dataset_confirmed"] is True
    assert result["labeled_dataset_name"] == "my_dataset_labeled"


# ---------------------------------------------------------------------------
# Round-trip: migrated dict is valid for WorkflowState.model_validate
# ---------------------------------------------------------------------------

def test_migrate_result_is_valid_workflow_state_fresh():
    raw = {"workflow_name": None}
    result = migrate(raw)
    state = WorkflowState.model_validate(result)
    assert state.workflow_name == ""


def test_migrate_result_is_valid_workflow_state_with_auto_labeling():
    raw = {
        "workflow_name": "auto_labeling",
        "dataset_name": "ds",
        "dataset_confirmed": True,
        "labeled_dataset_name": "",
        "auto_labeling_complete": True,
        "cvat_task_id": 7,
    }
    result = migrate(raw)
    state = WorkflowState.model_validate(result)
    assert state.workflow_name == "auto_labeling"
    assert state.auto_labeling is not None
    assert state.auto_labeling.auto_labeling_complete is True
    assert state.auto_labeling.cvat_task_id == 7
