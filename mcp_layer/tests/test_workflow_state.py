"""
Priority 1: valid_tool_names() and all six workflow-specific tool-filtering functions.

Each function under test is pure (no I/O); instances are constructed directly from
Pydantic models — no state is loaded from or saved to config.py.
"""
import pytest
from validate_workflow_state import (
    WorkflowState, AutoLabelingState, ClassMappingState,
    AnomalyDetectionState, EmbeddingSelectionState,
    ZeroShotAutoLabelingState, EnsembleSelectionState,
    LabelingBackend, AutoLabelingPhase,
)

ALWAYS = {"send_reply", "switch_workflow"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _al_state(**kwargs) -> WorkflowState:
    """WorkflowState for auto_labeling with a confirmed dataset."""
    al = AutoLabelingState(**kwargs)
    return WorkflowState(
        workflow_name="auto_labeling",
        dataset_name="test_ds",
        dataset_confirmed=True,
        auto_labeling=al,
    )


# ---------------------------------------------------------------------------
# valid_tool_names() — base routing
# ---------------------------------------------------------------------------

def test_valid_tool_names_no_workflow_selected():
    state = WorkflowState()
    result = state.valid_tool_names()
    assert result == ALWAYS | {"select_workflow"}


def test_valid_tool_names_non_class_mapping_workflow_dataset_not_confirmed():
    state = WorkflowState(workflow_name="auto_labeling")
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_selected_dataset", "list_datasets"}


def test_valid_tool_names_dataset_not_confirmed_anomaly_detection():
    state = WorkflowState(workflow_name="anomaly_detection")
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_selected_dataset", "list_datasets"}


# ---------------------------------------------------------------------------
# _auto_labeling_tools — Phase-locked states
# ---------------------------------------------------------------------------

def test_auto_labeling_tools_phase_complete():
    state = _al_state(phase=AutoLabelingPhase.COMPLETE, labeling_backend=LabelingBackend.CVAT)
    assert state.valid_tool_names() == ALWAYS | {"launch_voxel51_session"}


def test_auto_labeling_tools_phase_annotating_cvat_returns_import_from_cvat():
    state = _al_state(phase=AutoLabelingPhase.ANNOTATING, labeling_backend=LabelingBackend.CVAT)
    assert state.valid_tool_names() == ALWAYS | {"import_from_cvat"}


def test_auto_labeling_tools_phase_annotating_label_studio_returns_import_from_ls():
    state = _al_state(phase=AutoLabelingPhase.ANNOTATING, labeling_backend=LabelingBackend.LABEL_STUDIO)
    assert state.valid_tool_names() == ALWAYS | {"import_from_label_studio"}


def test_auto_labeling_tools_phase_training_cvat_returns_import_from_cvat():
    state = _al_state(phase=AutoLabelingPhase.TRAINING, labeling_backend=LabelingBackend.CVAT)
    assert state.valid_tool_names() == ALWAYS | {"import_from_cvat"}


def test_auto_labeling_tools_phase_training_label_studio_returns_import_from_ls():
    state = _al_state(phase=AutoLabelingPhase.TRAINING, labeling_backend=LabelingBackend.LABEL_STUDIO)
    assert state.valid_tool_names() == ALWAYS | {"import_from_label_studio"}


# ---------------------------------------------------------------------------
# _auto_labeling_tools — Backend-selection gate (al=None / no path yet)
# ---------------------------------------------------------------------------

def test_auto_labeling_tools_al_substate_none_awaiting_backend():
    """With al=None (not yet initialized), should ask for backend detection."""
    state = WorkflowState(
        workflow_name="auto_labeling",
        dataset_name="test_ds",
        dataset_confirmed=True,
        auto_labeling=None,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_selected_dataset", "get_labeling_backend", "set_labeling_backend"}


def test_auto_labeling_tools_no_path_backend_empty_awaiting_backend():
    state = _al_state(labeling_path="", labeling_backend="")
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_selected_dataset", "get_labeling_backend", "set_labeling_backend"}


def test_auto_labeling_tools_no_path_backend_confirmed_cvat():
    """Backend confirmed as CVAT but path not chosen — show path-selection tools."""
    state = _al_state(labeling_path="", labeling_backend=LabelingBackend.CVAT)
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_selected_dataset", "set_labeling_path", "set_labeling_backend"}


def test_auto_labeling_tools_no_path_backend_confirmed_label_studio():
    state = _al_state(labeling_path="", labeling_backend=LabelingBackend.LABEL_STUDIO)
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_selected_dataset", "set_labeling_path", "set_labeling_backend"}


# ---------------------------------------------------------------------------
# _auto_labeling_tools — Manual path (CVAT)
# ---------------------------------------------------------------------------

def test_auto_labeling_tools_manual_cvat_no_classes_no_task():
    state = _al_state(
        labeling_path="manual", labeling_backend=LabelingBackend.CVAT,
        manual_classes=[], cvat_task_id=0,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_selected_dataset", "export_to_cvat", "set_labeling_backend", "set_labeling_path"}


def test_auto_labeling_tools_manual_cvat_classes_set_export_not_confirmed():
    state = _al_state(
        labeling_path="manual", labeling_backend=LabelingBackend.CVAT,
        manual_classes=["cat", "dog"], export_confirmed=False,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"confirm_export", "set_selected_dataset", "set_labeling_backend", "set_labeling_path"}


def test_auto_labeling_tools_manual_cvat_task_exported():
    state = _al_state(
        labeling_path="manual", labeling_backend=LabelingBackend.CVAT,
        cvat_task_id=42,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"import_from_cvat"}


def test_auto_labeling_tools_manual_cvat_labels_imported():
    state = _al_state(
        labeling_path="manual", labeling_backend=LabelingBackend.CVAT,
        labels_imported=True,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"launch_voxel51_session"}


# ---------------------------------------------------------------------------
# _auto_labeling_tools — Manual path (Label Studio)
# ---------------------------------------------------------------------------

def test_auto_labeling_tools_manual_ls_no_classes_no_tasks():
    state = _al_state(
        labeling_path="manual", labeling_backend=LabelingBackend.LABEL_STUDIO,
        manual_classes=[], ls_task_ids=[],
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_selected_dataset", "export_to_label_studio", "set_labeling_backend", "set_labeling_path"}


def test_auto_labeling_tools_manual_ls_classes_set_export_not_confirmed():
    state = _al_state(
        labeling_path="manual", labeling_backend=LabelingBackend.LABEL_STUDIO,
        manual_classes=["cat"], ls_task_ids=[], export_confirmed=False,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"confirm_export", "set_selected_dataset", "set_labeling_backend", "set_labeling_path"}


def test_auto_labeling_tools_manual_ls_tasks_exported():
    state = _al_state(
        labeling_path="manual", labeling_backend=LabelingBackend.LABEL_STUDIO,
        ls_task_ids=[101, 102],
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"import_from_label_studio"}


def test_auto_labeling_tools_manual_ls_labels_imported():
    state = _al_state(
        labeling_path="manual", labeling_backend=LabelingBackend.LABEL_STUDIO,
        labels_imported=True,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"launch_voxel51_session"}


# ---------------------------------------------------------------------------
# _auto_labeling_tools — Auto path
# ---------------------------------------------------------------------------

def test_auto_labeling_tools_auto_path_models_listed_true_both_config_tools_visible():
    """
    Post-Session-2 collapse: configure_auto_labeling and set_auto_labeling_hyperparams
    are BOTH visible once models_listed=True (their precondition guards handle premature
    calls). This was the key collapsed behavior to lock in.
    """
    state = _al_state(
        labeling_path="auto", labeling_backend=LabelingBackend.CVAT,
        models_listed=True, model_configured=False, auto_labeling_complete=False,
    )
    result = state.valid_tool_names()
    assert "configure_auto_labeling" in result
    assert "set_auto_labeling_hyperparams" in result
    assert "list_model_sources_and_models" in result
    assert "run_auto_labeling" in result


def test_auto_labeling_tools_auto_path_not_complete_includes_list_and_set_dataset():
    state = _al_state(
        labeling_path="auto", labeling_backend=LabelingBackend.CVAT,
        auto_labeling_complete=False,
    )
    result = state.valid_tool_names()
    assert "list_model_sources_and_models" in result
    assert "set_selected_dataset" in result
    assert "set_labeling_backend" in result
    assert "set_labeling_path" in result


def test_auto_labeling_tools_auto_path_run_awaiting_confirmation_shows_confirm_run():
    state = _al_state(
        labeling_path="auto", labeling_backend=LabelingBackend.CVAT,
        auto_labeling_complete=False,
        run_awaiting_confirmation=True, run_confirmed=False,
    )
    result = state.valid_tool_names()
    assert "confirm_run" in result
    assert "run_auto_labeling" not in result


def test_auto_labeling_tools_auto_path_not_awaiting_shows_run_auto_labeling():
    state = _al_state(
        labeling_path="auto", labeling_backend=LabelingBackend.CVAT,
        auto_labeling_complete=False,
        run_awaiting_confirmation=False,
    )
    result = state.valid_tool_names()
    assert "run_auto_labeling" in result
    assert "confirm_run" not in result


def test_auto_labeling_tools_auto_path_complete_labels_imported():
    state = _al_state(
        labeling_path="auto", labeling_backend=LabelingBackend.CVAT,
        auto_labeling_complete=True, labels_imported=True,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"launch_voxel51_session"}


def test_auto_labeling_tools_auto_path_complete_not_imported_cvat():
    state = _al_state(
        labeling_path="auto", labeling_backend=LabelingBackend.CVAT,
        auto_labeling_complete=True, labels_imported=False,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"import_from_cvat"}


def test_auto_labeling_tools_auto_path_complete_not_imported_ls():
    state = _al_state(
        labeling_path="auto", labeling_backend=LabelingBackend.LABEL_STUDIO,
        auto_labeling_complete=True, labels_imported=False,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"import_from_label_studio"}


# ---------------------------------------------------------------------------
# _class_mapping_tools
# ---------------------------------------------------------------------------

def _cm_state(**kwargs) -> WorkflowState:
    cm = ClassMappingState(**kwargs)
    return WorkflowState(workflow_name="class_mapping", class_mapping=cm)


def test_class_mapping_tools_no_substate():
    """class_mapping with cm=None → show listing and configure tools."""
    state = WorkflowState(workflow_name="class_mapping", class_mapping=None)
    result = state.valid_tool_names()
    assert result == ALWAYS | {"list_class_mapping_models", "configure_class_mapping_model"}


def test_class_mapping_tools_model_not_configured():
    state = _cm_state(model_configured=False)
    result = state.valid_tool_names()
    assert result == ALWAYS | {"list_class_mapping_models", "configure_class_mapping_model"}


def test_class_mapping_tools_model_configured_awaiting_source():
    state = _cm_state(model_configured=True, source_dataset_set=False)
    result = state.valid_tool_names()
    assert result == ALWAYS | {
        "set_class_mapping_dataset_source", "set_selected_dataset", "launch_voxel51_session"
    }


def test_class_mapping_tools_source_set_awaiting_target():
    state = _cm_state(model_configured=True, source_dataset_set=True, target_dataset_set=False)
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_class_mapping_dataset_target", "launch_voxel51_session"}


def test_class_mapping_tools_source_and_target_set_awaiting_labels():
    state = _cm_state(
        model_configured=True, source_dataset_set=True,
        target_dataset_set=True, candidate_labels_set=False,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_class_mapping_candidate_labels", "launch_voxel51_session"}


def test_class_mapping_tools_all_set_ready_to_run():
    state = _cm_state(
        model_configured=True, source_dataset_set=True,
        target_dataset_set=True, candidate_labels_set=True,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"run_class_mapping", "launch_voxel51_session"}


# ---------------------------------------------------------------------------
# _anomaly_detection_tools
# ---------------------------------------------------------------------------

def _ad_state(**kwargs) -> WorkflowState:
    ad = AnomalyDetectionState(**kwargs)
    return WorkflowState(
        workflow_name="anomaly_detection",
        dataset_name="test_ds", dataset_confirmed=True,
        anomaly_detection=ad,
    )


def test_anomaly_detection_tools_no_substate():
    state = WorkflowState(
        workflow_name="anomaly_detection",
        dataset_name="test_ds", dataset_confirmed=True,
        anomaly_detection=None,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {
        "list_anomaly_detection_models", "configure_anomaly_detection_model", "launch_voxel51_session"
    }


def test_anomaly_detection_tools_model_not_configured():
    state = _ad_state(model_configured=False)
    result = state.valid_tool_names()
    assert result == ALWAYS | {
        "list_anomaly_detection_models", "configure_anomaly_detection_model", "launch_voxel51_session"
    }


def test_anomaly_detection_tools_model_configured_awaiting_data_source():
    state = _ad_state(model_configured=True, data_source_set=False)
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_anomaly_detection_data_source", "launch_voxel51_session"}


def test_anomaly_detection_tools_data_source_set_ready_to_run():
    state = _ad_state(model_configured=True, data_source_set=True)
    result = state.valid_tool_names()
    assert result == ALWAYS | {
        "set_anomaly_detection_hyperparams", "run_anomaly_detection", "launch_voxel51_session"
    }


# ---------------------------------------------------------------------------
# _embedding_selection_tools
# ---------------------------------------------------------------------------

def _es_state(**kwargs) -> WorkflowState:
    es = EmbeddingSelectionState(**kwargs)
    return WorkflowState(
        workflow_name="embedding_selection",
        dataset_name="test_ds", dataset_confirmed=True,
        embedding_selection=es,
    )


def test_embedding_selection_tools_no_substate():
    state = WorkflowState(
        workflow_name="embedding_selection",
        dataset_name="test_ds", dataset_confirmed=True,
        embedding_selection=None,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"list_embedding_selection_models", "configure_embedding_selection_model"}


def test_embedding_selection_tools_model_not_configured():
    state = _es_state(model_configured=False)
    result = state.valid_tool_names()
    assert result == ALWAYS | {"list_embedding_selection_models", "configure_embedding_selection_model"}


def test_embedding_selection_tools_model_configured_ready_to_run():
    state = _es_state(model_configured=True)
    result = state.valid_tool_names()
    assert result == ALWAYS | {"set_embedding_selection_params", "run_embedding_selection"}


# ---------------------------------------------------------------------------
# _zero_shot_tools
# ---------------------------------------------------------------------------

def _zs_state(**kwargs) -> WorkflowState:
    zs = ZeroShotAutoLabelingState(**kwargs)
    return WorkflowState(
        workflow_name="auto_labeling_zero_shot",
        dataset_name="test_ds", dataset_confirmed=True,
        auto_labeling_zero_shot=zs,
    )


def test_zero_shot_tools_no_substate():
    state = WorkflowState(
        workflow_name="auto_labeling_zero_shot",
        dataset_name="test_ds", dataset_confirmed=True,
        auto_labeling_zero_shot=None,
    )
    result = state.valid_tool_names()
    assert result == ALWAYS | {"list_zsal", "configure_auto_labeling_zero_shot_models"}


def test_zero_shot_tools_models_not_configured():
    state = _zs_state(models_configured=False)
    result = state.valid_tool_names()
    assert result == ALWAYS | {"list_zsal", "configure_auto_labeling_zero_shot_models"}


def test_zero_shot_tools_models_configured_classes_not_set():
    state = _zs_state(models_configured=True, classes_set=False)
    result = state.valid_tool_names()
    config_tools = {"set_auto_labeling_zero_shot_threshold", "set_auto_labeling_zero_shot_classes"}
    assert result == ALWAYS | config_tools


def test_zero_shot_tools_models_configured_classes_set_ready_to_run():
    state = _zs_state(models_configured=True, classes_set=True)
    result = state.valid_tool_names()
    config_tools = {"set_auto_labeling_zero_shot_threshold", "set_auto_labeling_zero_shot_classes"}
    assert result == ALWAYS | config_tools | {"run_zero_shot_auto_labeling"}


# ---------------------------------------------------------------------------
# _ensemble_tools
# ---------------------------------------------------------------------------

def _ens_state(**kwargs) -> WorkflowState:
    ens = EnsembleSelectionState(**kwargs)
    return WorkflowState(
        workflow_name="ensemble_selection",
        dataset_name="test_ds", dataset_confirmed=True,
        ensemble_selection=ens,
    )


def test_ensemble_tools_no_substate():
    state = WorkflowState(
        workflow_name="ensemble_selection",
        dataset_name="test_ds", dataset_confirmed=True,
        ensemble_selection=None,
    )
    result = state.valid_tool_names()
    config_tools = {"set_ensemble_selection_parameters", "set_ensemble_selection_classes"}
    assert result == ALWAYS | config_tools


def test_ensemble_tools_classes_not_set():
    state = _ens_state(classes_set=False)
    result = state.valid_tool_names()
    config_tools = {"set_ensemble_selection_parameters", "set_ensemble_selection_classes"}
    assert result == ALWAYS | config_tools


def test_ensemble_tools_classes_set_ready_to_run():
    state = _ens_state(classes_set=True)
    result = state.valid_tool_names()
    config_tools = {"set_ensemble_selection_parameters", "set_ensemble_selection_classes"}
    assert result == ALWAYS | config_tools | {"run_ensemble_selection"}
