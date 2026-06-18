"""
Priority 2: All can_* precondition guards across all six workflows.

Each guard is a method on a Pydantic state model. Tests verify:
  - Blocking case: precondition not met → (False, non-empty error string)
  - Passing case: precondition met → (True, "")

No I/O — models constructed directly.
"""
import pytest
from validate_workflow_state import (
    WorkflowState,
    AutoLabelingState,
    ClassMappingState,
    AnomalyDetectionState,
    EmbeddingSelectionState,
    ZeroShotAutoLabelingState,
    EnsembleSelectionState,
)


# ---------------------------------------------------------------------------
# WorkflowState.can_confirm_dataset
# ---------------------------------------------------------------------------

def test_can_confirm_dataset_blocks_when_no_workflow():
    state = WorkflowState()
    ok, msg = state.can_confirm_dataset()
    assert not ok
    assert msg


def test_can_confirm_dataset_passes_when_workflow_set():
    state = WorkflowState(workflow_name="auto_labeling")
    ok, msg = state.can_confirm_dataset()
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# AutoLabelingState.can_configure_auto_labeling
# ---------------------------------------------------------------------------

def test_can_configure_auto_labeling_blocks_when_models_not_listed():
    al = AutoLabelingState(models_listed=False)
    ok, msg = al.can_configure_auto_labeling()
    assert not ok
    assert msg


def test_can_configure_auto_labeling_passes_when_models_listed():
    al = AutoLabelingState(models_listed=True)
    ok, msg = al.can_configure_auto_labeling()
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# AutoLabelingState.can_set_auto_labeling_hyperparams
# ---------------------------------------------------------------------------

def test_can_set_hyperparams_blocks_when_model_not_configured():
    al = AutoLabelingState(model_configured=False)
    ok, msg = al.can_set_auto_labeling_hyperparams()
    assert not ok
    assert msg


def test_can_set_hyperparams_passes_when_model_configured():
    al = AutoLabelingState(model_configured=True)
    ok, msg = al.can_set_auto_labeling_hyperparams()
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# AutoLabelingState.can_run_auto_labeling
# ---------------------------------------------------------------------------

def test_can_run_auto_labeling_blocks_when_dataset_not_confirmed():
    al = AutoLabelingState(model_configured=True, labeling_path="auto")
    ok, msg = al.can_run_auto_labeling(dataset_confirmed=False, dataset_name="")
    assert not ok
    assert msg


def test_can_run_auto_labeling_blocks_when_model_not_configured():
    al = AutoLabelingState(model_configured=False, labeling_path="auto")
    ok, msg = al.can_run_auto_labeling(dataset_confirmed=True, dataset_name="ds")
    assert not ok
    assert msg


def test_can_run_auto_labeling_blocks_when_path_is_manual():
    al = AutoLabelingState(model_configured=True, labeling_path="manual")
    ok, msg = al.can_run_auto_labeling(dataset_confirmed=True, dataset_name="ds")
    assert not ok
    assert msg


def test_can_run_auto_labeling_passes_on_auto_path_with_model():
    al = AutoLabelingState(model_configured=True, labeling_path="auto")
    ok, msg = al.can_run_auto_labeling(dataset_confirmed=True, dataset_name="ds")
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# AutoLabelingState.can_export_to_cvat
# ---------------------------------------------------------------------------

def test_can_export_to_cvat_without_predictions_always_passes():
    al = AutoLabelingState(auto_labeling_complete=False)
    ok, msg = al.can_export_to_cvat(with_predictions=False)
    assert ok
    assert msg == ""


def test_can_export_to_cvat_with_predictions_blocks_when_not_complete():
    al = AutoLabelingState(auto_labeling_complete=False)
    ok, msg = al.can_export_to_cvat(with_predictions=True)
    assert not ok
    assert msg


def test_can_export_to_cvat_with_predictions_passes_when_complete():
    al = AutoLabelingState(auto_labeling_complete=True)
    ok, msg = al.can_export_to_cvat(with_predictions=True)
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# AutoLabelingState.can_import_from_cvat
# ---------------------------------------------------------------------------

def test_can_import_from_cvat_blocks_when_no_task():
    al = AutoLabelingState(cvat_task_id=0)
    ok, msg = al.can_import_from_cvat()
    assert not ok
    assert msg


def test_can_import_from_cvat_passes_when_task_exists():
    al = AutoLabelingState(cvat_task_id=999)
    ok, msg = al.can_import_from_cvat()
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# AutoLabelingState.can_export_to_label_studio
# ---------------------------------------------------------------------------

def test_can_export_to_label_studio_without_predictions_always_passes():
    al = AutoLabelingState(auto_labeling_complete=False)
    ok, msg = al.can_export_to_label_studio(with_predictions=False)
    assert ok
    assert msg == ""


def test_can_export_to_label_studio_with_predictions_blocks_when_not_complete():
    al = AutoLabelingState(auto_labeling_complete=False)
    ok, msg = al.can_export_to_label_studio(with_predictions=True)
    assert not ok
    assert msg


def test_can_export_to_label_studio_with_predictions_passes_when_complete():
    al = AutoLabelingState(auto_labeling_complete=True)
    ok, msg = al.can_export_to_label_studio(with_predictions=True)
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# AutoLabelingState.can_import_from_label_studio
# ---------------------------------------------------------------------------

def test_can_import_from_label_studio_blocks_when_no_task_ids():
    al = AutoLabelingState(ls_task_ids=[])
    ok, msg = al.can_import_from_label_studio()
    assert not ok
    assert msg


def test_can_import_from_label_studio_passes_when_task_ids_present():
    al = AutoLabelingState(ls_task_ids=[101])
    ok, msg = al.can_import_from_label_studio()
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# ClassMappingState.can_run_class_mapping
# ---------------------------------------------------------------------------

def test_can_run_class_mapping_blocks_when_dataset_not_confirmed():
    cm = ClassMappingState(
        model_configured=True, source_dataset_set=True, target_dataset_set=True
    )
    ok, msg = cm.can_run_class_mapping(dataset_confirmed=False)
    assert not ok
    assert msg


def test_can_run_class_mapping_blocks_when_model_not_configured():
    cm = ClassMappingState(
        model_configured=False, source_dataset_set=True, target_dataset_set=True
    )
    ok, msg = cm.can_run_class_mapping(dataset_confirmed=True)
    assert not ok
    assert msg


def test_can_run_class_mapping_blocks_when_source_not_set():
    cm = ClassMappingState(
        model_configured=True, source_dataset_set=False, target_dataset_set=True
    )
    ok, msg = cm.can_run_class_mapping(dataset_confirmed=True)
    assert not ok
    assert msg


def test_can_run_class_mapping_blocks_when_target_not_set():
    cm = ClassMappingState(
        model_configured=True, source_dataset_set=True, target_dataset_set=False
    )
    ok, msg = cm.can_run_class_mapping(dataset_confirmed=True)
    assert not ok
    assert msg


def test_can_run_class_mapping_passes_when_all_set():
    cm = ClassMappingState(
        model_configured=True, source_dataset_set=True, target_dataset_set=True
    )
    ok, msg = cm.can_run_class_mapping(dataset_confirmed=True)
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# AnomalyDetectionState.can_run_anomaly_detection
# ---------------------------------------------------------------------------

def test_can_run_anomaly_detection_blocks_when_dataset_not_confirmed():
    ad = AnomalyDetectionState(model_configured=True, data_source_set=True)
    ok, msg = ad.can_run_anomaly_detection(dataset_confirmed=False)
    assert not ok
    assert msg


def test_can_run_anomaly_detection_blocks_when_model_not_configured():
    ad = AnomalyDetectionState(model_configured=False, data_source_set=True)
    ok, msg = ad.can_run_anomaly_detection(dataset_confirmed=True)
    assert not ok
    assert msg


def test_can_run_anomaly_detection_blocks_when_data_source_not_set():
    ad = AnomalyDetectionState(model_configured=True, data_source_set=False)
    ok, msg = ad.can_run_anomaly_detection(dataset_confirmed=True)
    assert not ok
    assert msg


def test_can_run_anomaly_detection_passes_when_all_set():
    ad = AnomalyDetectionState(model_configured=True, data_source_set=True)
    ok, msg = ad.can_run_anomaly_detection(dataset_confirmed=True)
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# EmbeddingSelectionState.can_run_embedding_selection
# ---------------------------------------------------------------------------

def test_can_run_embedding_selection_blocks_when_dataset_not_confirmed():
    es = EmbeddingSelectionState(model_configured=True)
    ok, msg = es.can_run_embedding_selection(dataset_confirmed=False)
    assert not ok
    assert msg


def test_can_run_embedding_selection_blocks_when_model_not_configured():
    es = EmbeddingSelectionState(model_configured=False)
    ok, msg = es.can_run_embedding_selection(dataset_confirmed=True)
    assert not ok
    assert msg


def test_can_run_embedding_selection_passes_when_all_set():
    es = EmbeddingSelectionState(model_configured=True)
    ok, msg = es.can_run_embedding_selection(dataset_confirmed=True)
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# ZeroShotAutoLabelingState.can_run_zero_shot
# ---------------------------------------------------------------------------

def test_can_run_zero_shot_blocks_when_dataset_not_confirmed():
    zs = ZeroShotAutoLabelingState(models_configured=True, classes_set=True)
    ok, msg = zs.can_run_zero_shot(dataset_confirmed=False)
    assert not ok
    assert msg


def test_can_run_zero_shot_blocks_when_models_not_configured():
    zs = ZeroShotAutoLabelingState(models_configured=False, classes_set=True)
    ok, msg = zs.can_run_zero_shot(dataset_confirmed=True)
    assert not ok
    assert msg


def test_can_run_zero_shot_blocks_when_classes_not_set():
    zs = ZeroShotAutoLabelingState(models_configured=True, classes_set=False)
    ok, msg = zs.can_run_zero_shot(dataset_confirmed=True)
    assert not ok
    assert msg


def test_can_run_zero_shot_passes_when_all_set():
    zs = ZeroShotAutoLabelingState(models_configured=True, classes_set=True)
    ok, msg = zs.can_run_zero_shot(dataset_confirmed=True)
    assert ok
    assert msg == ""


# ---------------------------------------------------------------------------
# EnsembleSelectionState.can_run_ensemble_selection
# ---------------------------------------------------------------------------

def test_can_run_ensemble_selection_blocks_when_dataset_not_confirmed():
    ens = EnsembleSelectionState(classes_set=True)
    ok, msg = ens.can_run_ensemble_selection(dataset_confirmed=False)
    assert not ok
    assert msg


def test_can_run_ensemble_selection_blocks_when_classes_not_set():
    ens = EnsembleSelectionState(classes_set=False)
    ok, msg = ens.can_run_ensemble_selection(dataset_confirmed=True)
    assert not ok
    assert msg


def test_can_run_ensemble_selection_passes_when_all_set():
    ens = EnsembleSelectionState(classes_set=True)
    ok, msg = ens.can_run_ensemble_selection(dataset_confirmed=True)
    assert ok
    assert msg == ""
