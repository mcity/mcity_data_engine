# mcp_layer/validate_workflow_state.py
"""
Pydantic layer over WORKFLOW_STATE in config.py.

Provides:
  - Typed schema for session state (WorkflowState + per-workflow substates)
  - Typed tool input contracts (ToolInput* models) — validated in chat_pipeline.py
    before any MCP tool call, catching hallucinated fields and wrong types
  - can_run_* precondition methods — block run tools if required config is missing
  - can_export_to_cvat / can_import_from_cvat on AutoLabelingState
  - Workflow dependency rules — ensemble_selection requires zero_shot first
  - Persistence: load() reads from config.py, save() writes back
  - Migration: handles old flat WORKFLOW_STATE dict format gracefully
"""

import ast
import importlib
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

import config.config as _cc

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.py"

WORKFLOW_STATE_DEFAULT = {
    "workflow_name": "",
    "dataset_name": "",
    "dataset_confirmed": False,
    "labeled_dataset_name": "",
    "auto_labeling": None,
    "class_mapping": None,
    "anomaly_detection": None,
    "embedding_selection": None,
    "auto_labeling_zero_shot": None,
    "ensemble_selection": None,
}

# Workflows that require another workflow to have run first
WORKFLOW_DEPENDENCIES: dict[str, list[str]] = {
    "ensemble_selection": ["auto_labeling_zero_shot"],
}


# Tool input contracts
# Validated in chat_pipeline._dispatch() before calling the MCP tool.
# extra="forbid" rejects any field the LLM hallucinates that isn't in schema.

class SetAutoLabelingHyperparamsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Optional[list[Literal["train", "inference"]]] = None
    epochs: Optional[int] = Field(None, gt=0, le=1000)
    early_stop_patience: Optional[int] = Field(None, gt=0)
    early_stop_threshold: Optional[float] = Field(None, ge=0)
    learning_rate: Optional[float] = Field(None, gt=0)
    weight_decay: Optional[float] = Field(None, ge=0)
    max_grad_norm: Optional[float] = Field(None, gt=0)


class SetAnomalyDetectionHyperparamsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Optional[list[Literal["train", "inference"]]] = None
    epochs: Optional[int] = Field(None, gt=0, le=1000)
    early_stop_patience: Optional[int] = Field(None, gt=0)


class SetEmbeddingSelectionParamsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    compute_representativeness: Optional[float] = Field(None, ge=0, le=1)
    compute_unique_images_greedy: Optional[float] = Field(None, ge=0, le=1)
    compute_unique_images_deterministic: Optional[float] = Field(None, ge=0, le=1)
    compute_similar_images: Optional[float] = Field(None, ge=0, le=1)
    neighbour_count: Optional[int] = Field(None, gt=0)


class SetEnsembleSelectionParametersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agreement_threshold: int = Field(ge=1)
    iou_threshold: Optional[float] = Field(None, ge=0, le=1)
    max_bbox_size: Optional[float] = Field(None, ge=0, le=1)


class SetZeroShotThresholdInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    threshold: float = Field(gt=0, le=1)


class SetAnomalyDetectionDataSourceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: str = Field(min_length=1)
    rare_class: str = Field(min_length=1)


# Map tool name -> input model for validation in _dispatch
TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "set_auto_labeling_hyperparams": SetAutoLabelingHyperparamsInput,
    "set_anomaly_detection_hyperparams": SetAnomalyDetectionHyperparamsInput,
    "set_embedding_selection_params": SetEmbeddingSelectionParamsInput,
    "set_ensemble_selection_parameters": SetEnsembleSelectionParametersInput,
    "set_auto_labeling_zero_shot_threshold": SetZeroShotThresholdInput,
    "set_anomaly_detection_data_source": SetAnomalyDetectionDataSourceInput,
}


def validate_tool_input(fn_name: str, fn_args: dict) -> tuple[bool, str, dict]:
    """
    Validate tool arguments against the registered input model.

    Returns:
        (ok, error_message, cleaned_args)
        - ok=True: cleaned_args has been coerced and validated
        - ok=False: error_message explains what's wrong, cleaned_args is unchanged
    """
    model_cls = TOOL_INPUT_MODELS.get(fn_name)
    if model_cls is None:
        return True, "", fn_args  # no contract registered — pass through

    try:
        validated = model_cls.model_validate(fn_args)
        # Return only non-None fields so MCP tool receives clean args
        cleaned = {k: v for k, v in validated.model_dump().items() if v is not None}
        return True, "", cleaned
    except Exception as e:
        return False, f"Invalid arguments for {fn_name}: {e}", fn_args


# Per-workflow substates

class AutoLabelingState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labeling_path: Literal["manual", "auto", ""] = ""
    model_configured: bool = False
    hyperparams_confirmed: bool = False
    auto_labeling_complete: bool = False
    cvat_task_id: int = 0       # 0 = not yet exported to CVAT
    labels_imported: bool = False

    def can_run_auto_labeling(
        self, dataset_confirmed: bool, dataset_name: str
    ) -> tuple[bool, str]:
        if not dataset_confirmed or not dataset_name:
            return False, (
                f"No dataset has been confirmed for this session. "
                f"Config currently points to '{dataset_name or 'none'}'. "
                f"Please confirm the correct dataset name and call "
                f"set_selected_dataset first."
            )
        if not self.model_configured:
            return False, (
                "A model source and model must be configured before running "
                "auto-labeling. Please select a model first."
            )
        if not self.hyperparams_confirmed:
            return False, (
                "Hyperparameters must be confirmed before running auto-labeling. "
                "Please confirm or update the hyperparameters first."
            )
        return True, ""

    def can_export_to_cvat(self, with_predictions: bool) -> tuple[bool, str]:
        if with_predictions and not self.auto_labeling_complete:
            return False, (
                "Auto-labeling must complete before exporting predictions to CVAT. "
                "Please run auto-labeling first."
            )
        return True, ""

    def can_import_from_cvat(self) -> tuple[bool, str]:
        if self.cvat_task_id == 0:
            return False, (
                "The dataset must be exported to CVAT before importing annotations. "
                "Please export to CVAT first."
            )
        return True, ""


class ClassMappingState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_configured: bool = False
    source_dataset_set: bool = False
    target_dataset_set: bool = False
    candidate_labels_set: bool = False

    def can_run_class_mapping(self, dataset_confirmed: bool) -> tuple[bool, str]:
        if not dataset_confirmed:
            return False, "Dataset must be confirmed before running class mapping."
        if not self.model_configured:
            return False, "A model must be configured before running class mapping."
        if not self.source_dataset_set:
            return False, "Source dataset must be set before running class mapping."
        if not self.target_dataset_set:
            return False, "Target dataset must be set before running class mapping."
        return True, ""


class AnomalyDetectionState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_configured: bool = False
    data_source_set: bool = False   # location + rare_class

    def can_run_anomaly_detection(self, dataset_confirmed: bool) -> tuple[bool, str]:
        if not dataset_confirmed:
            return False, "Dataset must be confirmed before running anomaly detection."
        if not self.model_configured:
            return False, "A model must be configured before running anomaly detection."
        if not self.data_source_set:
            return False, (
                "Camera location and rare class must be set before running "
                "anomaly detection."
            )
        return True, ""


class EmbeddingSelectionState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_configured: bool = False

    def can_run_embedding_selection(self, dataset_confirmed: bool) -> tuple[bool, str]:
        if not dataset_confirmed:
            return False, "Dataset must be confirmed before running embedding selection."
        if not self.model_configured:
            return False, "A model must be configured before running embedding selection."
        return True, ""


class ZeroShotAutoLabelingState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models_configured: bool = False
    classes_set: bool = False

    def can_run_zero_shot(self, dataset_confirmed: bool) -> tuple[bool, str]:
        if not dataset_confirmed:
            return False, "Dataset must be confirmed before running zero-shot auto-labeling."
        if not self.models_configured:
            return False, "Zero-shot models must be configured before running."
        if not self.classes_set:
            return False, "Object classes must be set before running zero-shot auto-labeling."
        return True, ""


class EnsembleSelectionState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    classes_set: bool = False

    def can_run_ensemble_selection(self, dataset_confirmed: bool) -> tuple[bool, str]:
        if not dataset_confirmed:
            return False, "Dataset must be confirmed before running ensemble selection."
        if not self.classes_set:
            return False, "Positive classes must be set before running ensemble selection."
        return True, ""


# Top-level WorkflowState

VALID_WORKFLOW = Literal[
    "auto_labeling",
    "class_mapping",
    "anomaly_detection",
    "embedding_selection",
    "auto_labeling_zero_shot",
    "ensemble_selection",
    ""
]


class WorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Shared fields — apply to all workflows
    workflow_name: VALID_WORKFLOW = ""
    dataset_name: str = ""
    dataset_confirmed: bool = False
    labeled_dataset_name: str = ""

    # Per-workflow substates — only one is non-None at a time
    auto_labeling: Optional[AutoLabelingState] = None
    class_mapping: Optional[ClassMappingState] = None
    anomaly_detection: Optional[AnomalyDetectionState] = None
    embedding_selection: Optional[EmbeddingSelectionState] = None
    auto_labeling_zero_shot: Optional[ZeroShotAutoLabelingState] = None
    ensemble_selection: Optional[EnsembleSelectionState] = None

    @model_validator(mode="after")
    def dataset_confirmed_requires_name(self) -> "WorkflowState":
        if self.dataset_confirmed and not self.dataset_name:
            raise ValueError(
                "dataset_confirmed cannot be True when dataset_name is empty."
            )
        return self

    def can_confirm_dataset(self) -> tuple[bool, str]:
        if not self.workflow_name:
            return False, (
                "A workflow must be selected before confirming a dataset. "
                "Please select a workflow first."
            )
        return True, ""

    def check_workflow_dependencies(
        self, workflow_name: str, completed_workflows: list[str]
    ) -> tuple[bool, str]:
        """
        Check that required prerequisite workflows have been completed.
        completed_workflows: list of workflow names the user has run this session.
        """
        deps = WORKFLOW_DEPENDENCIES.get(workflow_name, [])
        missing = [d for d in deps if d not in completed_workflows]
        if missing:
            missing_readable = [d.replace("_", " ") for d in missing]
            return False, (
                f"The {workflow_name.replace('_', ' ')} workflow requires "
                f"{', '.join(missing_readable)} to be completed first."
            )
        return True, ""

    @classmethod
    def load(cls) -> "WorkflowState":
        """Read WORKFLOW_STATE from config.py and return a validated WorkflowState."""
        try:
            importlib.reload(_cc)
            raw = dict(_cc.WORKFLOW_STATE)
            raw = cls._migrate(raw)
            return cls.model_validate(raw)
        except Exception as e:
            logging.warning(
                f"[STATE] Failed to load WorkflowState: {e} — using defaults"
            )
            return cls()

    @classmethod
    def _migrate(cls, raw: dict) -> dict:
        """
        Migrate from old flat WORKFLOW_STATE dict to new nested schema.
        Handles:
          - workflow_name: None -> ""
          - auto_labeling_complete, cvat_task_id: top-level -> auto_labeling subdict
          - Unknown keys dropped cleanly
        """
        if raw.get("workflow_name") is None:
            raw["workflow_name"] = ""

        old_al_fields = {
            "auto_labeling_complete": "auto_labeling_complete",
            "cvat_task_id": "cvat_task_id",
        }
        wf = raw.get("workflow_name", "")
        if wf == "auto_labeling" and "auto_labeling" not in raw:
            substate = {}
            for old_key, new_key in old_al_fields.items():
                if old_key in raw:
                    val = raw.pop(old_key)
                    if new_key == "cvat_task_id" and val is None:
                        val = 0
                    substate[new_key] = val
            if substate:
                raw["auto_labeling"] = substate
        else:
            for old_key in old_al_fields:
                raw.pop(old_key, None)

        known = {
            "workflow_name", "dataset_name", "dataset_confirmed",
            "labeled_dataset_name", "auto_labeling", "class_mapping",
            "anomaly_detection", "embedding_selection",
            "auto_labeling_zero_shot", "ensemble_selection",
        }
        for key in list(raw.keys()):
            if key not in known:
                raw.pop(key)

        return raw

    def save(self) -> None:
        """Write current state back to config.py as WORKFLOW_STATE dict."""
        try:
            src = CONFIG_PATH.read_text()
            tree = ast.parse(src)
            lines = src.splitlines()
            state_dict = self.model_dump()
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "WORKFLOW_STATE"
                        ):
                            start = node.lineno - 1
                            end = node.end_lineno
                            lines[start:end] = [
                                f"WORKFLOW_STATE = {repr(state_dict)}"
                            ]
                            CONFIG_PATH.write_text("\n".join(lines) + "\n")
                            return
        except Exception as e:
            logging.warning(f"[STATE] Failed to save WorkflowState: {e}")

    @classmethod
    def reset(cls) -> "WorkflowState":
        """Return a default WorkflowState and persist it."""
        fresh = cls()
        fresh.save()
        return fresh

    def reset_for_workflow(self, workflow_name: str) -> "WorkflowState":
        """Full reset for a new or switched workflow."""
        fresh = WorkflowState(workflow_name=workflow_name)
        substate_map = {
            "auto_labeling": ("auto_labeling", AutoLabelingState),
            "class_mapping": ("class_mapping", ClassMappingState),
            "anomaly_detection": ("anomaly_detection", AnomalyDetectionState),
            "embedding_selection": ("embedding_selection", EmbeddingSelectionState),
            "auto_labeling_zero_shot": ("auto_labeling_zero_shot", ZeroShotAutoLabelingState),
            "ensemble_selection": ("ensemble_selection", EnsembleSelectionState),
        }
        if workflow_name in substate_map:
            field, klass = substate_map[workflow_name]
            setattr(fresh, field, klass())
        fresh.save()
        return fresh

    def current_substate(self):
        """Return the active workflow substate, or None."""
        return getattr(self, self.workflow_name, None) if self.workflow_name else None