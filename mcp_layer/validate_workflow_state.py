# mcp_layer/validate_workflow_state.py
"""
Pydantic layer over WORKFLOW_STATE in config.py.

Provides:
  - Typed schema for session state (WorkflowState + per-workflow substates)
  - Typed tool input contracts validated in chat_pipeline.py before any MCP
    tool call, catching hallucinated fields and wrong types
  - can_run_* precondition methods that block run tools if required config is missing
  - can_export_to_cvat / can_import_from_cvat on AutoLabelingState
  - Workflow dependency rules (ensemble_selection requires zero_shot first)
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

# Workflows that require another workflow to have run first.
WORKFLOW_DEPENDENCIES: dict[str, list[str]] = {
    "ensemble_selection": ["auto_labeling_zero_shot"],
}


# Tool input contracts
# extra="forbid" rejects any field the LLM hallucinates that isn't in the schema.
# Validated in chat_pipeline._dispatch() before calling the MCP tool.

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


class SetLabelingBackendInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["cvat", "label_studio"]


class ConfigureAutoLabelingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Source is validated here as a Literal; model name validation is done inside
    # configure_auto_labeling() against list_model_sources_and_models(), keeping
    # the model list in one place (auto_labeling.py).
    selected_source: Literal[
        "ultralytics", "hf_models_objectdetection", "custom_codetr", "roboflow"
    ]
    selected_model: str = Field(min_length=1)


TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "set_auto_labeling_hyperparams": SetAutoLabelingHyperparamsInput,
    "set_anomaly_detection_hyperparams": SetAnomalyDetectionHyperparamsInput,
    "set_embedding_selection_params": SetEmbeddingSelectionParamsInput,
    "set_ensemble_selection_parameters": SetEnsembleSelectionParametersInput,
    "set_auto_labeling_zero_shot_threshold": SetZeroShotThresholdInput,
    "set_anomaly_detection_data_source": SetAnomalyDetectionDataSourceInput,
    "configure_auto_labeling": ConfigureAutoLabelingInput,
    "set_labeling_backend": SetLabelingBackendInput,
}


def validate_tool_input(fn_name: str, fn_args: dict) -> tuple[bool, str, dict]:
    """
    Validate tool arguments against the registered input model.

    Returns (ok, error_message, cleaned_args).
    On success, cleaned_args contains only non-None validated fields.
    On failure, error_message explains the problem and cleaned_args is unchanged.
    """
    model_cls = TOOL_INPUT_MODELS.get(fn_name)
    if model_cls is None:
        return True, "", fn_args  # no contract registered — pass through

    try:
        validated = model_cls.model_validate(fn_args)
        cleaned = {k: v for k, v in validated.model_dump().items() if v is not None}
        return True, "", cleaned
    except Exception as e:
        return False, f"Invalid arguments for {fn_name}: {e}", fn_args


# Per-workflow substates

class AutoLabelingState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labeling_path: Literal["manual", "auto", ""] = ""
    labeling_backend: Literal["cvat", "label_studio", ""] = "cvat"
    manual_classes: list[str] = []
    models_listed: bool = False         # True after list_model_sources_and_models is called
    model_configured: bool = False
    hyperparams_confirmed: bool = False
    auto_labeling_complete: bool = False
    cvat_task_id: int = 0           # 0 = not yet exported to CVAT
    ls_task_ids: list[int] = []     # empty = not yet exported to Label Studio
    labels_imported: bool = False

    def can_configure_auto_labeling(self) -> tuple[bool, str]:
        if not self.models_listed:
            return False, (
                "The available models must be listed before configuring. "
                "Please call list_model_sources_and_models first so the user "
                "can select from the actual available models."
            )
        return True, ""

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
        if self.labeling_path == "manual":
            return False, (
                "Auto-labeling cannot run on the manual labeling path. "
                "Please export to the annotation tool and annotate manually."
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

    def can_export_to_label_studio(self, with_predictions: bool) -> tuple[bool, str]:
        if with_predictions and not self.auto_labeling_complete:
            return False, (
                "Auto-labeling must complete before exporting predictions to Label Studio. "
                "Please run auto-labeling first."
            )
        return True, ""

    def can_import_from_label_studio(self) -> tuple[bool, str]:
        if not self.ls_task_ids:
            return False, (
                "The dataset must be exported to Label Studio before importing annotations. "
                "Please export to Label Studio first."
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
    data_source_set: bool = False   # location + rare_class confirmed

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

    # Shared fields across all workflows.
    workflow_name: VALID_WORKFLOW = ""
    dataset_name: str = ""
    dataset_confirmed: bool = False
    labeled_dataset_name: str = ""

    # Per-workflow substates. Only one is non-None at a time.
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
        """Return (ok, error) based on whether prerequisite workflows have been completed."""
        deps = WORKFLOW_DEPENDENCIES.get(workflow_name, [])
        missing = [d for d in deps if d not in completed_workflows]
        if missing:
            missing_readable = [d.replace("_", " ") for d in missing]
            return False, (
                f"The {workflow_name.replace('_', ' ')} workflow requires "
                f"{', '.join(missing_readable)} to be completed first."
            )
        return True, ""

    def valid_tool_names(self) -> set[str] | None:
        """
        Return the set of tool names valid for the current workflow step.

        Called by chat_server.filter_tools_for_state() before each llm.chat()
        so the LLM structurally cannot call out-of-sequence tools. Returns None
        when the state is unknown — falls back to the full tool list.

        The can_* methods on substates handle post-call argument validation
        (wrong shape, missing required fields). This method handles sequencing.
        """
        ALWAYS = {"send_reply", "switch_workflow", "reset_workflow_state"}

        if not self.workflow_name:
            return ALWAYS | {"select_workflow"}

        if not self.dataset_confirmed:
            # class_mapping skips dataset selection per the system prompt
            if self.workflow_name == "class_mapping":
                return self._class_mapping_tools(ALWAYS)
            return ALWAYS | {"set_selected_dataset", "list_datasets"}

        if self.workflow_name == "auto_labeling":
            return self._auto_labeling_tools(ALWAYS)
        if self.workflow_name == "class_mapping":
            return self._class_mapping_tools(ALWAYS)
        if self.workflow_name == "anomaly_detection":
            return self._anomaly_detection_tools(ALWAYS)
        if self.workflow_name == "embedding_selection":
            return self._embedding_selection_tools(ALWAYS)
        if self.workflow_name == "auto_labeling_zero_shot":
            return self._zero_shot_tools(ALWAYS)
        if self.workflow_name == "ensemble_selection":
            return self._ensemble_tools(ALWAYS)

        return None  # unknown workflow — no filtering, fail open

    def _auto_labeling_tools(self, ALWAYS: set[str]) -> set[str]:
        al = self.auto_labeling
        if not al or not al.labeling_path:
            return ALWAYS | {
                "list_model_sources_and_models",
                "export_to_cvat",
                "export_to_label_studio",
                "get_labeling_backend",
                "set_labeling_backend",
            }
        if al.labeling_path == "manual":
            if al.labels_imported:
                return ALWAYS | {"launch_voxel51_session"}
            if al.cvat_task_id > 0:
                return ALWAYS | {"import_from_cvat"}
            if al.ls_task_ids:
                return ALWAYS | {"import_from_label_studio"}
            return ALWAYS | {"export_to_cvat", "export_to_label_studio"}
        if al.labeling_path == "auto":
            if not al.models_listed:
                return ALWAYS | {"list_model_sources_and_models"}
            if not al.model_configured:
                return ALWAYS | {"configure_auto_labeling"}
            if not al.auto_labeling_complete:
                # run_auto_labeling is always present once the model is configured
                # so the user can skip hyperparam confirmation (defaults are valid).
                return ALWAYS | {"set_auto_labeling_hyperparams", "run_auto_labeling"}
            if al.labels_imported:
                return ALWAYS | {"launch_voxel51_session"}
            return ALWAYS | {"import_from_cvat", "import_from_label_studio"}
        return ALWAYS

    def _class_mapping_tools(self, ALWAYS: set[str]) -> set[str]:
        cm = self.class_mapping
        if not cm or not cm.model_configured:
            return ALWAYS | {"list_class_mapping_models", "configure_class_mapping_model"}
        if not cm.source_dataset_set:
            return ALWAYS | {
                "set_class_mapping_dataset_source",
                "set_selected_dataset",
                "launch_voxel51_session",
            }
        if not cm.target_dataset_set:
            return ALWAYS | {"set_class_mapping_dataset_target", "launch_voxel51_session"}
        if not cm.candidate_labels_set:
            return ALWAYS | {
                "set_class_mapping_candidate_labels",
                "launch_voxel51_session",
            }
        return ALWAYS | {"run_class_mapping", "launch_voxel51_session"}

    def _anomaly_detection_tools(self, ALWAYS: set[str]) -> set[str]:
        ad = self.anomaly_detection
        if not ad or not ad.model_configured:
            return ALWAYS | {
                "list_anomaly_detection_models",
                "configure_anomaly_detection_model",
                "launch_voxel51_session",
            }
        if not ad.data_source_set:
            return ALWAYS | {
                "set_anomaly_detection_data_source",
                "launch_voxel51_session",
            }
        return ALWAYS | {
            "set_anomaly_detection_hyperparams",
            "run_anomaly_detection",
            "launch_voxel51_session",
        }

    def _embedding_selection_tools(self, ALWAYS: set[str]) -> set[str]:
        es = self.embedding_selection
        if not es or not es.model_configured:
            return ALWAYS | {
                "list_embedding_selection_models",
                "configure_embedding_selection_model",
            }
        return ALWAYS | {"set_embedding_selection_params", "run_embedding_selection"}

    def _zero_shot_tools(self, ALWAYS: set[str]) -> set[str]:
        zs = self.auto_labeling_zero_shot
        if not zs or not zs.models_configured:
            return ALWAYS | {
                "list_zsal",
                "configure_auto_labeling_zero_shot_models",
            }
        if not zs.classes_set:
            return ALWAYS | {
                "set_auto_labeling_zero_shot_threshold",
                "set_auto_labeling_zero_shot_classes",
            }
        return ALWAYS | {
            "set_auto_labeling_zero_shot_threshold",
            "set_auto_labeling_zero_shot_classes",
            "run_zero_shot_auto_labeling",
        }

    def _ensemble_tools(self, ALWAYS: set[str]) -> set[str]:
        ens = self.ensemble_selection
        if not ens or not ens.classes_set:
            return ALWAYS | {
                "set_ensemble_selection_parameters",
                "set_ensemble_selection_classes",
            }
        return ALWAYS | {
            "set_ensemble_selection_parameters",
            "set_ensemble_selection_classes",
            "run_ensemble_selection",
        }

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
        Migrate from old flat WORKFLOW_STATE dict to the nested schema.

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

        # Backfill fields added after initial AutoLabelingState releases.
        al = raw.get("auto_labeling")
        if isinstance(al, dict):
            al.setdefault("labeling_backend", "cvat")
            al.setdefault("ls_task_ids", [])
            al.setdefault("manual_classes", [])
            al.setdefault("models_listed", False)

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