tools = [

        {
            "type": "function",
            "function": {
                "name": "send_reply",
                "description": (
                    "Use this to send a plain text reply to the user when no other tool is needed. "
                    "Call this instead of responding with text directly. "
                    "Parameters: 'message' (required) — the reply text; "
                    "'source' (optional) — set when your reply provides knowledge (explains, compares, "
                    "or recommends based on ML expertise or tool documentation); omit when your reply "
                    "drives the workflow (confirms an action, reports state, or presents options). "
                    "See the SOURCE TAG RULE in the system prompt for full classification criteria and examples."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The reply text to send to the user."
                        },
                        "source": {
                            "type": "string",
                            "description": (
                                "Source attribution — set when your reply provides knowledge "
                                "(explains, describes, compares, or recommends). "
                                "Omit when your reply drives the workflow (confirms, presents options, reports state)."
                            )
                        }
                    },
                    "required": ["message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "confirm_export",
                "description": (
                    "Call this ONLY after the user has explicitly confirmed they want to export "
                    "(e.g., responded 'yes', 'proceed', 'go ahead', 'sounds good'). "
                    "This records their consent and unlocks the export tool for this session. "
                    "After calling this, immediately call export_to_cvat or export_to_label_studio "
                    "with the classes from SESSION_STATE manual_classes. "
                    "Do NOT call this speculatively — only call it when the user has actually confirmed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "confirm_run",
                "description": (
                    "Call this ONLY after the user has explicitly confirmed they want to start auto-labeling "
                    "(e.g., responded 'yes', 'proceed', 'go ahead', 'start it', 'run it'). "
                    "This records their consent and unlocks the run_auto_labeling tool for this session. "
                    "After calling this, immediately call run_auto_labeling. "
                    "Do NOT call this speculatively — only call it when the user has actually confirmed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "select_workflow",
                "description": (
                    "Call this ONLY at the very start of a session when the user picks a workflow "
                    "for the first time and no workflow is currently active. "
                    "Do NOT call this if SESSION_STATE already shows a workflow — "
                    "use switch_workflow instead if the user wants to change to a different workflow."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_name": {
                            "type": "string",
                            "description": "The workflow name the user selected. Must come from the user's message — do not infer or guess."
                        }
                    },
                    "required": ["workflow_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_datasets",
                "description": "Lists all available datasets from datasets.yaml (plus fixed defaults).",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "switch_workflow",
                "description": (
                    "Call this in two cases: "
                    "(1) the user names a specific different workflow to switch to "
                    "(e.g. 'switch to class mapping', 'let me try anomaly detection') — pass that workflow name; OR "
                    "(2) the user wants to start over or restart from scratch within the current workflow — "
                    "pass the current workflow name to reset all state back to dataset selection. "
                    "Do NOT call this if the user says they want 'a different workflow' without naming one — "
                    "use send_reply to present the six workflow options and ask which one instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_name": {
                            "type": "string",
                            "description": "The exact workflow name the user stated. Must come from the user's message — do not infer or guess."
                        }
                    },
                    "required": ["workflow_name"]
                }
            }
        },
       {
            "type": "function",
            "function": {
                "name": "set_selected_dataset",
                "description": (
                    "REQUIRED: Update SELECTED_DATASET in config.py. "
                    "Call this immediately whenever the user provides a dataset name — "
                    "whether from ingestion or from the existing list. "
                    "No downstream tool (run_auto_labeling, export_to_cvat, etc.) works correctly without this."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "The name of the dataset to select."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "configure_auto_labeling",
                "description": (
                    "Enable the selected model source and model inside config.py for the auto_labeling workflow. "
                    "ACCURACY GUARD: only call this when the user has explicitly named a specific model "
                    "(e.g. 'use yolo12n', 'rfdetr_2xlarge', 'I want the DETR model'). "
                    "A user describing their use case ('I want to detect cars and pedestrians'), "
                    "asking what to use ('what model should I use?'), or saying 'sure' to a vague suggestion "
                    "is NOT an explicit model selection. In those cases, recommend with send_reply and ask "
                    "'Which model would you like to use?' — then wait for a specific answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_source": {
                            "type": "string",
                            "description": (
                                "The model source to enable. Derive it from the model name: "
                                "rfdetr_* → 'roboflow'; "
                                "yolo* (yolo11n, yolo12x, etc.) → 'ultralytics'; "
                                "facebook/*, microsoft/*, SenseTime/*, hustvl/*, jozhang97/*, Omnifact/* → 'hf_models_objectdetection'; "
                                "co_deformable_detr_*.py or co_dino_*.py → 'custom_codetr'."
                            )
                        },
                        "selected_model": {
                            "type": "string",
                            "description": "The exact model name or config file the user typed (e.g. 'rfdetr_2xlarge', 'yolo11n', 'facebook/detr-resnet-50')"
                        }
                    },
                    "required": ["selected_source", "selected_model"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_model_sources_and_models",
                "description": "Lists valid model sources and models for auto_labeling.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_auto_labeling_hyperparams",
                "description": (
                    "Update hyperparameters like mode, epochs, learning_rate, etc. for auto_labeling workflow. "
                    "ACCURACY GUARD — TWO valid triggers only: "
                    "(1) user directly instructs a change ('set epochs to 5', 'change learning rate to 0.001') — act immediately; "
                    "(2) user explicitly confirms a recommendation you already made ('yes apply those', 'go ahead', 'sounds good', 'yes') — act after that confirmation. "
                    "NOT a trigger — any request for advice or suggestions, regardless of phrasing: "
                    "'what do you recommend?', 'what would you suggest?', 'what's best for my use case?', 'any recommendations?'. "
                    "In those cases: give the recommendation with send_reply, ask 'Would you like me to apply these?', then wait. "
                    "Do NOT call this tool speculatively or before receiving explicit confirmation. "
                    "If the user accepts current defaults without changes, do NOT call this tool."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Pipeline mode(s): inference."
                        },
                        "epochs": {"type": "integer", "description": "Number of training epochs."},
                        "early_stop_patience": {"type": "integer", "description": "Patience for early stopping."},
                        "early_stop_threshold": {"type": "number", "description": "Improvement threshold for early stopping."},
                        "learning_rate": {"type": "number", "description": "Learning rate for the optimizer."},
                        "weight_decay": {"type": "number", "description": "Weight decay (L2 penalty)."},
                        "max_grad_norm": {"type": "number", "description": "Max norm for gradient clipping."}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_class_mapping_models",
                "description": "Lists zero-shot classification models available for class mapping.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "configure_class_mapping_model",
                "description": "Enables the selected model for class_mapping by commenting out all other zero-shot models.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_model": {
                            "type": "string",
                            "description": "The HuggingFace zero-shot classification model to use."
                        }
                    },
                    "required": ["selected_model"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_class_mapping_dataset_source",
                "description": "Updates the dataset_source field inside the class_mapping section of config.py.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_source": {
                            "type": "string",
                            "description": "The name of the dataset source to use ('fisheye8k', 'fisheye8k_mini')"
                        }
                    },
                    "required": ["dataset_source"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_class_mapping_dataset_target",
                "description": "Updates the dataset_target field inside the class_mapping section of config.py.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_target": {
                            "type": "string",
                            "description": "The name of the dataset target to use ('mcity_fisheye_2000', 'mcity_fisheye_2100')"
                        }
                    },
                    "required": ["dataset_target"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_class_mapping_candidate_labels",
                "description": "Updates the candidate_labels section in the class_mapping workflow. Supports one-to-many and one-to-one mappings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidate_labels": {
                            "type": "object",
                            "description": "Mapping from source class label to list of generalized target class names.",
                            "additionalProperties": {
                                "type": "array",
                                "items": { "type": "string" }
                            }
                        }
                    },
                    "required": ["candidate_labels"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_anomaly_detection_models",
                "description": "Lists Anomalib image models available for anomaly detection.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "configure_anomaly_detection_model",
                "description": "Enables the selected model for anomaly_detection by commenting out all other Anomalib models.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_model": {
                            "type": "string",
                            "description": "The Anomalib image model to use (e.g., Padim, Draem, EfficientAd, Cfa)."
                        }
                    },
                    "required": ["selected_model"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_embedding_selection_params",
                "description": "Update embedding selection parameters like representativeness, uniqueness, similarity, and neighbour count.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "compute_representativeness": {
                            "type": "number",
                            "description": "Value between 0 and 1 that balances global representativeness."
                        },
                        "compute_unique_images_greedy": {
                            "type": "number",
                            "description": "Value between 0 and 1 for greedy uniqueness."
                        },
                        "compute_unique_images_deterministic": {
                            "type": "number",
                            "description": "Value between 0 and 1 for deterministic uniqueness."
                        },
                        "compute_similar_images": {
                            "type": "number",
                            "description": "Value between 0 and 1 for how many similar images to retain."
                        },
                        "neighbour_count": {
                            "type": "integer",
                            "description": "Number of neighbors to consider in embedding space."
                        }
                    }
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "list_embedding_selection_models",
                "description": "Lists models available for embedding selection",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "configure_embedding_selection_model",
                "description": "Enables the selected model for embedding_selection by commenting out all other models.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_model": {
                            "type": "string",
                            "description": "The Embedding Selection model to use."
                        }
                    },
                    "required": ["selected_model"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_anomaly_detection_data_source",
                "description": "Update the location and rare_class used for anomaly detection in the data_preparation section.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The camera location (e.g., cam1, cam2) to set for anomaly detection."
                        },
                        "rare_class": {
                            "type": "string",
                            "description": "The class to be treated as rare or anomalous (e.g., Bus, Truck, Pedestrian)."
                        }
                    },
                    "required": ["location", "rare_class"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_anomaly_detection_hyperparams",
                "description": "Updates mode, epochs, and early_stop_patience in the anomaly_detection config.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Execution mode(s), like ['train'], ['inference'], or both."
                        },
                        "epochs": {
                            "type": "integer",
                            "description": "Number of training epochs."
                        },
                        "early_stop_patience": {
                            "type": "integer",
                            "description": "Patience for early stopping (in epochs)."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_zsal",
                "description": "Lists models available for auto_labeling_zero_shot > hf_models_zeroshot_objectdetection.",
                "parameters": {
                "type": "object",
                "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "configure_auto_labeling_zero_shot_models",
                "description": "Enables the selected models for auto_labeling_zero_shot by commenting out all other models.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_models": {
                            "type": "array",
                            "description": "List of zero-shot object detection models to use.",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["selected_models"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_auto_labeling_zero_shot_threshold",
                "description": "Sets the detection threshold in the auto_labeling_zero_shot workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "threshold": {
                        "type": "number",
                        "description": "Detection confidence threshold to use (e.g., 0.3)"
                        }
                    },
                    "required": ["threshold"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_auto_labeling_zero_shot_classes",
                "description": "Replaces the list of object classes to be detected in the auto_labeling_zero_shot workflow.",
                "parameters": {
                "type": "object",
                "properties": {
                        "object_classes": {
                        "type": "array",
                        "description": "List of object classes that the zero-shot model should detect.",
                        "items": {
                                "type": "string"
                            }
                        }
                    },
                "required": ["object_classes"]
                }
            }
        },
        {
        "type": "function",
        "function": {
            "name": "set_ensemble_selection_parameters",
            "description": "Update parameters for the ensemble selection workflow, including agreement threshold, IoU threshold, and maximum bounding box size.",
            "parameters": {
                "type": "object",
                    "properties": {
                        "agreement_threshold": {
                            "type": "integer",
                            "description": "Required. Minimum number of models that must agree on overlapping detections; must be >= 1 and <= number of zero-shot models used."
                        },
                        "iou_threshold": {
                            "type": "number",
                            "description": "Optional. Minimum Intersection-over-Union (IoU) for bounding boxes to be considered overlapping; must be between 0 and 1. Suggested default is 0.5."
                        },
                        "max_bbox_size": {
                            "type": "number",
                            "description": "Optional. Maximum relative area of bounding boxes (normalized to image size); must be between 0 and 1. Useful for filtering out overly large detections."
                        }
                    },
                    "required": ["agreement_threshold"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_ensemble_selection_classes",
                "description": "Updates the list of object classes to be used as positives in the ensemble_selection workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "positive_classes": {
                            "type": "array",
                            "description": "List of object class names to retain as positive detections in ensemble selection.",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["positive_classes"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "launch_voxel51_session",
                "description": "Launch the Voxel51 session for a specific dataset. Always pass dataset_name explicitly — never call without it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the FiftyOne dataset to visualize. After import_from_cvat, use the labeled name (e.g. 'custom_dataset15_labeled')."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "reset_workflow_state",
                "description": "Reset workflow selection state after a workflow is complete",
                "parameters": {
                "type": "object",
                "properties": {},
                "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_auto_labeling",
                "description": (
                    "Run main.py for auto_labeling workflow and stream logs in real time. "
                    "IRREVERSIBLE: training consumes time and compute and cannot be undone once started. "
                    "Only call this after the user has given an explicit run confirmation — "
                    "valid phrases: 'Run the workflow', 'Start training', 'Let's begin', or equivalent. "
                    "Do not call this based on inferred intent or because the workflow appears ready."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "export_to_cvat",
                "description": "Export a FiftyOne dataset to CVAT for annotation. Use ONLY for the Manual Labeling path with with_predictions=False. For Auto Generated Labeling, this tool is called automatically by the system after run_auto_labeling completes — do NOT call it yourself for Auto Generated Labeling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the FiftyOne dataset to export."
                        },
                        "with_predictions": {
                            "type": "boolean",
                            "description": "If true, exports model predictions for Auto Generated Labeling. If false, exports images only for Manual Labeling."
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Label names to pre-configure in CVAT for the Manual Labeling path (e.g. ['car', 'pedestrian', 'cyclist']). Only used when with_predictions=False. Ask the user for these before calling export."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "import_from_cvat",
                "description": "Download completed annotations from CVAT for a previously uploaded dataset and save as a new labeled FiftyOne dataset named <dataset_name>_labeled.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the original FiftyOne dataset that was uploaded to CVAT."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_labeling_backend",
                "description": "Check which annotation backends (CVAT, Label Studio) are configured in .env and return the active one. Call this when the backend has not yet been detected for the session (e.g., after a workflow switch mid-session).",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_labeling_backend",
                "description": "Set the active annotation backend for this session. Call this ONLY when the user directly names a backend as their choice ('CVAT', 'Label Studio', 'use CVAT', 'I prefer Label Studio'). A question ('what's the difference?', 'tell me more') is NOT a selection — answer it with send_reply and re-ask which they prefer. Do NOT call this if only one backend is configured — it is selected automatically.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "backend": {
                            "type": "string",
                            "enum": ["cvat", "label_studio"],
                            "description": "The annotation backend to use. 'cvat' for CVAT, 'label_studio' for Label Studio."
                        }
                    },
                    "required": ["backend"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_labeling_path",
                "description": (
                    "Call this IMMEDIATELY when the user names their labeling path at Step 3b. "
                    "Use 'manual' when the user says 'Manual Labeling', 'annotate myself', etc. "
                    "Use 'auto' when the user says 'Auto Generated Labeling', 'run a model', etc. "
                    "This MUST be called before any export or model tool — it unlocks the correct "
                    "downstream tools and persists the path so backend switches do not lose it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "enum": ["manual", "auto"],
                            "description": "'manual' for Manual Labeling, 'auto' for Auto Generated Labeling."
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "export_to_label_studio",
                "description": "Export a FiftyOne dataset to Label Studio for annotation. Use ONLY when the active backend is Label Studio. For Manual Labeling use with_predictions=False. For Auto Generated Labeling, this is called automatically after run_auto_labeling — do NOT call it yourself.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the FiftyOne dataset to export."
                        },
                        "with_predictions": {
                            "type": "boolean",
                            "description": "If true, attaches model predictions for Auto Generated Labeling. If false, uploads images only for Manual Labeling."
                        },
                        "classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Label names to pre-configure in Label Studio for the Manual Labeling path (e.g. ['car', 'pedestrian', 'cyclist']). Only used when with_predictions=False. Ask the user for these before calling export."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "import_from_label_studio",
                "description": "Download completed annotations from Label Studio for a previously exported dataset and save as a new FiftyOne dataset named <dataset_name>_labeled.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the original FiftyOne dataset that was exported to Label Studio."
                        }
                    },
                    "required": ["dataset_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_class_mapping",
                "description": "Run main.py for class_mapping workflow and stream logs in real time.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_anomaly_detection",
                "description": "Run main.py for anomaly_detection workflow and return the last 500 characters from combined output.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_embedding_selection",
                "description": "Run main.py for embedding_selection workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_zero_shot_auto_labeling",
                "description": "Run main.py for zero shot autolabeling workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "run_ensemble_selection",
                "description": "Run main.py for ensemble_selection workflow and stream logs in real time.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }

]