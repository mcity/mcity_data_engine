tools = [

        {
            "type": "function",
            "function": {
                "name": "select_workflow",
                "description": "Set the selected workflow (auto_labeling or class_mapping) in the config file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_name": {
                            "type": "string",
                            "description": "The name of the workflow to activate (auto_labeling or class_mapping)"
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
                "description": "Switch to a new workflow by updating SELECTED_WORKFLOW in config.py and resetting dataset/parameter selections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_name": {
                            "type": "string",
                            "description": "The name of the workflow to switch to (e.g., auto_labeling, class_mapping, anomaly_detection, etc.)"
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
                "description": "Update the SELECTED_DATASET field in config.py to choose which dataset to use.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "The name of the dataset (e.g., 'fisheye8k', 'fisheye8k_mini')"
                        },
                        "n_samples": {
                            "type": ["integer", "null"],
                            "description": "Optional number of samples to use (use null for full dataset)"
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
                "description": "Enable the selected model source and model inside config.py for the auto_labeling workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_source": {
                            "type": "string",
                            "description": "The model source to enable (ultralytics, hf_models_objectdetection, or custom_codetr)"
                        },
                        "selected_model": {
                            "type": "string",
                            "description": "The specific model or config to enable within the selected source"
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
                "description": "Update hyperparameters like mode, epochs, learning_rate, etc. for auto_labeling workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Pipeline mode(s): train, inference, or both."
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
                            "description": "Mapping from generalized target class to list of source class labels.",
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
                            "description": "Required. Minimum number of models that must agree on overlapping detections; must be ≥ 1 and ≤ number of zero-shot models used."
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
                "description": "Launch the Voxel51 session to visualize workflow results.",
                "parameters": {
                    "type": "object",
                    "properties": {}
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
                "description": "Run main.py for auto_labeling workflow and stream logs in real time.",
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
                "description": "Export a FiftyOne dataset to CVAT for annotation. Use ONLY for manual labeling path with with_predictions=False. For auto-labeling, this tool is called automatically by the system after run_auto_labeling completes — do NOT call it yourself for auto-labeling.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "description": "Name of the FiftyOne dataset to export."
                        },
                        "with_predictions": {
                            "type": "boolean",
                            "description": "If true, exports auto-labeling predictions. If false, exports images only for manual annotation."
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
