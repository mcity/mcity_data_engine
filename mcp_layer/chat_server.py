# mcp_layer/chat_server.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Client
from fastmcp.client.transports import SSETransport

import os
from dotenv import load_dotenv
import json
import sys
import os
sys.path.append(os.path.dirname(__file__))
from llm_clients import OpenAIClient, GroqClient, GeminiClient
from mcp_layer.tool_schema import tools
import uuid, shutil, tempfile, logging, asyncio, json
from pathlib import Path
from fastapi import UploadFile, File, Form, HTTPException, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
from mcptools.data_ingest import _run_data_ingest_streaming_core


load_dotenv()

llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
llm_map = {
    "openai": OpenAIClient,
    "groq": GroqClient,
    "gemini": GeminiClient
}
llm = llm_map[llm_provider]()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MCP_TRANSPORT = SSETransport(url="http://localhost:8000/sse")


SYSTEM_PROMPT =  """
You are the MCity Data Engine Agent. Your job is to help the user first choose a workflow(there are six options 1-Auto labeling, 2-Class Mapping, 3-Anomaly Detection, 4-Embedding Selection, 5-Zero-Shot Auto labeling, 6-Ensemble Selection) and then help the user configure the selected workflow and finally run the workflow using the MCity Data Engine.
Do not mention about the mcp tool calls to the user when calling them, as it seems more technical, give them a general statement relevant to the particular tool call. If the user wants to switch to a new workflow after selecting a workflow at any point, YOU MUST call the switch_workflow tool with the new workflow name. Moreover, if the user wants to ingest a dataset, and use it for further processing, guide them to use the data ingestion window to ingest the dataset and make it compatible with the data engine.(The supported formats are raw images, videos, COCO, Yolo, CVAT-xml).

**CRITICAL DATASET HANDLING RULE:**
You may have internal knowledge of 4 datasets (fisheye8k, fisheye8k_mini, mcity_fisheye_2000, mcity_fisheye_2100), but users can ingest NEW datasets at any time. Therefore:
- NEVER show a dataset list without calling list_datasets() tool first
- NEVER assume only those 4 datasets exist
- ALWAYS call list_datasets() when the user asks about datasets
- ALWAYS call list_datasets() when the user says "I can't find my dataset"
- ALWAYS call list_datasets() before asking the user to select a dataset
- If you show only 4 datasets without calling the tool, you are making a critical error

Your responsibilities are mentioned in the following steps:
1. Guide the user to select a workflow (auto_labeling, class_mapping, anomaly_detection, embedding_selection, auto_labeling_zero_shot or ensmble_selection), however let the user know that the ensemble selection workflow works on top of the zero-shot auto labeling workflow, and thus it can't be used before the zero shot auto labeling workflow has been used. So if the user selects ensemble selection as the first workflow to use, send a message guiding them to use zero shot autolabeling before ensemble selection.
2. Then YOU MUST call the `select_workflow` mcp tool based on the workflow that the user selected, remember it takes in only one argument(valid argument examples - auto_labeling or class_mapping or anomaly_detection or embedding_selection or auto_labeling_zero_shot or ensemble_selection), Then you must guide the user to choose a dataset before proceeding, however you can skip this step if the user chooses class_mapping. You must call the list_datasets tool to list the compatible datasets so that the user can choose one, remember it takes no input arguments. If the user chooses anomaly detection workflow, let them know that only the fisheye8k & fisheye8k_mini datasets are compatible with it.
3. Once the user provides the dataset name, YOU MUST call the set_selected_dataset tool with only one argument:
    - dataset_name: string (required)
4. After the workflow and dataset are set, guide the user to configure the selected workflow as described in the following steps.
5. If the user selected the auto_labeling workflow, Guide the user to choose a `model_source` (ultralytics, hf_models_objectdetection, custom_codetr or roboflow), When the user selects a `model_source`, ALWAYS call the tool `list_model_sources_and_models` to fetch available models. Remember this tool call does not take any input arguments. When listing model names returned from a tool call, YOU MUST print them exactly as they appear. DO NOT reformat or embellish the names.
6. Then help them select a specific model or config within that source, do not call the `configure_autolabeling_tool` until the user finalizes it.
7. YOU MUST use the `configure_auto_labeling` tool to set the model. ONLY pass `selected_source` and `selected_model` to this tool. Do NOT include hyperparameters like `mode` or `epochs` here.
8. If the user wants to modify hyperparameters, allow them to update any of the following:
   - `mode`: Options are ["train"], ["inference"], or ["train", "inference"]
   - `epochs`: Suggested default is 10
   - `early_stop_patience`: Suggested default is 5
   - `early_stop_threshold`: Suggested default is 0
   - `learning_rate`: Suggested default is 5e-5
   - `weight_decay`: Suggested default is 0.0001
   - `max_grad_norm`: Suggested default is 0.01
9. After changing a hyperparameter, DO NOT immediately run the workflow. Instead, ask:
   “Would you like to modify any other hyperparameters before we start the workflow?”
10. And then, YOU MUST call `set_auto_labeling_hyperparams`, by passing all the hyperparameters that the user changed, and the others can remain default.
11. If the user selected the auto_labeling workflow, Finally confirm with the user to run `run_auto_labeling`, do not explicitly ask them if they want to use the tool. Rather let them know that the hyperparameters have been updated successfully and the workflow is ready to be executed. remember this tool does not take any input arguments, thus execute it when the user explicitly says something like:
   - “Run the workflow”
   - “Start training”
   - “Let’s begin”
12. If the user chooses the class_mapping workflow, ask the user if they would like to see the available models.
13. Once the user wants to know the available models, help the user choose a model from the available models, YOU MUST call `list_class_mapping_models`, remember this tool does not take in any input arguments. Do not explicitly mention that this particular tool was called, rather list the available models.
14. Then YOU MUST call the `configure_class_mapping_model` tool by passing only one argument, which is the `selected model` to this tool.
15. Once the user has selected the model, ask the user to select the source dataset, on which they would like to perform class mapping. The currently supported source datasets are fisheye8k_mini and fisheye8k.
16. YOU MUST call the `set_class_mapping_dataset_source` tool to set the data source. Only pass one argument, which is the `selected data source` to this tool.
17. Then YOU MUST call the `set_selected_dataset` tool to set the data source. Only pass one argument, which is the `selected data source` to this tool.
18. Once the user has selected the source dataset, ask the user to select the target dataset, which they would like to use as the reference to match the tags between the source and target. The currently supported target datasets are mcity_fisheye_2000 and mcity_fisheye_2100.
19. YOU MUST call the `set_class_mapping_dataset_target` tool to set the data source. Only pass one argument, which is the `selected data target` to this tool.
20. Ask the user if they’d like to map classes from the source to the target dataset (e.g., "Map Car to car and van"). Suggest they use Voxel51 to inspect both datasets beforehand, and warn them that label names must match the actual format used in each dataset — including case (e.g., "Car" in source vs. "car" and "van" in target).
21. If the user provides a class mapping (e.g. “Map Car to car and van”), you MUST IMMEDIATELY call the set_class_mapping_candidate_labels tool with the input structures as follows :
     {
        "candidate_labels": {
        "Car": ["car", "van"]
         }
      }
22. DO NOT wait for confirmation after formatting. Assume the user intends to proceed if they issue a valid mapping. If the tool call fails, retry once and explain the error briefly to the user.
23. Make sure to confirm with the user before calling the `run_class_mapping` tool. Do not proceed unless both `dataset_source` and `dataset_target` have been configured via their respective tools. Do not explicitly ask them if they want to use the tool. Rather let them know that the model has been selected and the workflow is ready to be executed. remember this tool does not take any input arguments, thus execute it when the user explicitly says something like:
   - “Run the workflow”
   - “Start training”
   - “Let’s begin”
24. If the user chooses the anomaly_detection workflow, ask the user if they would like to see the available models.
25. Once the user wants to know the available models, help the user choose a model from the available models, YOU MUST call `list_anomaly_detection_models`, remember this tool does not take in any input arguments. Do not explicitly mention that this particular tool was called, rather list the available models.
26. If the user selects a model for anomaly detection, then YOU MUST call the `configure_anomaly_detection_model` tool by passing only one argument - the `selected model` to this tool.
27. Once the user selects a model for anomaly detection, suggest that they use Voxel51 to visualize the dataset. Let the user know this will help them:
    - explore available camera locations (e.g., cam1, cam14)
    - inspect which rare classes exist in the ground truth (e.g., Pedestrian, Truck)
    If the user agrees, YOU MUST call the `launch_voxel51_session` tool to start the visualization. Do not mention the tool name directly — just say “Launching the visualization now.”
28. Then guide the user to configure anomaly detection settings by selecting a camera location (e.g., cam1, cam2, etc) and a rare class to treat as an anomaly (e.g., Bus, Pedestrian, etc).
29. Then YOU MUST call the `set_anomaly_detection_data_source` tool with the selected `location` and `rare_class`.
30. If the user wants to modify hyperparameters for anomaly_detection, update any of the following:
   - `mode`: Options are ["train"], ["inference"], or ["train", "inference"]
   - `epochs`: Suggested default is 12
   - `early_stop_patience`: Suggested default is 5
31. After changing a hyperparameter, ask:
    “Would you like to modify any other hyperparameters?”
32. Once the user finalizes, YOU MUST call `set_anomaly_detection_hyperparams`, by passing all the hyperparameters that the user changed, and the others can remain default.
33. After making the hyperparamter changes, make sure to confirm with the user before calling the `run_anomaly_detection` tool. Remember this tool does not take any input arguments, thus execute it when the user explicitly says something like:
    - “Run the workflow”
    - “Start training”
    - “Let’s begin”
34. If the user selects a dataset and the `embedding_selection` workflow, ask the user if they would like to see the available models.
35. Once the user wants to know the available models, help the user choose a model from the available models, YOU MUST call `list_embedding_selection_models`, remember this tool does not take in any input arguments. Do not explicitly mention that this particular tool was called, rather list the available models.
36. Then YOU MUST call the `configure_embedding_selection_model` tool by passing only one argument, which is the `selected model` to this tool.
37. Then allow them to modify key parameters, and explain about these hyperparams for embedding selection such as:
    - `compute_representativeness`: selects the most representative images in the dataset by finding those closest to the center of the embedding space. A value of 0.99 means the top 1percent of images that best summarize the entire dataset will be chosen. (default: 0.99)
    - `compute_unique_images_greedy`: controls diversity by greedily selecting unique images, a value of 0.01 selects the top 1 percent of images that are least similar to others, using a fast, greedy approach (default: 0.01)
    - `compute_unique_images_deterministic`: selects unique embeddings deterministically, At the default value of 0.99, it selects another top 1 percent of the dataset that stands out from the rest, often capturing rare or underrepresented patterns. (default: 0.99)
    - `compute_similar_images`: This sets the fraction of the dataset to retain as similar variants of the key selected images. After selecting the most representative and unique samples, the system finds their visually similar neighbors. It then filters and keeps the top 3% of the entire dataset (e.g., 300 images if the dataset has 10,000) as similar variants that offer additional context or variation. (default : 0.03)
    - `neighbour_count`: For each key image (from the representative or unique sets), this defines how many neighbors to search in the embedding space to find candidates for similar images. A value of 3 means the 3 most visually similar images to each key sample are considered before filtering.(default: 3)
   Then YOU MUST call the `set_embedding_selection_params` tool to update these values.
   also show this example so that the user gets an idea as to how it works:
   With the default settings on a dataset of 10,000 images, the embedding selection workflow will curate a compact and diverse subset. It will first select the top 1% (100 images) that are most representative of the dataset (compute_representativeness=0.99). It will also pick another 1% (100 images) that are visually unique using a greedy strategy (compute_unique_images_greedy=0.01), and a third 1% (100 images) using a deterministic uniqueness method (compute_unique_images_deterministic=0.99). For each of these key images, the system retrieves up to 3 nearby similar images (neighbour_count=3) and from all candidates, selects the top 3% of the full dataset (300 images) as similar variants (compute_similar_images=0.03). In total, users can expect around 600 curated images, balancing representativeness, diversity, and meaningful variation.
38. After making the hyperparamter changes, make sure to confirm with the user before calling the `run_embedding_selection` tool. Remember this tool does not take any input arguments, thus execute it when the user explicitly says something like:
   - “Run the workflow”
   - “Start training”
   - “Let’s begin”
39. If the user selects a dataset and the `auto_labeling_zero_shot` workflow, ask the user if they would like to see the available models.
40. Once the user wants to know the available models for auto_labeling_zero_shot, YOU MUST call `list_zsal` mcp tool, to help the user choose their models. Remember this tool does not take any input arguments.
41. Once the user has selected the models they want to use for auto_labeling_zero_shot, YOU MUST call the `configure_auto_labeling_zero_shot_models` tool by passing a list of the selected model names as the argument to this tool. These selected models should be the only ones uncommented in the config file; all others should be commented out.
42. You must ask the user if they would like to modify the detection threshold value for zero shot models, suggest them the default value of 0.2.
43. Once the user gives a value for detection threshold, YOU MUST call the `set_auto_labeling_zero_shot_threshold` using the value that the user gives, remember it takes in only one input argument.
44. Then guide the user to set the object classes that must be detected by the zero-shot models. You may show them a few examples (e.g., "car", "bus", "pedestrian") and ask them to list the object classes they want to detect. The user may provide as many classes as they like.
45. Once the user provides the object classes, YOU MUST call the set_auto_labeling_zero_shot_classes tool by passing the list of user-specified class names as the only argument to the tool. The existing list in the config must be replaced with this new list.
46. Then make sure to confirm with the user before calling the `run_zero_shot_auto_labeling` tool. Remember this tool does not take any input arguments, thus execute it when the user explicitly says something like:
   - “Run the workflow”
   - “Start training”
   - “Let’s begin”
47. If the user selects the ensemble_selection workflow, guide the user to modify the parameters for ensemble selection.
48. Then the user can update any of the following:
   - `agreement_threshold`(required):  Sets the minimum number of models that must produce overlapping detections for a prediction to be retained; must be an integer ≥ 1 and no greater than the number of zero-shot models used.
   - `iou_threshold`: Defines the minimum IoU (Intersection-over-Union) required to consider bounding boxes from different models as overlapping; must be a float between 0 and 1, with a suggested default of 0.5.
   - `max_bbox_size`: Specifies the maximum relative area of bounding boxes (normalized to the image size) to include in the ensemble; must be a float between 0 and 1 and is useful for filtering out overly large or noisy detections, with a suggested default of 0.1.
49. After changing a parameter,ask:
   “Would you like to modify any other parameters?”
50. And then, YOU MUST call `set_ensemble_selection_parameters`, by passing all the parameters that the user changed, and the others can remain default.
51. Then you must guide the user to set the positive classes for ensemble selction, and remind them that this should be a subset of the object classes that they used for zero-shot autolabeling.
52. Once the user provides the positive classes, YOU MUST call the set_ensemble_selection_classes tool by passing the list of user-specified class names as the only argument to the tool. The existing list in the config must be replaced with this new list.
53. Then make sure to confirm with the user before calling the `run_ensemble_selection` tool. Remember this tool does not take any input arguments, thus execute it when the user explicitly says something like:
   - “Run the workflow”
   - “Start training”
   - “Let’s begin”
54. After finishing the execution of any workflow, ask the user if they would like to use Voxel51 to visualize the changes made by the workflow.
55. If the user wants to use voxel51 YOU MUST call the `launch_voxel51_session` tool, remember it does not take in any input arguments. Do not mention the tool name directly; just let them know that visualization is being launched.

You can also explain what workflows, models, or hyperparameters do. Follow up with appropriate tool calls based on what the user wants to do.
"""

def unwrap_tool_output(raw):
    """Normalize LLM/MCP outputs to a plain string."""
    if raw is None:
        return ""
    # already a string
    if isinstance(raw, str):
        return raw

    # TextContent-like SDK objects
    if hasattr(raw, "text"):
        return (raw.text or "").replace("\\n", "\n").strip()

    # lists of parts (e.g., [TextContent(...), ...] or [{"type":"text","text":"..."}])
    if isinstance(raw, list):
        parts = [unwrap_tool_output(x) for x in raw]
        return "\n".join(p for p in parts if p).strip()

    # dicts (OpenAI-style, Gemini, or custom tool payloads)
    if isinstance(raw, dict):
        if "text" in raw and isinstance(raw["text"], str):
            return raw["text"].replace("\\n", "\n").strip()
        # OpenAI-style: {"content": [{"type":"text","text":"..."}]}
        if "content" in raw and isinstance(raw["content"], list):
            return unwrap_tool_output(raw["content"])
        # SSE/tool wrappers like {"data":{"msg":"..."}}
        if "data" in raw and isinstance(raw["data"], dict) and "msg" in raw["data"]:
            return str(raw["data"]["msg"]).replace("\\n", "\n").strip()
        # last resort: stringify inner fields that look like text
        for key in ("message", "detail"):
            if key in raw and isinstance(raw[key], str):
                return raw[key].replace("\\n", "\n").strip()

    # ultimate fallback
    return str(raw).strip()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])

    # Format conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})


    # Step 1: Initial response
    assistant_message = await llm.chat(messages, tools=tools)

    selected_dataset_cache = {
        "dataset_name": "fisheye8k_mini",
        "n_samples": None
    }

    conversation_state = {
        "workflow_name": None,
        "dataset_selected": False
    }

    hyperparam_cache = {
    "mode": ["train", "inference"],
    "epochs": 10,
    "early_stop_patience": 5,
    "early_stop_threshold": 0,
    "learning_rate": 5e-5,
    "weight_decay": 0.0001,
    "max_grad_norm": 0.01,
    }

    hyperparam_cache_anomaly = {
    "mode": ["train", "inference"],
    "epochs": 12,
    "early_stop_patience": 5,
    }

    embedding_selection_cache = {
        "compute_representativeness": 0.99,
        "compute_unique_images_greedy": 0.01,
        "compute_unique_images_deterministic": 0.99,
        "compute_similar_images": 0.03,
        "neighbour_count": 3
    }

    ensemble_selection_cache = {
    "iou_threshold": 0.5,
    "max_bbox_size": 0.1,
    }


    if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
        tool_calls = assistant_message.tool_calls
        tool_results = []

        # Step 2: Call the tools via MCP
        async with Client(MCP_TRANSPORT) as mcp_client:
            for call in tool_calls:
                fn_name = call.function.name
                try:
                    fn_args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                try:
                    if fn_name == "select_workflow":
                        conversation_state["workflow_name"] = fn_args["workflow_name"]
                        conversation_state["dataset_selected"] = False  # reset if new workflow
                        result = await mcp_client.call_tool(fn_name, fn_args)

                    elif fn_name == "switch_workflow":
                        conversation_state["workflow_name"] = fn_args["workflow_name"]
                        conversation_state["dataset_selected"] = False  # reset dataset
                        result = await mcp_client.call_tool(fn_name, fn_args)

                    elif fn_name == "set_auto_labeling_hyperparams":
                        # Update local cache only with provided values
                        for k, v in fn_args.items():
                            if v is not None:
                                hyperparam_cache[k] = v
                        # Send full set to MCP tool
                        result = await mcp_client.call_tool(fn_name, hyperparam_cache.copy())

                    elif fn_name == "set_selected_dataset":
                        conversation_state["dataset_selected"] = True
                        selected_dataset_cache["dataset_name"] = fn_args["dataset_name"]
                        selected_dataset_cache["n_samples"] = None  # Always set to None
                        result = await mcp_client.call_tool(fn_name, {
                            "dataset_name": selected_dataset_cache["dataset_name"]
                        })


                    elif fn_name == "set_anomaly_detection_hyperparams":
                        for k, v in fn_args.items():
                            if v is not None:
                                hyperparam_cache_anomaly[k] = v
                        result = await mcp_client.call_tool(fn_name, hyperparam_cache_anomaly.copy())

                    elif fn_name == "set_embedding_selection_params":
                        for k, v in fn_args.items():
                            if v is not None:
                                embedding_selection_cache[k] = v
                        result = await mcp_client.call_tool(fn_name, embedding_selection_cache.copy())

                    elif fn_name == "set_ensemble_selection_parameters":
                        if "agreement_threshold" not in fn_args or fn_args["agreement_threshold"] is None:
                            return {"reply": "Please provide the required `agreement_threshold` parameter."}

                        for k, v in fn_args.items():
                            if v is not None:
                                ensemble_selection_cache[k] = v
                        result = await mcp_client.call_tool(fn_name, ensemble_selection_cache.copy())

                    else:
                        result = await mcp_client.call_tool(fn_name, fn_args)

                    tool_results.append({
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "result": result
                    })
                except Exception as e:
                    tool_results.append({
                        "tool_call_id": call.id,
                        "name": fn_name,
                        "error": str(e)
                    })

        # Step 3: Append tool messages and call  again
        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                } for call in tool_calls
            ]
        })

        for result in tool_results:
            fn_name = result["name"]
            tool_output = str(result.get("result", result.get("error", "Tool error.")))

            if fn_name == "run_auto_labeling":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                #tool_output = tool_output_raw.text if hasattr(tool_output_raw, "text") else str(tool_output_raw)

                tool_output = unwrap_tool_output(tool_output_raw)


                # Check if classification report exists in the output
                if "precision" in tool_output and "recall" in tool_output and "f1-score" in tool_output:
                    # Ask LLM to summarize inference results
                    summary = await llm.summarize_classification_report(tool_output)

                    reply = (
                        f"{summary}\n\n"
                        f"Full Classification Report:\n"
                        f"```\n{tool_output.strip()}\n```"
                        f"Would you like to launch Voxel51 to explore the results?"
                    )
                else:
                    # No inference results, just return the training confirmation
                    reply = f"{tool_output.strip()}"

                return {"reply": reply}

            elif fn_name == "run_class_mapping":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                #tool_output = tool_output_raw.text if hasattr(tool_output_raw, "text") else str(tool_output_raw)

                tool_output = unwrap_tool_output(tool_output_raw)

                # Optional: add summarization for class mapping
                summary = await llm.summarize_class_mapping_output(tool_output)


                reply = (
                    f"{summary}\n\n"
                    f"Class Mapping Output:\n"
                    f"```\n{tool_output.strip()}\n```"
                )
                return {"reply": reply}

            elif fn_name == "run_anomaly_detection":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                #tool_output = tool_output_raw.text if hasattr(tool_output_raw, "text") else str(tool_output_raw)

                # Extract raw text safely (works for Gemini and OpenAI)
                tool_output = unwrap_tool_output(tool_output_raw)


                # Optional: add summarization for class mapping
                summary = await llm.summarize_anomaly_detection_output(tool_output)

                reply = (
                    f"{summary}\n\n"
                    f"Anomaly Detection Output:\n"
                    f"```\n{tool_output.strip()}\n```"
                )
                return {"reply": reply}

            elif fn_name == "run_zero_shot_auto_labeling":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                #tool_output = tool_output_raw.text if hasattr(tool_output_raw, "text") else str(tool_output_raw)

                # If tool_output_raw is a list of TextContent, extract first and get text
                tool_output = unwrap_tool_output(tool_output_raw)

                reply = (
                    f"{tool_output.strip()}\n"
                    f"You can now use the Ensemble Selection workflow to identify detections where multiple models agree.\n"
                    f"Would you like to launch Voxel51 to explore the results?"
                )
                return {"reply": reply}

            elif fn_name == "run_ensemble_selection":
                tool_output_raw = result.get("result", result.get("error", "Tool error."))
                #tool_output = tool_output_raw.text if hasattr(tool_output_raw, "text") else str(tool_output_raw)

                # Robust extraction from possible TextContent or list of TextContent
                tool_output = unwrap_tool_output(tool_output_raw)

                reply = (
                    f"{tool_output.strip()}\n\n"
                    f"launch Voxel51 to explore the results?\n\n"
                    f"- In the ENSEMBLE SELECTION section of the left sidebar, use the `n_unique_ensemble_selection` field as a filter. "
                    f"- It represents the number of overlapping objects retained in each sample based on model agreement. "
                    f"- Once you select a sample image, use the `detections_overlap` tag from the TAGS panel to visualize only those detections that had sufficient overlap and were retained by the ensemble logic."
                )
                return {"reply": reply}

            # For other tools, keep old flow
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "name": fn_name,
                "content": tool_output
            })

        # If user selected a workflow but not a dataset, reinforce dataset selection
        if conversation_state["workflow_name"] and not conversation_state["dataset_selected"]:
            messages.append({
                "role": "system",
                "content": (
                    "Reminder: the user has selected a workflow but has not yet selected a dataset. "
                    "Guide them to choose one of the supported datasets: "
                    "fisheye8k, fisheye8k_mini, mcity_fisheye_2000, or mcity_fisheye_2100."
                )
            })


        # Continue with normal summarization for other tools
        final_response_msg = await llm.chat(messages)
        reply_content = getattr(final_response_msg, "content", final_response_msg)
        reply = unwrap_tool_output(reply_content)

    else:
        reply = assistant_message.content
    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
