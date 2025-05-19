# mcp_layer/chat_server.py

from fastapi import FastAPI, Request
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from groq import AsyncGroq
import os
import json

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
MCP_TRANSPORT = SSETransport(url="http://localhost:8000/sse")

SYSTEM_PROMPT = """
You are the MCity Data Engine Agent. Your job is to help the user configure and run the `auto_labeling` workflow using the MCity Data Engine.

Your responsibilities are:

1. Guide the user to choose a `model_source` (ultralytics, hf_models_objectdetection, or custom_codetr), When the user selects a `model_source`, ALWAYS call the tool `list_model_sources_and_models` to fetch available models from the local config — DO NOT guess or hallucinate. Remember this tool call does not take any input arguments.
2. Then help them select a specific model or config within that source, do not call the `configute_autolabeling_tool` until the user finalizes it.
3. Use the `configure_auto_labeling` tool to set the model. ONLY pass `selected_source` and `selected_model` to this tool. Do NOT include hyperparameters like `mode` or `epochs` here.
4. If the user wants to modify hyperparameters, update any of the following:
   - `mode`: Options are ["train"], ["inference"], or ["train", "inference"]
   - `epochs`: Suggested default is 10
   - `early_stop_patience`: Suggested default is 5
   - `early_stop_threshold`: Suggested default is 0
   - `learning_rate`: Suggested default is 5e-5
   - `weight_decay`: Suggested default is 0.0001
   - `max_grad_norm`: Suggested default is 0.01
5. After changing a hyperparameter, DO NOT immediately run the workflow. Instead, ask:
   “Would you like to modify any other hyperparameters before we start the workflow?” And Finally  call `set_auto_labeling_hyperparams`, by passing all the hyperparameters that the user changed, and the others can remain default.
6. Ensure that the hyperparameters have been updated by the `set_auto_labeling_hyperparams`, with the parameters that the user mentioned.
7. Finally confirm with the user to run `run_auto_labeling`, do not explicityly ask them ifthey want to use the tool. Rather let them know that the hyperparameters have been updated successfully and the workflow is ready to be executed. remember this tool does not take any input arguments, thus execute it when the user explicitly says something like:
   - “Run the workflow”
   - “Start training”
   - “Let’s begin”

You can also explain what workflows, models, or hyperparameters do. Follow up with appropriate tool calls based on what the user wants to do.
"""

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

    # Define MCP tools for Groq
    tools = [
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
                "name": "run_auto_labeling",
                "description": "Run main.py and stream logs in real time.",
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
        }
    ]

    # Step 1: Initial response
    response = await groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = response.choices[0].message
    hyperparam_cache = {
    "mode": ["train", "inference"],
    "epochs": 10,
    "early_stop_patience": 5,
    "early_stop_threshold": 0,
    "learning_rate": 5e-5,
    "weight_decay": 0.0001,
    "max_grad_norm": 0.01,
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
                    if fn_name == "set_auto_labeling_hyperparams":
                        # Update local cache only with provided values
                        for k, v in fn_args.items():
                            if v is not None:
                                hyperparam_cache[k] = v
                        # Send full set to MCP tool
                        result = await mcp_client.call_tool(fn_name, hyperparam_cache.copy())
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

        # Step 3: Append tool messages and call Groq again
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
                tool_output = tool_output_raw.text if hasattr(tool_output_raw, "text") else str(tool_output_raw)
                # Ask LLM to summarize the classification report
                summary_prompt = f"""
                Here's a classification report from an object detection model run. Briefly summarize how the model performed.
                {tool_output}
                Only mention:
                - Which class did best
                - Which class was worst
                - What does the micro/macro F1 tell us about generalization

                """
                summary_response = await groq_client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                summary = summary_response.choices[0].message.content.strip()

                # Combine both into the final reply
                reply = (
                    f"{summary}\n\n"
                    f"📊 **Full Classification Report:**\n"
                    f"```\n{tool_output.strip()}\n```"
                )
                return {"reply": reply}

            # For other tools, keep old flow
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "name": fn_name,
                "content": tool_output
            })

        # Continue with normal summarization for other tools
        final_response = await groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages
        )
        reply = final_response.choices[0].message.content
    else:
        reply = assistant_message.content

    return {"reply": reply}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
