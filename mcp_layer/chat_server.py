# mcp_layer/chat_server.py

from fastapi import FastAPI, Request
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from groq import AsyncGroq
import os
import json

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
MCP_TRANSPORT = SSETransport(url="http://localhost:8000/sse")

SYSTEM_PROMPT = """
You are the MCity Data Engine Agent. Your job is to help the user configure the `auto_labeling` workflow.

* Guide the user to choose a `model_source` (ultralytics, hf_models_objectdetection, or custom_codetr).
* Then help them select a specific model or config within that source.
* Call MCP tools to update config.py and run main.py.
* You can also explain what workflows and models do.
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
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "name": result["name"],
                "content": str(result.get("result", result.get("error", "Tool error.")))
            })

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
