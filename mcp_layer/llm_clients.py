import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from groq import AsyncGroq
import google.generativeai as genai


class _FakeFunction:
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    __slots__ = ("id", "function")

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    __slots__ = ("content", "tool_calls")

    def __init__(self, content: Optional[str], tool_calls: list):
        self.content = content
        self.tool_calls = tool_calls


class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, messages, tools: Optional[List] = None, tool_choice: Optional[str] = None):
        raise NotImplementedError

    async def summarize_classification_report(self, tool_output: str) -> str:
        prompt = f"""
        Here's a classification report/results from an object detection model run. Briefly summarize how the model performed.

        {tool_output}

        Only mention:
        - Which class did best
        - Which class was worst
        - What does the micro/macro F1 tell us about generalization
        """
        return await self._summarize(prompt)

    async def summarize_class_mapping_output(self, tool_output: str) -> str:
        prompt = f"""
        Here's the output from a class mapping workflow. Summarize what was done based on the tag addition results section, and finally ask the user if they would like to visualize the results of the workflow using Voxel51.

        {tool_output}

        Only include:
        - How many tags were added, in total.
        - How many tags were added, in each category.
        - A brief summary based on the tag addition.
        """
        return await self._summarize(prompt)

    async def summarize_anomaly_detection_output(self, tool_output: str) -> str:
        prompt = f"""
        Here's the output from anomaly detection workflow. Summarize these results, and give a brief overview of what these values mean and indicate about the model performance.

        {tool_output}

        Finally, suggest that the user explore the results using Voxel51 by selecting the appropriate camera view and inspecting anomaly scores and masks. In particular, encourage them to filter samples by anomaly score (e.g., `pred_anomaly_score_<model>`) to view the most anomalous examples, and to use the corresponding `pred_anomaly_mask_<model>` field to visualize pixel-wise anomaly regions. This mask field name varies depending on the model used (e.g., `Padim`, `STFPM`, etc.).
        """
        return await self._summarize(prompt)

    @abstractmethod
    async def _summarize(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def chat(self, messages, tools=None, tool_choice=None):
        kwargs = dict(model=self.model, messages=messages, temperature=0.1)
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = False  # prevents missing-field errors with tool_choice="required"
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    async def _summarize(self, prompt: str) -> str:
        response = await self.chat([{"role": "user", "content": prompt}])
        return getattr(response, "content", "") or ""


class GroqClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    async def chat(self, messages, tools=None, tool_choice=None):
        kwargs = dict(model=self.model, messages=messages, temperature=0.1)
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    async def _summarize(self, prompt: str) -> str:
        response = await self.chat([{"role": "user", "content": prompt}])
        return getattr(response, "content", "") or ""


class GeminiClient(BaseLLMClient):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    async def chat(self, messages, tools=None, tool_choice=None):
        if tools:
            raise NotImplementedError(
                "GeminiClient does not yet support tool use; do not set "
                "LLM_PROVIDER=gemini for workflows requiring tool calls."
            )
        # Gemini: no tool use; skip system/tool roles; map "assistant" → "model".
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("system", "tool") or not isinstance(content, str):
                continue
            gemini_role = "model" if role == "assistant" else role
            parts.append({"role": gemini_role, "parts": [content]})
        while parts and parts[0]["role"] != "user":
            parts.pop(0)
        if not parts:
            parts = [{"role": "user", "parts": [""]}]
        try:
            response = await self.model.generate_content_async(parts, generation_config={"temperature": 0.1})
            return _FakeMessage(content=response.text.strip(), tool_calls=[])
        except Exception as e:
            return _FakeMessage(content=f"[Gemini error] {str(e)}", tool_calls=[])

    async def _summarize(self, prompt: str) -> str:
        try:
            response = await self.model.generate_content_async(prompt, generation_config={"temperature": 0.1})
            return response.text.strip()
        except Exception as e:
            return f"[Gemini summarization error] {str(e)}"


class ClaudeClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    async def chat(self, messages, tools=None, tool_choice=None):
        system_parts, anthropic_messages = self._to_anthropic_messages(messages)

        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": "Hello"}]

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
            "temperature": 0.1,
        }

        if system_parts:
            # Only cache the first (static) block; dynamic injections don't get cache_control.
            system_blocks = [
                {"type": "text", "text": system_parts[0], "cache_control": {"type": "ephemeral"}},
                *[{"type": "text", "text": part} for part in system_parts[1:]],
            ]
            kwargs["system"] = system_blocks

        if tools:
            converted = self._convert_tools(tools)
            if converted:
                converted[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = converted
            kwargs["tool_choice"] = {"type": "any"} if tool_choice == "required" else {"type": "auto"}

        try:
            response = await self.client.messages.create(**kwargs)
        except Exception as e:
            logging.warning(f"[CLAUDE] API error: {e}")
            raise

        return self._to_fake_message(response)

    async def _summarize(self, prompt: str) -> str:
        response = await self.chat([{"role": "user", "content": prompt}])
        return response.content or ""

    @staticmethod
    def _convert_tools(tools: list) -> list:
        result = []
        for t in tools:
            fn = t.get("function", {})
            result.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result

    @staticmethod
    def _to_anthropic_messages(openai_messages: list) -> tuple[str, list]:
        """Convert OpenAI-format messages to Anthropic format (system list + messages list)."""
        system_parts: list[str] = []
        result: list[dict] = []

        for msg in openai_messages:
            role = msg.get("role", "")

            if role == "system":
                text = (msg.get("content", "") or "").strip()
                if text:
                    system_parts.append(text)
                continue

            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": str(msg.get("content", "")),
                }
                if result and result[-1]["role"] == "user":
                    prev = result[-1]["content"]
                    if isinstance(prev, list):
                        prev.append(block)
                    else:
                        result[-1]["content"] = ([{"type": "text", "text": prev}] if prev else []) + [block]
                else:
                    result.append({"role": "user", "content": [block]})
                continue

            if role == "assistant":
                tool_calls_raw = msg.get("tool_calls") or []
                blocks: list[dict] = []

                text = msg.get("content") or ""
                if text:
                    blocks.append({"type": "text", "text": text})

                for tc in tool_calls_raw:
                    if isinstance(tc, dict):
                        tc_id = tc.get("id", "")
                        fn = tc.get("function") or {}
                        fn_name = fn.get("name", "") if isinstance(fn, dict) else ""
                        fn_args_str = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
                    else:
                        tc_id = getattr(tc, "id", "")
                        fn_obj = getattr(tc, "function", None)
                        fn_name = getattr(fn_obj, "name", "") if fn_obj else ""
                        fn_args_str = getattr(fn_obj, "arguments", "{}") if fn_obj else "{}"

                    try:
                        inp = json.loads(fn_args_str)
                    except Exception:
                        inp = {}

                    blocks.append({"type": "tool_use", "id": tc_id, "name": fn_name, "input": inp})

                if not blocks:
                    blocks = [{"type": "text", "text": " "}]

                result.append({"role": "assistant", "content": blocks})
                continue

            if role == "user":
                content = msg.get("content", "") or ""
                if result and result[-1]["role"] == "user":
                    prev = result[-1]["content"]
                    if isinstance(prev, list):
                        if content:
                            prev.append({"type": "text", "text": content})
                    else:
                        if content:
                            result[-1]["content"] = (prev + "\n" + content).strip()
                else:
                    result.append({"role": "user", "content": content})
                continue

        while result and result[0]["role"] != "user":
            result.pop(0)

        result = [
            m for m in result
            if m.get("content") not in (None, "", [], [{"type": "text", "text": ""}])
        ]

        return system_parts, result

    @staticmethod
    def _to_fake_message(response) -> "_FakeMessage":
        text_parts: list[str] = []
        tool_calls: list[_FakeToolCall] = []

        for block in response.content:
            if block.type == "text":
                if block.text:
                    text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(_FakeToolCall(id=block.id, name=block.name, arguments=json.dumps(block.input)))

        return _FakeMessage(content="".join(text_parts) or None, tool_calls=tool_calls)