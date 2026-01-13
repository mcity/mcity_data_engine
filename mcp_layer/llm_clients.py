import os
from typing import List, Optional
from openai import AsyncOpenAI
from groq import AsyncGroq
import google.generativeai as genai


class BaseLLMClient:
    async def chat(self, messages, tools: Optional[List] = None):
        raise NotImplementedError("Subclasses must implement this method")

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

    async def _summarize(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement summarization logic")


class OpenAIClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def chat(self, messages, tools=None):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None
        )
        return response.choices[0].message

    async def _summarize(self, prompt: str) -> str:
        response = await self.chat([{"role": "user", "content": prompt}])
        return response.content


class GroqClient(BaseLLMClient):
    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    async def chat(self, messages, tools=None):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None
        )
        return response.choices[0].message

    async def _summarize(self, prompt: str) -> str:
        response = await self.chat([{"role": "user", "content": prompt}])
        return response.content


class GeminiClient(BaseLLMClient):
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    async def chat(self, messages, tools=None):
        parts = [{"role": m["role"], "parts": [m["content"]]} for m in messages]
        try:
            response = await self.model.generate_content_async(parts)
            return {"content": response.text.strip(), "tool_calls": []}
        except Exception as e:
            return {"content": f"[Gemini error] {str(e)}", "tool_calls": []}

    async def _summarize(self, prompt: str) -> str:
        try:
            response = await self.model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            return f"[Gemini summarization error] {str(e)}"
