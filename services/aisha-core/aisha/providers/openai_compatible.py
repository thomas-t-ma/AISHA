from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext


class OpenAICompatibleLLMProvider:
    """Generic /v1/chat/completions streaming adapter.

    Intended for hosted APIs and local servers such as vLLM/SGLang when configured
    to expose an OpenAI-compatible endpoint. Provider-specific features should be
    added through capabilities/adapters rather than leaking into AISHA Core.
    """

    name = "openai-compatible"
    capabilities = ProviderCapabilities(streaming_text=True, remote=True)

    def __init__(self, model: str, base_url: str, api_key: str | None = None) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": context.system_prompt}]
        messages.extend(
            {"role": message.role, "content": message.text}
            for message in context.messages
            if message.role in {"user", "assistant", "system"}
        )
        messages.append({"role": "user", "content": context.user_input})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"model": self.model, "messages": messages, "stream": True}
        timeout = httpx.Timeout(connect=15.0, read=None, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    delta = event.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
