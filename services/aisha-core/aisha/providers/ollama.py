from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext


class OllamaLLMProvider:
    name = "ollama"
    capabilities = ProviderCapabilities(streaming_text=True, tools=False, vision=False)

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": context.system_prompt}]
        messages.extend(
            {"role": message.role, "content": message.text}
            for message in context.messages
            if message.role in {"user", "assistant", "system"}
        )
        messages.append({"role": "user", "content": context.user_input})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(data["error"])
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if data.get("done"):
                        break
