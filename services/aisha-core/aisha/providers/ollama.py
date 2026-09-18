from __future__ import annotations

import json
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

import httpx

from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext
from aisha.providers.base import AISHAProviderError, LLMStreamChunk


def _duration_ms(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    return round(value / 1_000_000, 3)


class OllamaLLMProvider:
    name = "ollama"
    capabilities = ProviderCapabilities(streaming_text=True, tools=False, vision=False)

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        think: bool | str | None = None,
        keep_alive: str | int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.think = think
        self.keep_alive = keep_alive
        self.options = options or {}

    async def warmup(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.options:
            payload["options"] = self.options

        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AISHAProviderError(f"Ollama warmup failed: {exc}") from exc

        return {
            "preloaded": True,
            "wall_clock_ms": round((perf_counter() - started) * 1000, 3),
            "load_ms": _duration_ms(data.get("load_duration")),
            "ollama_total_ms": _duration_ms(data.get("total_duration")),
        }

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[LLMStreamChunk]:
        messages = [{"role": "system", "content": context.system_prompt}]
        messages.extend(
            {"role": message.role, "content": message.text}
            for message in context.messages
            if message.role in {"user", "assistant", "system"}
        )
        messages.append({"role": "user", "content": context.user_input})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.options:
            payload["options"] = self.options

        timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client, client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise AISHAProviderError(str(data["error"]))

                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield LLMStreamChunk(text=content)

                    if data.get("done"):
                        metrics = {
                            "ollama_total_ms": _duration_ms(data.get("total_duration")),
                            "load_ms": _duration_ms(data.get("load_duration")),
                            "prompt_eval_ms": _duration_ms(data.get("prompt_eval_duration")),
                            "eval_ms": _duration_ms(data.get("eval_duration")),
                            "prompt_tokens": data.get("prompt_eval_count"),
                            "cached_prompt_tokens": data.get("prompt_eval_cached_count"),
                            "output_tokens": data.get("eval_count"),
                        }
                        yield LLMStreamChunk(final=True, metrics=metrics)
                        break
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise AISHAProviderError(f"Ollama request failed: {exc}") from exc
