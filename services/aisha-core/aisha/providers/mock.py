from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext
from aisha.providers.base import LLMStreamChunk


class MockLLMProvider:
    name = "mock"
    model = "mock-aisha"
    capabilities = ProviderCapabilities(streaming_text=True)

    async def warmup(self) -> dict[str, Any]:
        return {"provider": "mock", "preloaded": True}

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[LLMStreamChunk]:
        text = f"AISHA core is working. You said: {context.user_input}"
        for token in text.split(" "):
            await asyncio.sleep(0.01)
            yield LLMStreamChunk(text=token + " ")
        yield LLMStreamChunk(final=True, metrics={"provider": "mock"})
