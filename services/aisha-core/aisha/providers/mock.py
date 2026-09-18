from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext


class MockLLMProvider:
    name = "mock"
    model = "mock-aisha"
    capabilities = ProviderCapabilities(streaming_text=True)

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[str]:
        text = f"AISHA core is working. You said: {context.user_input}"
        for token in text.split(" "):
            await asyncio.sleep(0.01)
            yield token + " "
