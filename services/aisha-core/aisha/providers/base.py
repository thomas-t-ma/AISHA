from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext


class AISHAProviderError(RuntimeError):
    """Normalized failure raised by AISHA compute-provider adapters."""


class LLMProvider(Protocol):
    name: str
    model: str
    capabilities: ProviderCapabilities

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[str]: ...
