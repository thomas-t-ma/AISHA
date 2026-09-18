from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext


class AISHAProviderError(RuntimeError):
    """Normalized failure raised by AISHA compute-provider adapters."""


@dataclass(frozen=True)
class LLMStreamChunk:
    text: str = ""
    final: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str
    model: str
    capabilities: ProviderCapabilities

    async def warmup(self) -> dict[str, Any]: ...

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[LLMStreamChunk]: ...
