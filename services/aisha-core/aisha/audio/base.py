from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field


@dataclass(slots=True)
class GeneratedSpeech:
    data: bytes
    content_type: str
    sample_rate: int
    duration_ms: float


class SpeechArtifact(BaseModel):
    utterance_id: str = Field(default_factory=lambda: f"utt_{uuid4().hex}")
    content_type: str = "audio/wav"
    sample_rate: int = Field(gt=0)
    duration_ms: float = Field(ge=0.0)
    byte_length: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TTSProvider(Protocol):
    name: str
    model: str

    async def synthesize(self, text: str) -> GeneratedSpeech | None: ...

    def status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
