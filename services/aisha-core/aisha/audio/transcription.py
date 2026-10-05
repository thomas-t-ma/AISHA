from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    text: str = Field(min_length=1)
    language: str | None = None
    language_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_ms: float | None = Field(default=None, ge=0.0)
    provider: str
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class STTProvider(Protocol):
    name: str
    model: str

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str,
    ) -> TranscriptionResult | None: ...

    def status(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class DisabledSTTProvider:
    name = "disabled"
    model = "disabled"

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str,
    ) -> TranscriptionResult | None:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "provider": self.name,
            "model": self.model,
        }

    def close(self) -> None:
        return None


class MockSTTProvider:
    name = "mock"
    model = "mock-stt"

    def __init__(self, text: str = "mock transcription") -> None:
        self.text = text

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str,
    ) -> TranscriptionResult | None:
        if not audio or not self.text.strip():
            return None
        return TranscriptionResult(
            text=self.text.strip(),
            language="en",
            language_probability=1.0,
            duration_ms=None,
            provider=self.name,
            model=self.model,
        )

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "provider": self.name,
            "model": self.model,
        }

    def close(self) -> None:
        return None
