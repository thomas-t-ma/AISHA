from __future__ import annotations

import io
import wave
from typing import Any

from aisha.audio.base import GeneratedSpeech


class DisabledTTSProvider:
    name = "disabled"
    model = "disabled"

    async def synthesize(self, text: str) -> GeneratedSpeech | None:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "provider": self.name,
            "model": self.model,
        }

    def close(self) -> None:
        return None


class MockTTSProvider:
    """Generates a tiny silent WAV for deterministic audio-path tests."""

    name = "mock"
    model = "mock-tts"

    def __init__(self, *, sample_rate: int = 16000, duration_ms: float = 100.0) -> None:
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms

    async def synthesize(self, text: str) -> GeneratedSpeech | None:
        if not text.strip():
            return None

        frame_count = max(1, round(self.sample_rate * self.duration_ms / 1000.0))
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(b"\x00\x00" * frame_count)

        return GeneratedSpeech(
            data=buffer.getvalue(),
            content_type="audio/wav",
            sample_rate=self.sample_rate,
            duration_ms=self.duration_ms,
        )

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "provider": self.name,
            "model": self.model,
        }

    def close(self) -> None:
        return None
