from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aisha.audio.transcription import STTProvider, TranscriptionResult


class TranscriptionRuntime:
    """Bounded ephemeral speech-to-text runtime.

    Raw microphone bytes exist only for the duration of one request and are never
    retained by the runtime.
    """

    def __init__(
        self,
        provider: STTProvider,
        *,
        max_audio_bytes: int = 12 * 1024 * 1024,
    ) -> None:
        self.provider = provider
        self.max_audio_bytes = max(64 * 1024, max_audio_bytes)
        self._transcriptions = 0
        self._last_error: str | None = None

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str,
    ) -> TranscriptionResult | None:
        if not self.provider.status().get("enabled", False):
            self._last_error = "Speech-to-text provider is disabled"
            return None
        if not audio:
            self._last_error = "Audio payload was empty"
            return None
        if len(audio) > self.max_audio_bytes:
            self._last_error = (
                f"Audio payload exceeded {self.max_audio_bytes} bytes"
            )
            return None

        try:
            result = await self.provider.transcribe(
                audio,
                content_type=content_type,
            )
        except Exception as exc:  # noqa: BLE001 - optional STT must not fail Core
            self._last_error = str(exc)
            return None

        if result is None:
            provider_error = self.provider.status().get("last_error")
            self._last_error = (
                str(provider_error)
                if isinstance(provider_error, str) and provider_error
                else "No speech was transcribed"
            )
            return None

        self._transcriptions += 1
        self._last_error = None
        return result

    def status(self) -> dict[str, Any]:
        provider_status = self.provider.status()
        provider_error = provider_status.get("last_error")
        return {
            **provider_status,
            "transcriptions": self._transcriptions,
            "max_audio_bytes": self.max_audio_bytes,
            "audio_persisted": False,
            "last_error": self._last_error or provider_error,
            "checked_at": datetime.now(UTC).isoformat(),
        }

    def close(self) -> None:
        self.provider.close()
