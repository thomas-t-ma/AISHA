from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aisha.audio.base import SpeechArtifact, TTSProvider
from aisha.audio.store import EphemeralAudioStore, StoredAudio


class SpeechRuntime:
    """Coordinates TTS generation and transient local playback artifacts."""

    def __init__(
        self,
        provider: TTSProvider,
        store: EphemeralAudioStore,
        *,
        max_text_chars: int = 12000,
    ) -> None:
        self.provider = provider
        self.store = store
        self.max_text_chars = max(1, max_text_chars)
        self._syntheses = 0
        self._last_error: str | None = None

    async def synthesize(self, text: str) -> SpeechArtifact | None:
        clean = text.strip()
        if not clean:
            return None
        if not self.provider.status().get("enabled", False):
            return None
        if len(clean) > self.max_text_chars:
            self._last_error = (
                f"Speech text exceeded {self.max_text_chars} characters"
            )
            return None

        try:
            generated = await self.provider.synthesize(clean)
        except Exception as exc:  # noqa: BLE001 - optional TTS must not fail a turn
            self._last_error = str(exc)
            return None
        if generated is None:
            return None

        artifact = SpeechArtifact(
            content_type=generated.content_type,
            sample_rate=generated.sample_rate,
            duration_ms=generated.duration_ms,
            byte_length=len(generated.data),
        )
        self.store.put(artifact, generated.data)
        self._syntheses += 1
        self._last_error = None
        return artifact

    def get(self, utterance_id: str) -> StoredAudio | None:
        return self.store.get(utterance_id)

    def status(self) -> dict[str, Any]:
        return {
            **self.provider.status(),
            **self.store.status(),
            "syntheses": self._syntheses,
            "last_error": self._last_error,
            "checked_at": datetime.now(UTC).isoformat(),
        }

    def close(self) -> None:
        self.provider.close()
        self.store.clear()
