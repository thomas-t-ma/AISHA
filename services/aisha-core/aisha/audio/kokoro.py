from __future__ import annotations

import asyncio
import importlib
import importlib.util
import io
from contextlib import suppress
from pathlib import Path
from typing import Any

from aisha.audio.base import GeneratedSpeech


class KokoroTTSProvider:
    """Optional local Kokoro ONNX provider; loaded only on first synthesis."""

    name = "kokoro-onnx"
    model = "kokoro-v1.0"

    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        language: str = "en-us",
    ) -> None:
        self.model_path = model_path.expanduser().resolve()
        self.voices_path = voices_path.expanduser().resolve()
        self.voice = voice
        self.speed = speed
        self.language = language
        self._engine: Any | None = None
        self._last_error: str | None = None

    @staticmethod
    def _dependency_available() -> bool:
        return (
            importlib.util.find_spec("kokoro_onnx") is not None
            and importlib.util.find_spec("soundfile") is not None
        )

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if not self.model_path.exists():
            raise FileNotFoundError(f"Kokoro model not found: {self.model_path}")
        if not self.voices_path.exists():
            raise FileNotFoundError(f"Kokoro voices not found: {self.voices_path}")

        module = importlib.import_module("kokoro_onnx")
        self._engine = module.Kokoro(
            str(self.model_path),
            str(self.voices_path),
        )
        return self._engine

    def _synthesize_sync(self, text: str) -> GeneratedSpeech:
        engine = self._ensure_engine()
        samples, sample_rate = engine.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=self.language,
        )

        soundfile = importlib.import_module("soundfile")
        buffer = io.BytesIO()
        soundfile.write(
            buffer,
            samples,
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        duration_ms = (
            0.0
            if sample_rate <= 0
            else len(samples) / float(sample_rate) * 1000.0
        )
        return GeneratedSpeech(
            data=buffer.getvalue(),
            content_type="audio/wav",
            sample_rate=int(sample_rate),
            duration_ms=duration_ms,
        )

    async def synthesize(self, text: str) -> GeneratedSpeech | None:
        if not text.strip():
            return None
        try:
            generated = await asyncio.to_thread(self._synthesize_sync, text)
        except Exception as exc:  # noqa: BLE001 - isolate optional TTS backend failures
            self._last_error = str(exc)
            return None
        self._last_error = None
        return generated

    def status(self) -> dict[str, Any]:
        return {
            "enabled": (
                self._dependency_available()
                and self.model_path.exists()
                and self.voices_path.exists()
            ),
            "provider": self.name,
            "model": self.model,
            "voice": self.voice,
            "speed": self.speed,
            "language": self.language,
            "dependency_available": self._dependency_available(),
            "model_available": self.model_path.exists(),
            "voices_available": self.voices_path.exists(),
            "model_path": str(self.model_path),
            "voices_path": str(self.voices_path),
            "loaded": self._engine is not None,
            "last_error": self._last_error,
        }

    def close(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is None:
            return
        voices = getattr(engine, "voices", None)
        close = getattr(voices, "close", None)
        if callable(close):
            with suppress(Exception):
                close()
