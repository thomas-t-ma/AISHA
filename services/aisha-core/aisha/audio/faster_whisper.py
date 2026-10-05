from __future__ import annotations

import asyncio
import importlib
import importlib.util
import io
from contextlib import suppress
from pathlib import Path
from typing import Any

from aisha.audio.transcription import TranscriptionResult


class FasterWhisperSTTProvider:
    """Lazy local faster-whisper provider for bounded push-to-talk audio."""

    name = "faster-whisper"

    def __init__(
        self,
        model: str,
        *,
        download_root: Path,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "en",
        beam_size: int = 3,
        vad_filter: bool = True,
        local_files_only: bool = True,
    ) -> None:
        self.model = model
        self.download_root = download_root.expanduser().resolve()
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.local_files_only = local_files_only
        self._engine: Any | None = None
        self._last_error: str | None = None

    @staticmethod
    def _dependency_available() -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        self.download_root.mkdir(parents=True, exist_ok=True)
        module = importlib.import_module("faster_whisper")
        self._engine = module.WhisperModel(
            self.model,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.download_root),
            local_files_only=self.local_files_only,
        )
        return self._engine

    def _transcribe_sync(
        self,
        audio: bytes,
        content_type: str,
    ) -> TranscriptionResult | None:
        del content_type  # PyAV detects the container from the file-like object.
        engine = self._ensure_engine()
        segments, info = engine.transcribe(
            io.BytesIO(audio),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        completed = list(segments)
        text = "".join(segment.text for segment in completed).strip()
        if not text:
            return None

        language = getattr(info, "language", None)
        probability = getattr(info, "language_probability", None)
        duration = getattr(info, "duration", None)
        return TranscriptionResult(
            text=text,
            language=str(language) if language else self.language,
            language_probability=(
                float(probability) if probability is not None else None
            ),
            duration_ms=(
                float(duration) * 1000.0 if duration is not None else None
            ),
            provider=self.name,
            model=self.model,
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str,
    ) -> TranscriptionResult | None:
        if not audio:
            return None
        try:
            result = await asyncio.to_thread(
                self._transcribe_sync,
                audio,
                content_type,
            )
        except Exception as exc:  # noqa: BLE001 - isolate optional STT backend failures
            self._last_error = str(exc)
            return None
        self._last_error = None
        return result

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._dependency_available(),
            "provider": self.name,
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "local_files_only": self.local_files_only,
            "dependency_available": self._dependency_available(),
            "download_root": str(self.download_root),
            "loaded": self._engine is not None,
            "last_error": self._last_error,
        }

    def close(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is None:
            return
        model = getattr(engine, "model", None)
        unload = getattr(model, "unload_model", None)
        if callable(unload):
            with suppress(Exception):
                unload()
