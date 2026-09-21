from __future__ import annotations

import asyncio
import importlib.util
import io
import platform
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from time import perf_counter
from typing import Any


class VoiceUnavailableError(RuntimeError):
    """A required optional speech runtime is not installed or configured."""


class VoiceInputError(ValueError):
    """An audio/text request is invalid or exceeds the voice safety bounds."""


MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_SPEECH_SECONDS = 30
MAX_SPEECH_TEXT = 1200
SAMPLE_RATE = 24000


class VoiceService:
    """Mac-first, optional voice inference. No recordings are retained.

    Models are loaded lazily. Long-running calls execute in worker threads so
    Ollama's async streaming and Studio's WebSocket are not blocked.
    """

    def __init__(
        self,
        *,
        stt_provider: str = "disabled",
        stt_model: str = "mlx-community/whisper-small",
        tts_provider: str = "disabled",
        tts_voice: str = "af_heart",
    ) -> None:
        self.stt_provider = stt_provider
        self.stt_model = stt_model
        self.tts_provider = tts_provider
        self.tts_voice = tts_voice
        self._stt_lock = asyncio.Lock()
        self._tts_lock = asyncio.Lock()
        self._tts_pipeline: Any = None

    def status(self) -> dict[str, Any]:
        mac = platform.system() == "Darwin" and platform.machine() == "arm64"
        ffmpeg = shutil.which("ffmpeg") is not None
        espeak = shutil.which("espeak-ng") is not None
        whisper = importlib.util.find_spec("mlx_whisper") is not None
        kokoro = importlib.util.find_spec("kokoro") is not None
        stt_ready = self.stt_provider == "mlx-whisper" and mac and ffmpeg and whisper
        tts_ready = self.tts_provider == "kokoro-local" and kokoro and espeak
        return {
            "stt": {
                "provider": self.stt_provider,
                "model": self.stt_model,
                "ready": stt_ready,
                "missing": (
                    [name for name, available in [
                        ("Apple Silicon macOS", mac),
                        ("ffmpeg", ffmpeg),
                        ("mlx-whisper", whisper),
                    ] if not available]
                    if self.stt_provider == "mlx-whisper" else ["not enabled in profile"]
                ),
            },
            "tts": {
                "provider": self.tts_provider,
                "voice": self.tts_voice,
                "ready": tts_ready,
                "missing": (
                    [name for name, available in [
                        ("kokoro", kokoro),
                        ("espeak-ng", espeak),
                    ] if not available]
                    if self.tts_provider == "kokoro-local" else ["not enabled in profile"]
                ),
            },
            "max_audio_seconds": MAX_SPEECH_SECONDS,
            "max_audio_bytes": MAX_AUDIO_BYTES,
            "max_speech_text": MAX_SPEECH_TEXT,
        }

    async def transcribe(self, audio: bytes, extension: str) -> tuple[str, float]:
        state = self.status()["stt"]
        if not state["ready"]:
            raise VoiceUnavailableError("Local transcription unavailable: " + ", ".join(state["missing"]))
        if not audio or len(audio) > MAX_AUDIO_BYTES:
            raise VoiceInputError("Recording must be between 1 byte and 12 MiB.")
        if extension not in {".webm", ".mp4", ".ogg", ".wav"}:
            raise VoiceInputError("Unsupported recording format.")

        started = perf_counter()
        async with self._stt_lock:
            text = await asyncio.to_thread(self._transcribe_sync, audio, extension)
        return text.strip(), round((perf_counter() - started) * 1000, 3)

    def _transcribe_sync(self, audio: bytes, extension: str) -> str:
        import mlx_whisper

        with tempfile.TemporaryDirectory(prefix="aisha-voice-") as directory:
            source = Path(directory) / ("input" + extension)
            target = Path(directory) / "speech.wav"
            source.write_bytes(audio)
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-y", "-i", str(source), "-t", str(MAX_SPEECH_SECONDS),
                        "-ar", "16000", "-ac", "1", str(target),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=45,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise VoiceInputError("Could not decode the recording.") from exc

            result = mlx_whisper.transcribe(
                str(target),
                path_or_hf_repo=self.stt_model,
                verbose=None,
            )
            return str(result.get("text", ""))

    async def synthesize(self, text: str) -> tuple[bytes, float]:
        state = self.status()["tts"]
        if not state["ready"]:
            raise VoiceUnavailableError("Local speech output unavailable: " + ", ".join(state["missing"]))
        clean_text = text.strip()
        if not clean_text or len(clean_text) > MAX_SPEECH_TEXT:
            raise VoiceInputError(f"Speech text must be 1–{MAX_SPEECH_TEXT} characters.")

        started = perf_counter()
        async with self._tts_lock:
            audio = await asyncio.to_thread(self._synthesize_sync, clean_text)
        return audio, round((perf_counter() - started) * 1000, 3)

    def _synthesize_sync(self, text: str) -> bytes:
        import numpy as np
        from kokoro import KPipeline

        if self._tts_pipeline is None:
            self._tts_pipeline = KPipeline(lang_code="a")

        chunks = []
        for _graphemes, _phonemes, audio in self._tts_pipeline(text, voice=self.tts_voice):
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not chunks:
            raise VoiceInputError("Speech synthesis produced no audio.")

        waveform = np.concatenate(chunks)
        pcm = (np.clip(waveform, -1, 1) * 32767).astype("<i2")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(pcm.tobytes())
        return buffer.getvalue()
