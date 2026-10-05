from __future__ import annotations

from pathlib import Path

from aisha.audio.faster_whisper import FasterWhisperSTTProvider
from aisha.audio.kokoro import KokoroTTSProvider
from aisha.audio.mock import DisabledTTSProvider
from aisha.audio.transcription import DisabledSTTProvider
from aisha.audio.transcription_runtime import TranscriptionRuntime
from aisha.audio.runtime import SpeechRuntime
from aisha.audio.store import EphemeralAudioStore
from aisha.settings import RuntimeProfile, Settings


def _resolve_local_path(
    settings: Settings,
    configured: str | None,
    default_name: str,
) -> Path:
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = settings.data_dir / path
        return path.resolve()
    return (settings.data_dir / "models" / "tts" / default_name).resolve()


def build_speech_runtime(
    settings: Settings,
    profile: RuntimeProfile,
) -> SpeechRuntime:
    store = EphemeralAudioStore(
        ttl_seconds=profile.tts.artifact_ttl_seconds,
        max_items=8,
    )

    if profile.tts.provider == "local-kokoro":
        provider = KokoroTTSProvider(
            _resolve_local_path(
                settings,
                profile.tts.model_path,
                "kokoro-v1.0.onnx",
            ),
            _resolve_local_path(
                settings,
                profile.tts.voices_path,
                "voices-v1.0.bin",
            ),
            voice=profile.tts.voice,
            speed=profile.tts.speed,
            language=profile.tts.language,
        )
        return SpeechRuntime(provider, store)

    return SpeechRuntime(DisabledTTSProvider(), store)



def build_transcription_runtime(
    settings: Settings,
    profile: RuntimeProfile,
) -> TranscriptionRuntime:
    if profile.stt.provider == "local-faster-whisper":
        provider = FasterWhisperSTTProvider(
            profile.stt.model,
            download_root=(settings.data_dir / "models" / "stt").resolve(),
            device=profile.stt.device,
            compute_type=profile.stt.compute_type,
            language=profile.stt.language,
            beam_size=profile.stt.beam_size,
            vad_filter=profile.stt.vad_filter,
        )
        return TranscriptionRuntime(
            provider,
            max_audio_bytes=profile.stt.max_audio_bytes,
        )

    return TranscriptionRuntime(
        DisabledSTTProvider(),
        max_audio_bytes=profile.stt.max_audio_bytes,
    )
