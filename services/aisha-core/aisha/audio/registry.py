from __future__ import annotations

from pathlib import Path

from aisha.audio.kokoro import KokoroTTSProvider
from aisha.audio.mock import DisabledTTSProvider
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
