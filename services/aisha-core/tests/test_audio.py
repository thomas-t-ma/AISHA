from __future__ import annotations

import wave
from io import BytesIO

import pytest

import aisha.audio.store as audio_store_module
from aisha.audio.mock import DisabledTTSProvider, MockTTSProvider
from aisha.audio.registry import build_speech_runtime
from aisha.audio.runtime import SpeechRuntime
from aisha.audio.store import EphemeralAudioStore
from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.providers.mock import MockLLMProvider
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


@pytest.mark.asyncio
async def test_disabled_tts_produces_no_artifact():
    runtime = SpeechRuntime(
        DisabledTTSProvider(),
        EphemeralAudioStore(),
    )

    assert await runtime.synthesize("hello") is None
    assert runtime.status()["enabled"] is False
    assert runtime.status()["artifact_count"] == 0


@pytest.mark.asyncio
async def test_mock_tts_artifact_is_valid_ephemeral_wav():
    runtime = SpeechRuntime(
        MockTTSProvider(sample_rate=16000, duration_ms=100.0),
        EphemeralAudioStore(ttl_seconds=120.0, max_items=2),
    )

    artifact = await runtime.synthesize("hello")
    assert artifact is not None
    assert artifact.content_type == "audio/wav"
    assert artifact.sample_rate == 16000
    assert artifact.duration_ms == pytest.approx(100.0)
    assert artifact.byte_length > 44

    stored = runtime.get(artifact.utterance_id)
    assert stored is not None
    assert stored.artifact.utterance_id == artifact.utterance_id
    with wave.open(BytesIO(stored.data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000

    assert runtime.status()["persisted"] is False
    assert runtime.status()["artifact_count"] == 1


def test_ephemeral_audio_store_expires_and_evicts(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(audio_store_module, "monotonic", lambda: now[0])

    runtime = SpeechRuntime(
        MockTTSProvider(),
        EphemeralAudioStore(ttl_seconds=2.0, max_items=1),
    )

    # Exercise the store directly so the test clock remains deterministic.
    from aisha.audio.base import SpeechArtifact

    one = SpeechArtifact(
        sample_rate=16000,
        duration_ms=10.0,
        byte_length=3,
    )
    runtime.store.put(one, b"one")
    assert runtime.get(one.utterance_id) is not None

    two = SpeechArtifact(
        sample_rate=16000,
        duration_ms=10.0,
        byte_length=3,
    )
    runtime.store.put(two, b"two")
    assert runtime.get(one.utterance_id) is None
    assert runtime.get(two.utterance_id) is not None

    now[0] = 103.0
    assert runtime.get(two.utterance_id) is None


def test_ephemeral_audio_store_enforces_total_byte_limit():
    from aisha.audio.base import SpeechArtifact

    store = EphemeralAudioStore(
        ttl_seconds=120.0,
        max_items=8,
        max_bytes=1024,
    )
    oversized = SpeechArtifact(
        sample_rate=16000,
        duration_ms=100.0,
        byte_length=2048,
    )

    assert store.put(oversized, b"x" * 2048) is False
    assert store.get(oversized.utterance_id) is None
    assert store.status()["artifact_count"] == 0
    assert store.status()["total_bytes"] == 0
    assert store.status()["max_bytes"] == 1024


@pytest.mark.asyncio
async def test_completed_turn_emits_transient_audio_after_turn_finished(tmp_path):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    session_id = await store.create_session()
    speech = SpeechRuntime(
        MockTTSProvider(),
        EphemeralAudioStore(),
    )
    orchestrator = AISHAOrchestrator(
        store,
        load_persona(settings.character_dir),
        MockLLMProvider(),
        speech_synthesizer=speech.synthesize,
    )

    events = [
        event
        async for event in orchestrator.stream_user_turn(session_id, "hello")
    ]
    types = [event.type for event in events]

    assert "aisha.turn.finished" in types
    assert "aisha.audio.ready" in types
    assert types.index("aisha.audio.ready") > types.index("aisha.turn.finished")

    audio_event = next(event for event in events if event.type == "aisha.audio.ready")
    assert audio_event.payload["content_type"] == "audio/wav"
    assert audio_event.payload["playback_url"].startswith("/v1/audio/utt_")
    assert speech.get(audio_event.payload["utterance_id"]) is not None

    persisted_events = await store.session_events(session_id, limit=200)
    persisted_types = [event["type"] for event in persisted_events]
    assert "aisha.turn.finished" in persisted_types
    assert "aisha.audio.ready" not in persisted_types



def test_speech_registry_selects_kokoro_without_loading_models(tmp_path):
    settings = Settings(
        aisha_profile="mock",
        aisha_data_dir=str(tmp_path),
        aisha_tts_provider="local-kokoro",
    )
    profile = settings.load_profile()
    runtime = build_speech_runtime(settings, profile)

    status = runtime.status()
    assert status["provider"] == "kokoro-onnx"
    assert status["model"] == "kokoro-v1.0"
    assert status["loaded"] is False
    assert status["model_available"] is False
    assert status["voices_available"] is False
    assert status["artifact_count"] == 0
    assert status["persisted"] is False
