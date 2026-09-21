from __future__ import annotations

import io
import wave

from fastapi.testclient import TestClient

from aisha.main import app
from aisha.voice.service import VoiceService


def _make_wave() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(1600))
    return buffer.getvalue()


def test_voice_status_does_not_require_optional_speech_packages(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        response = client.get("/v1/voice/status")
        assert response.status_code == 200
        assert response.json()["stt"]["ready"] is False
        assert response.json()["tts"]["ready"] is False


def test_local_voice_transcribe_and_synthesis_use_swappable_service(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    async def fake_transcribe(self, audio, extension):
        assert audio[:4] == b"RIFF"
        assert extension == ".wav"
        return "Hello AISHA.", 123.4

    async def fake_synthesize(self, text):
        assert text == "Hi there."
        return _make_wave(), 21.0

    monkeypatch.setattr(VoiceService, "transcribe", fake_transcribe)
    monkeypatch.setattr(VoiceService, "synthesize", fake_synthesize)

    with TestClient(app) as client:
        headers = {"Origin": "http://127.0.0.1:5173"}
        upload = client.post(
            "/v1/voice/transcribe",
            headers=headers,
            files={"file": ("speech.wav", _make_wave(), "audio/wav")},
        )
        assert upload.status_code == 200
        assert upload.json() == {"text": "Hello AISHA.", "transcribe_ms": 123.4}

        response = client.post(
            "/v1/voice/synthesize",
            headers=headers,
            json={"text": "Hi there."},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-aisha-tts-ms"] == "21.0"
        assert response.content[:4] == b"RIFF"


def test_voice_rejects_other_browser_origins(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/v1/voice/transcribe",
            headers={"Origin": "https://unrelated.example"},
            files={"file": ("speech.wav", _make_wave(), "audio/wav")},
        )
        assert response.status_code == 403

        oversized = client.post(
            "/v1/voice/synthesize",
            json={"text": "x" * 1201},
        )
        assert oversized.status_code == 422
