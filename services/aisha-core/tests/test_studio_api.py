from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from aisha.audio.mock import MockTTSProvider
from aisha.audio.runtime import SpeechRuntime
from aisha.audio.store import EphemeralAudioStore
from aisha.contracts.turns import Message
from aisha.main import app


def test_studio_restores_committed_session_history(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        response = client.post("/v1/sessions")
        assert response.status_code == 200
        session_id = response.json()["session_id"]

        empty = client.get(f"/v1/sessions/{session_id}/messages")
        assert empty.status_code == 200
        assert empty.json() == []

        store = client.app.state.aisha["store"]
        asyncio.run(store.add_message(session_id, Message(role="user", text="hello")))
        asyncio.run(store.add_message(session_id, Message(role="assistant", text="hi!")))
        asyncio.run(
            store.add_message(
                session_id,
                Message(role="assistant", text="discard this partial"),
                status="cancelled",
            )
        )

        history = client.get(f"/v1/sessions/{session_id}/messages")
        assert history.status_code == 200
        assert [(item["role"], item["text"]) for item in history.json()] == [
            ("user", "hello"),
            ("assistant", "hi!"),
        ]


def test_studio_browser_websocket_streams_same_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        session_id = client.post("/v1/sessions").json()["session_id"]
        events = []
        with client.websocket_connect(
            f"/v1/ws/{session_id}",
            headers={"origin": "http://127.0.0.1:5173"},
        ) as socket:
            socket.send_json({"type": "aisha.user.text", "payload": {"text": "hello"}})
            while True:
                event = socket.receive_json()
                events.append(event["type"])
                if event["type"] == "aisha.turn.finished":
                    break

        assert events[0] == "aisha.turn.started"
        assert "aisha.assistant.text_delta" in events
        assert events[-1] == "aisha.turn.finished"


def test_session_memory_mode_defaults_normal_and_can_be_toggled(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        session_id = client.post("/v1/sessions").json()["session_id"]

        initial = client.get(f"/v1/sessions/{session_id}")
        assert initial.status_code == 200
        assert initial.json() == {
            "session_id": session_id,
            "memory_mode": "normal",
        }

        changed = client.patch(
            f"/v1/sessions/{session_id}",
            json={"memory_mode": "test"},
            headers={"origin": "http://127.0.0.1:5173"},
        )
        assert changed.status_code == 200
        assert changed.json()["memory_mode"] == "test"

        restored = client.patch(
            f"/v1/sessions/{session_id}",
            json={"memory_mode": "normal"},
        )
        assert restored.status_code == 200
        assert restored.json()["memory_mode"] == "normal"

        invalid = client.patch(
            f"/v1/sessions/{session_id}",
            json={"memory_mode": "fiction"},
        )
        assert invalid.status_code == 422

        rejected_origin = client.patch(
            f"/v1/sessions/{session_id}",
            json={"memory_mode": "test"},
            headers={"origin": "https://unrelated.example"},
        )
        assert rejected_origin.status_code == 403





def test_transient_audio_artifact_can_be_served_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        runtime = SpeechRuntime(
            MockTTSProvider(),
            EphemeralAudioStore(ttl_seconds=120.0),
        )
        client.app.state.aisha["speech_runtime"] = runtime
        artifact = asyncio.run(runtime.synthesize("hello"))
        assert artifact is not None

        response = client.get(f"/v1/audio/{artifact.utterance_id}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.headers["cache-control"] == "no-store"
        assert response.content.startswith(b"RIFF")
        assert len(response.content) == artifact.byte_length


def test_embodiment_and_perception_status_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        speech = client.get("/v1/audio/status")
        assert speech.status_code == 200
        assert speech.json()["enabled"] is False
        assert speech.json()["provider"] == "disabled"
        assert speech.json()["artifact_count"] == 0
        assert speech.json()["persisted"] is False

        missing_audio = client.get("/v1/audio/utt_missing")
        assert missing_audio.status_code == 404

        embodiment = client.get("/v1/embodiment/state")
        assert embodiment.status_code == 200
        assert embodiment.json()["activity"] == "idle"
        assert embodiment.json()["expression"] == "neutral"
        assert embodiment.json()["affect"] == "neutral"
        assert embodiment.json()["affect_intensity"] == 0.0

        affected = client.post(
            "/v1/embodiment/affect",
            json={"affect": "amused", "intensity": 0.35},
            headers={"origin": "http://127.0.0.1:5173"},
        )
        assert affected.status_code == 200
        assert affected.json()["activity"] == "idle"
        assert affected.json()["affect"] == "amused"
        assert affected.json()["affect_intensity"] == 0.35

        rejected_affect = client.post(
            "/v1/embodiment/affect",
            json={"affect": "amused", "intensity": 0.5},
            headers={"origin": "https://unrelated.example"},
        )
        assert rejected_affect.status_code == 403

        pulsed = client.post(
            "/v1/embodiment/affect",
            json={
                "affect": "curious",
                "intensity": 0.45,
                "duration_seconds": 4.0,
            },
            headers={"origin": "http://127.0.0.1:5173"},
        )
        assert pulsed.status_code == 200
        assert pulsed.json()["affect"] == "curious"
        assert pulsed.json()["affect_expires_at"] is not None

        too_short = client.post(
            "/v1/embodiment/affect",
            json={
                "affect": "warm",
                "intensity": 0.5,
                "duration_seconds": 0.1,
            },
            headers={"origin": "http://127.0.0.1:5173"},
        )
        assert too_short.status_code == 422

        too_long = client.post(
            "/v1/embodiment/affect",
            json={
                "affect": "warm",
                "intensity": 0.5,
                "duration_seconds": 31.0,
            },
            headers={"origin": "http://127.0.0.1:5173"},
        )
        assert too_long.status_code == 422

        perception = client.get("/v1/perception/status")
        assert perception.status_code == 200
        assert perception.json()["enabled"] is False
        assert perception.json()["provider"] == "disabled"
        assert perception.json()["observation_count"] == 0

        camera = client.get("/v1/perception/camera")
        assert camera.status_code == 200
        assert camera.json()["available"] is False
        assert camera.json()["enabled"] is False
        assert camera.json()["privacy"]["camera_active"] is False
        assert camera.json()["privacy"]["raw_pixels_in_semantic_state"] is False
        assert camera.json()["privacy"]["capture_persisted"] is False
        assert camera.json()["privacy"]["raw_frame_scope"] == "none"

        enable_camera = client.post(
            "/v1/perception/camera",
            json={"enabled": True},
            headers={"origin": "http://127.0.0.1:5173"},
        )
        assert enable_camera.status_code == 200
        assert enable_camera.json()["enabled"] is False

        rejected_camera = client.post(
            "/v1/perception/camera",
            json={"enabled": True},
            headers={"origin": "https://unrelated.example"},
        )
        assert rejected_camera.status_code == 403

        latest = client.get("/v1/perception/latest")
        assert latest.status_code == 200
        assert latest.json() is None

        summary = client.get("/v1/perception/summary")
        assert summary.status_code == 200
        assert summary.json()["person_present"] is False
        assert summary.json()["person_count"] == 0
        assert summary.json()["gaze_toward_camera"] is False
        assert summary.json()["observation_kinds"] == []

        runtime = client.get("/v1/perception/runtime")
        assert runtime.status_code == 200
        assert runtime.json()["camera"]["enabled"] is False
        assert runtime.json()["analyzer"]["enabled"] is False
        assert runtime.json()["analysis_steps"] == 0

        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["speech"]["enabled"] is False
        assert health.json()["speech"]["provider"] == "disabled"
        assert health.json()["embodiment"]["activity"] == "idle"
        assert health.json()["embodiment"]["affect"] == "curious"
        assert health.json()["embodiment"]["affect_expires_at"] is not None
        assert health.json()["perception"]["enabled"] is False
        assert health.json()["perception"]["camera"]["enabled"] is False
        assert health.json()["perception"]["camera"]["privacy"]["camera_active"] is False
        assert health.json()["perception"]["analyzer"]["enabled"] is False
        assert health.json()["perception"]["analysis_steps"] == 0
