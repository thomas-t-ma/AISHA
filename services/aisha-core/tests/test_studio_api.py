from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

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



def test_embodiment_and_perception_status_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        embodiment = client.get("/v1/embodiment/state")
        assert embodiment.status_code == 200
        assert embodiment.json()["activity"] == "idle"
        assert embodiment.json()["expression"] == "neutral"

        perception = client.get("/v1/perception/status")
        assert perception.status_code == 200
        assert perception.json()["enabled"] is False
        assert perception.json()["provider"] == "disabled"
        assert perception.json()["observation_count"] == 0

        latest = client.get("/v1/perception/latest")
        assert latest.status_code == 200
        assert latest.json() is None

        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["embodiment"]["activity"] == "idle"
        assert health.json()["perception"]["enabled"] is False
