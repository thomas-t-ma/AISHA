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
