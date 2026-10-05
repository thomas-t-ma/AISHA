from __future__ import annotations

from fastapi.testclient import TestClient

from aisha.main import app


def test_body_state_exposes_only_renderer_safe_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        response = client.get("/v1/body/state")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"embodiment", "perception"}

    assert set(payload["embodiment"]) == {
        "sequence",
        "activity",
        "expression",
        "intensity",
        "affect",
        "affect_intensity",
        "updated_at",
    }
    assert set(payload["perception"]) == {
        "person_present",
        "person_count",
        "gaze_toward_camera",
        "head_approximately_frontal",
        "primary_person_x",
        "primary_person_y",
        "visible_objects",
    }

    serialized = response.text.lower()
    for forbidden in (
        "provider",
        "model",
        "profile",
        "frame_id",
        "source_id",
        "captured_at",
        "observation_kinds",
        "head_frontal_score",
        "memory",
        "camera_active",
    ):
        assert forbidden not in serialized



def test_body_stream_is_renderer_only_event_source(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))

    with TestClient(app) as client:
        with client.stream("GET", "/v1/body/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["cache-control"] == "no-store"
            first = next(response.iter_lines())

    assert first.startswith("data: ")
    serialized = first.lower()
    for forbidden in ("provider", "model", "memory", "frame_id", "source_id"):
        assert forbidden not in serialized
