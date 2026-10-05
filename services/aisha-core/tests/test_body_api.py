from __future__ import annotations

from fastapi.testclient import TestClient

from aisha.api.routes import router
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
        "affect_expires_at",
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



def test_body_stream_route_is_registered():
    paths = {route.path for route in router.routes}
    assert "/v1/body/state" in paths
    assert "/v1/body/stream" in paths
