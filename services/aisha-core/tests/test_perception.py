from __future__ import annotations

import pytest

from aisha.perception.base import BoundingBox, VisionFrame, VisionObservation
from aisha.perception.camera import CameraFrameDescriptor, MockCameraSource
from aisha.perception.controller import CameraController
from aisha.perception.hub import PerceptionHub
from aisha.perception.mock import DisabledVisionProvider, MockVisionProvider


@pytest.mark.asyncio
async def test_disabled_vision_provider_has_no_observation():
    provider = DisabledVisionProvider()
    assert provider.status() == {"enabled": False, "provider": "disabled"}
    assert await provider.observe() is None


@pytest.mark.asyncio
async def test_mock_vision_provider_returns_structured_observations_in_order():
    frame = VisionFrame(
        source_id="camera_front",
        width=1920,
        height=1080,
        observations=[
            VisionObservation(
                kind="person",
                confidence=0.98,
                label="person",
                bounding_box=BoundingBox(
                    x=0.25,
                    y=0.10,
                    width=0.40,
                    height=0.80,
                ),
                attributes={"track_id": "person_1"},
            ),
            VisionObservation(
                kind="gaze",
                confidence=0.82,
                label="toward_camera",
                attributes={"subject_track_id": "person_1"},
            ),
        ],
        image_ref="provider-local://camera_front/frame_1",
    )
    provider = MockVisionProvider([frame])

    status = provider.status()
    assert status["enabled"] is True
    assert status["queued_frames"] == 1

    observed = await provider.observe()
    assert observed is not None
    assert observed.source_id == "camera_front"
    assert [item.kind for item in observed.observations] == ["person", "gaze"]
    assert observed.observations[0].bounding_box is not None
    assert observed.observations[0].bounding_box.width == 0.40
    assert observed.image_ref == "provider-local://camera_front/frame_1"

    assert await provider.observe() is None
    assert provider.status()["queued_frames"] == 0


def test_vision_contract_excludes_raw_image_bytes():
    fields = VisionFrame.model_fields
    assert "image" not in fields
    assert "bytes" not in fields
    assert "image_ref" in fields



@pytest.mark.asyncio
async def test_perception_hub_tracks_latest_structured_frame():
    first = VisionFrame(
        source_id="camera_front",
        observations=[
            VisionObservation(kind="person", confidence=0.9, label="person")
        ],
    )
    second = VisionFrame(
        source_id="camera_front",
        observations=[
            VisionObservation(kind="gaze", confidence=0.8, label="toward_camera")
        ],
    )
    provider = MockVisionProvider([first, second])
    hub = PerceptionHub(provider)

    assert hub.latest() is None
    await hub.poll_once()
    assert hub.latest() is not None
    assert hub.latest().frame_id == first.frame_id
    assert hub.status()["frames_seen"] == 1

    await hub.poll_once()
    latest = hub.latest()
    assert latest is not None
    assert latest.frame_id == second.frame_id
    assert hub.status()["frames_seen"] == 2
    assert hub.status()["observation_count"] == 1

    hub.clear()
    assert hub.latest() is None
    assert hub.status()["observation_count"] == 0



@pytest.mark.asyncio
async def test_camera_controller_is_explicitly_disabled_until_enabled():
    frame = CameraFrameDescriptor(
        frame_ref="mock://frame/1",
        source_id="camera_front",
        width=1280,
        height=720,
    )
    source = MockCameraSource([frame])
    cleared = []
    controller = CameraController(source, on_disable=lambda: cleared.append(True))

    initial = controller.status()
    assert initial["enabled"] is False
    assert initial["privacy"]["camera_active"] is False
    assert initial["privacy"]["raw_pixels_in_core"] is False
    assert initial["privacy"]["capture_persisted"] is False
    assert await controller.capture_once() is None

    enabled = controller.enable()
    assert enabled["enabled"] is True
    assert enabled["privacy"]["camera_active"] is True

    captured = await controller.capture_once()
    assert captured is not None
    assert captured.frame_ref == "mock://frame/1"
    assert controller.status()["captures_seen"] == 1
    assert controller.latest() is not None

    disabled = controller.disable()
    assert disabled["enabled"] is False
    assert disabled["privacy"]["camera_active"] is False
    assert controller.latest() is None
    assert cleared == [True]


def test_camera_descriptor_excludes_raw_pixels():
    fields = CameraFrameDescriptor.model_fields
    assert "image" not in fields
    assert "bytes" not in fields
    assert "pixels" not in fields
    assert "frame_ref" in fields
