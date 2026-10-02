from __future__ import annotations

import pytest

from aisha.perception.base import BoundingBox, VisionFrame, VisionObservation
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
