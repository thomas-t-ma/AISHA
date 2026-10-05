from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from aisha.perception.analyzer import MockVisionAnalyzer
from aisha.perception.base import BoundingBox, VisionFrame, VisionObservation
from aisha.perception.camera import CameraFrameDescriptor, MockCameraSource
from aisha.perception.controller import CameraController
from aisha.perception.hub import PerceptionHub
from aisha.perception.local import (
    EphemeralFrameStore,
    MediaPipeFaceAnalyzer,
    OpenCVCameraSource,
)
from aisha.perception.mock import DisabledVisionProvider, MockVisionProvider
from aisha.perception.registry import build_perception_components
from aisha.perception.runtime import PerceptionRuntime
from aisha.settings import Settings


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
    first_summary = hub.summary()
    assert first_summary.person_present is True
    assert first_summary.person_count == 1
    assert first_summary.gaze_toward_camera is False
    assert first_summary.observation_kinds == ["person"]

    await hub.poll_once()
    latest = hub.latest()
    assert latest is not None
    assert latest.frame_id == second.frame_id
    assert hub.status()["frames_seen"] == 2
    assert hub.status()["observation_count"] == 1
    second_summary = hub.summary()
    assert second_summary.person_present is False
    assert second_summary.gaze_toward_camera is True
    assert second_summary.observation_kinds == ["gaze"]

    hub.clear()
    assert hub.latest() is None
    assert hub.status()["observation_count"] == 0
    empty_summary = hub.summary()
    assert empty_summary.person_present is False
    assert empty_summary.person_count == 0
    assert empty_summary.gaze_toward_camera is False
    assert empty_summary.observation_kinds == []



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
    assert initial["privacy"]["raw_pixels_in_semantic_state"] is False
    assert initial["privacy"]["capture_persisted"] is False
    assert initial["privacy"]["raw_frame_scope"] == "none"
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



def test_perception_summary_expires_stale_scene_state():
    frame = VisionFrame(
        source_id="camera_front",
        captured_at=datetime.now(UTC) - timedelta(seconds=10),
        observations=[
            VisionObservation(
                kind="face",
                confidence=0.5,
                label="face",
                bounding_box=BoundingBox(
                    x=0.20,
                    y=0.20,
                    width=0.40,
                    height=0.50,
                ),
            )
        ],
    )
    hub = PerceptionHub(
        MockVisionProvider(),
        max_summary_age_seconds=2.0,
    )
    hub.accept(frame)

    assert hub.latest() is not None
    assert hub.status()["latest_stale"] is True
    assert hub.status()["latest_age_ms"] >= 9000
    summary = hub.summary()
    assert summary.person_present is False
    assert summary.person_count == 0
    assert summary.primary_person_x is None
    assert summary.primary_person_y is None


@pytest.mark.asyncio
async def test_perception_summary_ignores_low_confidence_observations():
    frame = VisionFrame(
        source_id="camera_front",
        observations=[
            VisionObservation(kind="person", confidence=0.49, label="person"),
            VisionObservation(kind="gaze", confidence=0.95, label="away"),
            VisionObservation(kind="gaze", confidence=0.51, label="toward_camera"),
        ],
    )
    hub = PerceptionHub(MockVisionProvider([frame]))
    await hub.poll_once()

    summary = hub.summary()
    assert summary.person_present is False
    assert summary.person_count == 0
    assert summary.gaze_toward_camera is True
    assert summary.observation_kinds == ["gaze"]



@pytest.mark.asyncio
async def test_perception_runtime_requires_camera_enablement_and_emits_structured_state():
    capture = CameraFrameDescriptor(
        frame_ref="mock://capture/1",
        source_id="camera_front",
        width=640,
        height=480,
    )
    camera_source = MockCameraSource([capture])
    hub = PerceptionHub(DisabledVisionProvider())
    controller = CameraController(camera_source, on_disable=hub.clear)
    analyzer = MockVisionAnalyzer(
        [[
            VisionObservation(kind="person", confidence=0.97, label="person"),
            VisionObservation(
                kind="gaze",
                confidence=0.88,
                label="toward_camera",
            ),
        ]]
    )
    runtime = PerceptionRuntime(controller, analyzer, hub)

    assert await runtime.step() is None
    assert hub.latest() is None
    assert runtime.status()["analysis_steps"] == 0

    controller.enable()
    frame = await runtime.step()
    assert frame is not None
    assert frame.image_ref == "mock://capture/1"
    assert frame.source_id == "camera_front"
    assert frame.width == 640
    assert frame.height == 480
    assert [item.kind for item in frame.observations] == ["person", "gaze"]

    summary = hub.summary()
    assert summary.person_present is True
    assert summary.person_count == 1
    assert summary.gaze_toward_camera is True
    assert runtime.status()["analysis_steps"] == 1

    controller.disable()
    assert hub.latest() is None
    assert hub.summary().person_present is False



@pytest.mark.asyncio
async def test_perception_runtime_start_and_stop_are_idempotent():
    source = MockCameraSource()
    hub = PerceptionHub(DisabledVisionProvider())
    controller = CameraController(source, on_disable=hub.clear)
    runtime = PerceptionRuntime(
        controller,
        MockVisionAnalyzer(),
        hub,
        poll_interval_seconds=0.05,
    )

    assert runtime.status()["running"] is False
    runtime.start()
    first_task = runtime._task
    runtime.start()
    assert runtime._task is first_task
    assert runtime.status()["running"] is True

    await asyncio.sleep(0)
    await runtime.stop()
    assert runtime.status()["running"] is False

    await runtime.stop()
    assert runtime.status()["running"] is False



class _FakeCapture:
    def __init__(self, frame):
        self.frame = frame
        self.opened = True
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return True, self.frame

    def release(self):
        self.opened = False
        self.released = True


class _FakeCV2:
    CAP_ANY = 0
    CAP_MSMF = 1400

    def __init__(self, capture):
        self.capture = capture
        self.open_calls = []

    def VideoCapture(self, index, backend):
        self.open_calls.append((index, backend))
        return self.capture


@pytest.mark.asyncio
async def test_opencv_camera_source_uses_bounded_ephemeral_store():
    frame = SimpleNamespace(shape=(480, 640, 3))
    capture = _FakeCapture(frame)
    cv2 = _FakeCV2(capture)
    store = EphemeralFrameStore(max_frames=2)
    source = OpenCVCameraSource(
        store,
        camera_index=2,
        source_id="camera_test",
        cv2_module=cv2,
    )

    source.enable()
    assert source.status()["enabled"] is True
    assert source.status()["raw_frame_scope"] == "ephemeral-provider-memory"
    descriptor = await source.capture()
    assert descriptor is not None
    assert descriptor.source_id == "camera_test"
    assert descriptor.width == 640
    assert descriptor.height == 480
    assert len(store) == 1
    assert store.pop(descriptor.frame_ref) is frame
    assert len(store) == 0

    source.disable()
    assert capture.released is True
    assert source.status()["enabled"] is False
    assert len(store) == 0


def test_ephemeral_frame_store_evicts_oldest_frame():
    store = EphemeralFrameStore(max_frames=2)
    store.put("one", object())
    store.put("two", object())
    third = object()
    store.put("three", third)

    assert store.pop("one") is None
    assert store.pop("two") is not None
    assert store.pop("three") is third


def test_mediapipe_result_translation_stays_geometry_only():
    landmarks = [
        SimpleNamespace(x=0.20, y=0.10),
        SimpleNamespace(x=0.60, y=0.80),
    ]

    class _Matrix:
        def tolist(self):
            return [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]

    result = SimpleNamespace(
        face_landmarks=[landmarks],
        face_blendshapes=[],
        facial_transformation_matrixes=[_Matrix()],
    )

    observations = MediaPipeFaceAnalyzer.observations_from_result(result)
    assert [item.kind for item in observations] == ["face", "head_pose"]

    face = observations[0]
    assert face.kind == "face"
    assert face.label == "face"
    assert face.bounding_box is not None
    assert face.bounding_box.x == pytest.approx(0.20)
    assert face.bounding_box.y == pytest.approx(0.10)
    assert face.bounding_box.width == pytest.approx(0.40)
    assert face.bounding_box.height == pytest.approx(0.70)
    assert face.confidence == 0.5
    assert face.attributes == {
        "face_index": 0,
        "confidence_source": "presence_unscored",
    }

    head_pose = observations[1]
    assert head_pose.kind == "head_pose"
    assert head_pose.label == "approximately_frontal"
    assert head_pose.confidence == 0.5
    assert head_pose.attributes["frontal_score"] == pytest.approx(1.0)
    assert head_pose.attributes["score_kind"] == "forward_axis_alignment"

    hub = PerceptionHub(DisabledVisionProvider())
    hub.accept(
        VisionFrame(
            source_id="camera_front",
            observations=observations,
        )
    )
    summary = hub.summary()
    assert summary.person_present is True
    assert summary.person_count == 1
    assert summary.primary_person_x == pytest.approx(0.40)
    assert summary.primary_person_y == pytest.approx(0.45)
    assert summary.head_approximately_frontal is True
    assert summary.head_frontal_score == pytest.approx(1.0)
    assert summary.observation_kinds == ["face", "head_pose"]


def test_head_pose_geometry_distinguishes_turned_face():
    # Third rotation-matrix column has 60 degrees of yaw from camera Z.
    matrix = [
        [0.5, 0.0, 0.8660254, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-0.8660254, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    score = MediaPipeFaceAnalyzer._frontal_score(matrix)

    assert score == pytest.approx(0.5, abs=1e-6)

    result = SimpleNamespace(
        face_landmarks=[[
            SimpleNamespace(x=0.30, y=0.20),
            SimpleNamespace(x=0.70, y=0.75),
        ]],
        face_blendshapes=[],
        facial_transformation_matrixes=[
            SimpleNamespace(tolist=lambda: matrix),
        ],
    )
    observations = MediaPipeFaceAnalyzer.observations_from_result(result)
    head_pose = next(item for item in observations if item.kind == "head_pose")
    assert head_pose.label == "turned"
    assert head_pose.attributes["frontal_score"] == pytest.approx(0.5, abs=1e-6)



def test_perception_registry_selects_local_backend_without_opening_camera(tmp_path):
    settings = Settings(
        aisha_profile="mock",
        aisha_data_dir=str(tmp_path),
        aisha_vision_provider="local-mediapipe",
    )
    profile = settings.load_profile()
    components = build_perception_components(settings, profile)

    camera_status = components.camera.status()
    analyzer_status = components.runtime.analyzer.status()

    assert camera_status["source"] == "opencv-camera"
    assert camera_status["enabled"] is False
    assert camera_status["raw_frame_scope"] == "ephemeral-provider-memory"
    assert analyzer_status["analyzer"] == "mediapipe-face-landmarker"
    assert analyzer_status["enabled"] is False
    assert analyzer_status["model_path"].endswith("models/face_landmarker.task")
    assert components.hub.latest() is None
