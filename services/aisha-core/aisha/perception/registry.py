from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aisha.perception.analyzer import DisabledVisionAnalyzer
from aisha.perception.camera import DisabledCameraSource
from aisha.perception.controller import CameraController
from aisha.perception.hub import PerceptionHub
from aisha.perception.local import (
    EphemeralFrameStore,
    MediaPipeFaceAnalyzer,
    OpenCVCameraSource,
)
from aisha.perception.mock import DisabledVisionProvider
from aisha.perception.runtime import PerceptionRuntime
from aisha.settings import RuntimeProfile, Settings


@dataclass
class PerceptionComponents:
    hub: PerceptionHub
    camera: CameraController
    runtime: PerceptionRuntime


def _model_path(settings: Settings, profile: RuntimeProfile) -> Path:
    configured = profile.vision.model_path
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = settings.data_dir / path
        return path.resolve()
    return (settings.data_dir / "models" / "face_landmarker.task").resolve()


def build_perception_components(
    settings: Settings,
    profile: RuntimeProfile,
) -> PerceptionComponents:
    hub = PerceptionHub(DisabledVisionProvider())

    if profile.vision.provider == "local-mediapipe":
        frame_store = EphemeralFrameStore(max_frames=3)
        source = OpenCVCameraSource(
            frame_store,
            camera_index=profile.vision.camera_index,
            source_id=profile.vision.source_id,
        )
        camera = CameraController(source, on_disable=hub.clear)
        analyzer = MediaPipeFaceAnalyzer(
            frame_store,
            _model_path(settings, profile),
            num_faces=profile.vision.num_faces,
        )
        runtime = PerceptionRuntime(
            camera,
            analyzer,
            hub,
            poll_interval_seconds=profile.vision.poll_interval_seconds,
        )
        return PerceptionComponents(hub=hub, camera=camera, runtime=runtime)

    camera = CameraController(
        DisabledCameraSource(),
        on_disable=hub.clear,
    )
    runtime = PerceptionRuntime(
        camera,
        DisabledVisionAnalyzer(),
        hub,
        poll_interval_seconds=profile.vision.poll_interval_seconds,
    )
    return PerceptionComponents(hub=hub, camera=camera, runtime=runtime)
