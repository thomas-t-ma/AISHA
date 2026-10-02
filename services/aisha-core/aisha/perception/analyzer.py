from __future__ import annotations

from collections import deque
from typing import Any, Protocol

from aisha.perception.base import VisionFrame, VisionObservation
from aisha.perception.camera import CameraFrameDescriptor


class VisionAnalyzer(Protocol):
    name: str

    async def analyze(
        self,
        capture: CameraFrameDescriptor,
    ) -> VisionFrame | None: ...

    def status(self) -> dict[str, Any]: ...


class DisabledVisionAnalyzer:
    name = "disabled-analyzer"

    async def analyze(
        self,
        capture: CameraFrameDescriptor,
    ) -> VisionFrame | None:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "analyzer": self.name,
        }


class MockVisionAnalyzer:
    """Queue-backed analyzer that produces structured observations only."""

    name = "mock-analyzer"

    def __init__(
        self,
        batches: list[list[VisionObservation]] | None = None,
    ) -> None:
        self._batches = deque(batches or [])

    async def analyze(
        self,
        capture: CameraFrameDescriptor,
    ) -> VisionFrame | None:
        if not self._batches:
            return None
        observations = self._batches.popleft()
        return VisionFrame(
            source_id=capture.source_id,
            captured_at=capture.captured_at,
            width=capture.width,
            height=capture.height,
            observations=[
                observation.model_copy(deep=True)
                for observation in observations
            ],
            image_ref=capture.frame_ref,
        )

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "analyzer": self.name,
            "queued_batches": len(self._batches),
        }
