from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aisha.perception.analyzer import VisionAnalyzer
from aisha.perception.base import VisionFrame
from aisha.perception.controller import CameraController
from aisha.perception.hub import PerceptionHub


class PerceptionRuntime:
    """Connects capture, analysis, and structured perception state."""

    def __init__(
        self,
        camera: CameraController,
        analyzer: VisionAnalyzer,
        hub: PerceptionHub,
    ) -> None:
        self.camera = camera
        self.analyzer = analyzer
        self.hub = hub
        self._analysis_steps = 0

    async def step(self) -> VisionFrame | None:
        capture = await self.camera.capture_once()
        if capture is None:
            return None

        frame = await self.analyzer.analyze(capture)
        if frame is None:
            return None

        self._analysis_steps += 1
        return self.hub.accept(frame)

    def status(self) -> dict[str, Any]:
        return {
            "camera": self.camera.status(),
            "analyzer": self.analyzer.status(),
            "hub": self.hub.status(),
            "analysis_steps": self._analysis_steps,
            "checked_at": datetime.now(UTC).isoformat(),
        }
