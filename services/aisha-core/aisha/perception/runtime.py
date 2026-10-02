from __future__ import annotations

import asyncio
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
        *,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        self.camera = camera
        self.analyzer = analyzer
        self.hub = hub
        self.poll_interval_seconds = max(0.05, poll_interval_seconds)
        self._analysis_steps = 0
        self._task: asyncio.Task[None] | None = None
        self._last_error: str | None = None

    async def step(self) -> VisionFrame | None:
        capture = await self.camera.capture_once()
        if capture is None:
            return None

        frame = await self.analyzer.analyze(capture)
        if frame is None:
            return None

        self._analysis_steps += 1
        self._last_error = None
        return self.hub.accept(frame)

    async def _run(self) -> None:
        try:
            while True:
                if self.camera.status().get("enabled", False):
                    try:
                        await self.step()
                    except Exception as exc:  # provider boundary; keep runtime alive
                        self._last_error = str(exc)
                await asyncio.sleep(self.poll_interval_seconds)
        except asyncio.CancelledError:
            raise

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(
            self._run(),
            name="aisha-perception-runtime",
        )

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def status(self) -> dict[str, Any]:
        task = self._task
        return {
            "camera": self.camera.status(),
            "analyzer": self.analyzer.status(),
            "hub": self.hub.status(),
            "analysis_steps": self._analysis_steps,
            "running": task is not None and not task.done(),
            "poll_interval_seconds": self.poll_interval_seconds,
            "last_error": self._last_error,
            "checked_at": datetime.now(UTC).isoformat(),
        }
