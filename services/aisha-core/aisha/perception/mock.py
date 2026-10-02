from __future__ import annotations

from collections import deque

from aisha.perception.base import VisionFrame


class DisabledVisionProvider:
    name = "disabled"

    async def observe(self) -> VisionFrame | None:
        return None

    def status(self) -> dict:
        return {"enabled": False, "provider": self.name}


class MockVisionProvider:
    """Deterministic queue-backed provider for Core and CI tests."""

    name = "mock-vision"

    def __init__(self, frames: list[VisionFrame] | None = None) -> None:
        self._frames = deque(frames or [])

    def push(self, frame: VisionFrame) -> None:
        self._frames.append(frame)

    async def observe(self) -> VisionFrame | None:
        return self._frames.popleft() if self._frames else None

    def status(self) -> dict:
        return {
            "enabled": True,
            "provider": self.name,
            "queued_frames": len(self._frames),
        }
