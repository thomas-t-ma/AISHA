from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field


class CameraFrameDescriptor(BaseModel):
    """Opaque reference to one captured frame.

    Raw pixels are intentionally excluded. The capture backend owns image bytes
    and may expose them only through a provider-local reference.
    """

    frame_ref: str = Field(default_factory=lambda: f"capture_{uuid4().hex}")
    source_id: str = Field(min_length=1, max_length=120)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class CameraSource(Protocol):
    name: str

    def enable(self) -> None: ...

    def disable(self) -> None: ...

    async def capture(self) -> CameraFrameDescriptor | None: ...

    def status(self) -> dict[str, Any]: ...


class DisabledCameraSource:
    name = "disabled-camera"

    def enable(self) -> None:
        return None

    def disable(self) -> None:
        return None

    async def capture(self) -> CameraFrameDescriptor | None:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "available": False,
            "enabled": False,
            "source": self.name,
        }


class MockCameraSource:
    """Queue-backed capture source for deterministic privacy/lifecycle tests."""

    name = "mock-camera"

    def __init__(self, frames: list[CameraFrameDescriptor] | None = None) -> None:
        self._frames = deque(frames or [])
        self._enabled = False

    def push(self, frame: CameraFrameDescriptor) -> None:
        self._frames.append(frame)

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    async def capture(self) -> CameraFrameDescriptor | None:
        if not self._enabled or not self._frames:
            return None
        return self._frames.popleft()

    def status(self) -> dict[str, Any]:
        return {
            "available": True,
            "enabled": self._enabled,
            "source": self.name,
            "queued_frames": len(self._frames),
        }
