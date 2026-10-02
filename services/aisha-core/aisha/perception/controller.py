from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from aisha.perception.camera import CameraFrameDescriptor, CameraSource


class CameraController:
    """Owns explicit camera enablement and transient capture metadata."""

    def __init__(
        self,
        source: CameraSource,
        *,
        on_disable: Callable[[], None] | None = None,
    ) -> None:
        self.source = source
        self._on_disable = on_disable
        self._latest: CameraFrameDescriptor | None = None
        self._captures_seen = 0

    def enable(self) -> dict[str, Any]:
        self.source.enable()
        return self.status()

    def disable(self) -> dict[str, Any]:
        self.source.disable()
        self._latest = None
        if self._on_disable is not None:
            self._on_disable()
        return self.status()

    async def capture_once(self) -> CameraFrameDescriptor | None:
        if not self.source.status().get("enabled", False):
            return None
        frame = await self.source.capture()
        if frame is not None:
            self._latest = frame.model_copy(deep=True)
            self._captures_seen += 1
        return frame

    def latest(self) -> CameraFrameDescriptor | None:
        return None if self._latest is None else self._latest.model_copy(deep=True)

    def status(self) -> dict[str, Any]:
        source_status = self.source.status()
        latest = self._latest
        return {
            **source_status,
            "captures_seen": self._captures_seen,
            "latest_capture_ref": None if latest is None else latest.frame_ref,
            "latest_source_id": None if latest is None else latest.source_id,
            "latest_captured_at": (
                None
                if latest is None
                else latest.captured_at.astimezone(UTC).isoformat()
            ),
            "privacy": {
                "camera_active": bool(source_status.get("enabled", False)),
                "raw_pixels_in_semantic_state": False,
                "capture_persisted": False,
                "raw_frame_scope": source_status.get("raw_frame_scope", "none"),
            },
            "checked_at": datetime.now(UTC).isoformat(),
        }
