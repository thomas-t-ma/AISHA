from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aisha.perception.base import VisionFrame, VisionProvider


class PerceptionHub:
    """Owns the latest structured perception state, not camera pixels."""

    def __init__(self, provider: VisionProvider) -> None:
        self.provider = provider
        self._latest: VisionFrame | None = None
        self._frames_seen = 0

    async def poll_once(self) -> VisionFrame | None:
        frame = await self.provider.observe()
        if frame is not None:
            self._latest = frame.model_copy(deep=True)
            self._frames_seen += 1
        return frame

    def latest(self) -> VisionFrame | None:
        return None if self._latest is None else self._latest.model_copy(deep=True)

    def clear(self) -> None:
        self._latest = None

    def status(self) -> dict[str, Any]:
        provider_status = self.provider.status()
        latest = self._latest
        return {
            **provider_status,
            "frames_seen": self._frames_seen,
            "latest_frame_id": None if latest is None else latest.frame_id,
            "latest_source_id": None if latest is None else latest.source_id,
            "latest_captured_at": (
                None
                if latest is None
                else latest.captured_at.astimezone(UTC).isoformat()
            ),
            "observation_count": (
                0 if latest is None else len(latest.observations)
            ),
            "checked_at": datetime.now(UTC).isoformat(),
        }
