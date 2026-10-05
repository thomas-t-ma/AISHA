from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aisha.perception.base import PerceptionSummary, VisionFrame, VisionProvider


class PerceptionHub:
    """Owns the latest structured perception state, not camera pixels."""

    def __init__(
        self,
        provider: VisionProvider,
        *,
        max_summary_age_seconds: float = 2.0,
    ) -> None:
        self.provider = provider
        self.max_summary_age_seconds = max(0.1, max_summary_age_seconds)
        self._latest: VisionFrame | None = None
        self._frames_seen = 0

    def accept(self, frame: VisionFrame) -> VisionFrame:
        self._latest = frame.model_copy(deep=True)
        self._frames_seen += 1
        return frame

    async def poll_once(self) -> VisionFrame | None:
        frame = await self.provider.observe()
        if frame is not None:
            self.accept(frame)
        return frame

    def latest(self) -> VisionFrame | None:
        return None if self._latest is None else self._latest.model_copy(deep=True)

    def clear(self) -> None:
        self._latest = None

    def summary(self) -> PerceptionSummary:
        latest = self._latest
        if latest is None:
            return PerceptionSummary()

        age_seconds = max(
            0.0,
            (datetime.now(UTC) - latest.captured_at.astimezone(UTC)).total_seconds(),
        )
        if age_seconds > self.max_summary_age_seconds:
            return PerceptionSummary()

        reliable = [
            observation
            for observation in latest.observations
            if observation.confidence >= 0.5
        ]
        faces = [
            observation
            for observation in reliable
            if observation.kind == "face"
        ]
        detected_people = [
            observation
            for observation in reliable
            if observation.kind == "person"
        ]
        # Prefer face detections when present so a future object detector does
        # not double-count the same visible person as both "person" and "face".
        people = faces if faces else detected_people
        gaze_toward_camera = any(
            observation.kind == "gaze"
            and observation.label == "toward_camera"
            for observation in reliable
        )

        primary_person = None
        boxed_people = [
            observation
            for observation in people
            if observation.bounding_box is not None
        ]
        if boxed_people:
            primary_person = max(
                boxed_people,
                key=lambda observation: (
                    observation.bounding_box.width
                    * observation.bounding_box.height
                ),
            )

        primary_person_x = None
        primary_person_y = None
        if primary_person is not None and primary_person.bounding_box is not None:
            box = primary_person.bounding_box
            primary_person_x = min(1.0, max(0.0, box.x + box.width / 2.0))
            primary_person_y = min(1.0, max(0.0, box.y + box.height / 2.0))

        head_pose = [
            observation
            for observation in reliable
            if observation.kind == "head_pose"
        ]
        frontal_scores = [
            float(observation.attributes["frontal_score"])
            for observation in head_pose
            if isinstance(observation.attributes.get("frontal_score"), (int, float))
        ]
        head_frontal_score = max(frontal_scores) if frontal_scores else None
        head_approximately_frontal = any(
            observation.label == "approximately_frontal"
            for observation in head_pose
        )

        return PerceptionSummary(
            frame_id=latest.frame_id,
            source_id=latest.source_id,
            person_present=bool(people),
            person_count=len(people),
            gaze_toward_camera=gaze_toward_camera,
            head_approximately_frontal=head_approximately_frontal,
            head_frontal_score=head_frontal_score,
            primary_person_x=primary_person_x,
            primary_person_y=primary_person_y,
            observation_kinds=sorted({observation.kind for observation in reliable}),
            captured_at=latest.captured_at,
        )

    def status(self) -> dict[str, Any]:
        provider_status = self.provider.status()
        latest = self._latest
        latest_age_ms = None
        latest_stale = False
        if latest is not None:
            latest_age_ms = round(
                max(
                    0.0,
                    (
                        datetime.now(UTC)
                        - latest.captured_at.astimezone(UTC)
                    ).total_seconds()
                    * 1000,
                ),
                3,
            )
            latest_stale = latest_age_ms > self.max_summary_age_seconds * 1000

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
            "latest_age_ms": latest_age_ms,
            "latest_stale": latest_stale,
            "summary_max_age_ms": round(self.max_summary_age_seconds * 1000, 3),
            "checked_at": datetime.now(UTC).isoformat(),
        }
