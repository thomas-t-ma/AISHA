from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

EmbodimentActivity = Literal["idle", "listening", "thinking", "speaking"]
ExpressionIntent = Literal["neutral", "attentive", "focused", "engaged"]
AffectIntent = Literal[
    "neutral",
    "warm",
    "amused",
    "curious",
    "concerned",
    "surprised",
]


class EmbodimentState(BaseModel):
    """Renderer-independent semantic state for AISHA's visible body.

    Activity describes what AISHA is doing. Affect describes the emotional
    coloring applied independently of that activity. The expression and
    intensity fields remain activity-derived compatibility hints for current
    renderers while the renderer contract migrates to explicit blending.
    """

    sequence: int = Field(ge=0)
    activity: EmbodimentActivity
    expression: ExpressionIntent
    intensity: float = Field(ge=0.0, le=1.0)
    affect: AffectIntent = "neutral"
    affect_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    affect_expires_at: datetime | None = None
    updated_at: datetime


_ACTIVITY_STYLE: dict[EmbodimentActivity, tuple[ExpressionIntent, float]] = {
    "idle": ("neutral", 0.20),
    "listening": ("attentive", 0.55),
    "thinking": ("focused", 0.50),
    "speaking": ("engaged", 0.65),
}


class EmbodimentDirector:
    """Deterministic semantic-state director with bounded affect cues."""

    def __init__(
        self,
        *,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
        now = self._now()
        self._state = EmbodimentState(
            sequence=0,
            activity="idle",
            expression="neutral",
            intensity=0.20,
            affect="neutral",
            affect_intensity=0.0,
            affect_expires_at=None,
            updated_at=now,
        )

    def _now(self) -> datetime:
        now = self._now_provider()
        if now.tzinfo is None:
            return now.replace(tzinfo=UTC)
        return now.astimezone(UTC)

    def _expire_affect_if_needed(self) -> None:
        expires_at = self._state.affect_expires_at
        if (
            self._state.affect == "neutral"
            or expires_at is None
            or self._now() < expires_at
        ):
            return

        self._state = self._state.model_copy(
            update={
                "sequence": self._state.sequence + 1,
                "affect": "neutral",
                "affect_intensity": 0.0,
                "affect_expires_at": None,
                "updated_at": self._now(),
            }
        )

    def snapshot(self) -> EmbodimentState:
        self._expire_affect_if_needed()
        return self._state.model_copy(deep=True)

    def transition(self, activity: EmbodimentActivity) -> EmbodimentState:
        self._expire_affect_if_needed()
        if activity == self._state.activity:
            return self.snapshot()

        expression, intensity = _ACTIVITY_STYLE[activity]
        self._state = EmbodimentState(
            sequence=self._state.sequence + 1,
            activity=activity,
            expression=expression,
            intensity=intensity,
            affect=self._state.affect,
            affect_intensity=self._state.affect_intensity,
            affect_expires_at=self._state.affect_expires_at,
            updated_at=self._now(),
        )
        return self.snapshot()

    def set_affect(
        self,
        affect: AffectIntent,
        *,
        intensity: float = 0.5,
        duration_seconds: float | None = None,
    ) -> EmbodimentState:
        self._expire_affect_if_needed()
        now = self._now()
        normalized_intensity = (
            0.0
            if affect == "neutral"
            else max(0.0, min(1.0, intensity))
        )
        expires_at = None
        if affect != "neutral" and duration_seconds is not None:
            bounded_duration = max(0.25, min(30.0, duration_seconds))
            expires_at = now + timedelta(seconds=bounded_duration)

        if (
            affect == self._state.affect
            and normalized_intensity == self._state.affect_intensity
            and expires_at == self._state.affect_expires_at
        ):
            return self.snapshot()

        self._state = self._state.model_copy(
            update={
                "sequence": self._state.sequence + 1,
                "affect": affect,
                "affect_intensity": normalized_intensity,
                "affect_expires_at": expires_at,
                "updated_at": now,
            }
        )
        return self.snapshot()

    def clear_affect(self) -> EmbodimentState:
        return self.set_affect("neutral", intensity=0.0)
