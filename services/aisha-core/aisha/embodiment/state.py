from __future__ import annotations

from datetime import UTC, datetime
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
    updated_at: datetime


_ACTIVITY_STYLE: dict[EmbodimentActivity, tuple[ExpressionIntent, float]] = {
    "idle": ("neutral", 0.20),
    "listening": ("attentive", 0.55),
    "thinking": ("focused", 0.50),
    "speaking": ("engaged", 0.65),
}


class EmbodimentDirector:
    """Deterministic semantic-state director."""

    def __init__(self) -> None:
        self._state = EmbodimentState(
            sequence=0,
            activity="idle",
            expression="neutral",
            intensity=0.20,
            affect="neutral",
            affect_intensity=0.0,
            updated_at=datetime.now(UTC),
        )

    def snapshot(self) -> EmbodimentState:
        return self._state.model_copy(deep=True)

    def transition(self, activity: EmbodimentActivity) -> EmbodimentState:
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
            updated_at=datetime.now(UTC),
        )
        return self.snapshot()

    def set_affect(
        self,
        affect: AffectIntent,
        *,
        intensity: float = 0.5,
    ) -> EmbodimentState:
        normalized_intensity = (
            0.0
            if affect == "neutral"
            else max(0.0, min(1.0, intensity))
        )
        if (
            affect == self._state.affect
            and normalized_intensity == self._state.affect_intensity
        ):
            return self.snapshot()

        self._state = self._state.model_copy(
            update={
                "sequence": self._state.sequence + 1,
                "affect": affect,
                "affect_intensity": normalized_intensity,
                "updated_at": datetime.now(UTC),
            }
        )
        return self.snapshot()

    def clear_affect(self) -> EmbodimentState:
        return self.set_affect("neutral", intensity=0.0)
