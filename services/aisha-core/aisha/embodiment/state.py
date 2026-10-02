from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


EmbodimentActivity = Literal["idle", "listening", "thinking", "speaking"]
ExpressionIntent = Literal["neutral", "attentive", "focused", "engaged"]


class EmbodimentState(BaseModel):
    """Renderer-independent semantic state for AISHA's visible body."""

    sequence: int = Field(ge=0)
    activity: EmbodimentActivity
    expression: ExpressionIntent
    intensity: float = Field(ge=0.0, le=1.0)
    updated_at: datetime


_ACTIVITY_STYLE: dict[EmbodimentActivity, tuple[ExpressionIntent, float]] = {
    "idle": ("neutral", 0.20),
    "listening": ("attentive", 0.55),
    "thinking": ("focused", 0.50),
    "speaking": ("engaged", 0.65),
}


class EmbodimentDirector:
    """Deterministic lifecycle-to-expression mapper.

    Core emits semantic intent only. A future Unity/Three.js/Live2D renderer
    decides how these intents map to blendshapes, animation clips, gaze, and
    procedural motion.
    """

    def __init__(self) -> None:
        self._state = EmbodimentState(
            sequence=0,
            activity="idle",
            expression="neutral",
            intensity=0.20,
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
            updated_at=datetime.now(UTC),
        )
        return self.snapshot()
