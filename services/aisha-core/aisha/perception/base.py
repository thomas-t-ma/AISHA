from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized image-space bounding box."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class VisionObservation(BaseModel):
    """Structured perception output; Core never needs raw pixels."""

    observation_id: str = Field(default_factory=lambda: f"obs_{uuid4().hex}")
    kind: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    label: str | None = Field(default=None, max_length=160)
    bounding_box: BoundingBox | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class VisionFrame(BaseModel):
    """One provider-produced set of observations for a camera frame."""

    frame_id: str = Field(default_factory=lambda: f"frame_{uuid4().hex}")
    source_id: str = Field(min_length=1, max_length=120)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    observations: list[VisionObservation] = Field(default_factory=list)
    # Opaque provider-local reference only. Raw image bytes are deliberately
    # excluded from the cognition contract.
    image_ref: str | None = Field(default=None, max_length=500)


class PerceptionSummary(BaseModel):
    """Small observable-only summary suitable for cognition/renderers."""

    frame_id: str | None = None
    source_id: str | None = None
    person_present: bool = False
    person_count: int = Field(default=0, ge=0)
    gaze_toward_camera: bool = False
    observation_kinds: list[str] = Field(default_factory=list)
    captured_at: datetime | None = None


class VisionProvider(Protocol):
    name: str

    async def observe(self) -> VisionFrame | None: ...

    def status(self) -> dict[str, Any]: ...
