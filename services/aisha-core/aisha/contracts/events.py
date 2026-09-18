from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class AISHAEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex}")
    session_id: str
    turn_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class TextDeltaPayload(BaseModel):
    text: str


class TurnFinishedPayload(BaseModel):
    status: Literal["completed", "cancelled", "failed"]
    message_id: str | None = None
