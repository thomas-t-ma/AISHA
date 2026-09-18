from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: f"msg_{uuid4().hex}")
    role: Literal["user", "assistant", "operator", "system"]
    text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TurnContext(BaseModel):
    session_id: str
    turn_id: str = Field(default_factory=lambda: f"turn_{uuid4().hex}")
    persona_version: str
    system_prompt: str
    messages: list[Message]
    user_input: str
