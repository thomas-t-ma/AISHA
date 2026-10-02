from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.contracts.turns import TurnContext
from aisha.perception.base import PerceptionSummary
from aisha.perception.policy import PerceptionPromptPolicy
from aisha.providers.base import LLMStreamChunk
from aisha.providers.mock import MockLLMProvider
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


class CapturingMockProvider(MockLLMProvider):
    def __init__(self) -> None:
        self.last_context: TurnContext | None = None

    async def stream_turn(
        self,
        context: TurnContext,
    ) -> AsyncIterator[LLMStreamChunk]:
        self.last_context = context
        async for chunk in super().stream_turn(context):
            yield chunk


def test_perception_prompt_policy_whitelists_only_transient_facts():
    summary = PerceptionSummary(
        frame_id="frame_secret",
        source_id="camera_secret",
        person_present=True,
        person_count=2,
        gaze_toward_camera=True,
        observation_kinds=["face", "gaze"],
    )

    context = PerceptionPromptPolicy().render(summary)

    assert context is not None
    assert "2 people currently visible" in context
    assert "toward_camera" in context
    assert "frame_secret" not in context
    assert "camera_secret" not in context
    assert "face" not in context
    assert "not user-authored memory" in context
    assert "unobserved traits" in context


def test_perception_prompt_policy_omits_empty_scene():
    assert PerceptionPromptPolicy().render(PerceptionSummary()) is None


@pytest.mark.asyncio
async def test_turn_receives_transient_perception_without_persisting_sensor_context(
    tmp_path,
):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    session_id = await store.create_session()
    provider = CapturingMockProvider()
    summary = PerceptionSummary(
        frame_id="frame_secret",
        source_id="camera_secret",
        person_present=True,
        person_count=1,
        observation_kinds=["face"],
    )
    orchestrator = AISHAOrchestrator(
        store,
        load_persona(settings.character_dir),
        provider,
        perception_summary_provider=lambda: summary,
    )

    events = [
        event
        async for event in orchestrator.stream_user_turn(
            session_id,
            "Hello there",
        )
    ]

    assert events[-1].type == "aisha.turn.finished"
    assert provider.last_context is not None
    prompt = provider.last_context.system_prompt
    assert "TRANSIENT LOCAL PERCEPTION CONTEXT" in prompt
    assert "1 person currently visible" in prompt
    assert "frame_secret" not in prompt
    assert "camera_secret" not in prompt

    messages = await store.session_messages(session_id)
    persisted_text = "\n".join(message["text"] for message in messages)
    assert "Hello there" in persisted_text
    assert "TRANSIENT LOCAL PERCEPTION CONTEXT" not in persisted_text
    assert "frame_secret" not in persisted_text
    assert "camera_secret" not in persisted_text
