from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext
from aisha.providers.base import AISHAProviderError, LLMStreamChunk
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


class FailingProvider:
    name = "failing"
    model = "test-model"
    capabilities = ProviderCapabilities(streaming_text=True)

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[LLMStreamChunk]:
        del context
        if False:
            yield LLMStreamChunk()
        raise AISHAProviderError("expected provider failure")


@pytest.mark.asyncio
async def test_provider_failure_emits_failed_turn_without_assistant_message(tmp_path):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    session_id = await store.create_session()
    orchestrator = AISHAOrchestrator(store, load_persona(settings.character_dir), FailingProvider())

    events = [event async for event in orchestrator.stream_user_turn(session_id, "hello")]

    assert events[0].type == "aisha.turn.started"
    assert events[-1].type == "aisha.turn.failed"
    messages = await store.recent_messages(session_id)
    assert [message.role for message in messages] == ["user"]
