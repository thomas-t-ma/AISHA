from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext
from aisha.providers.base import LLMStreamChunk
from aisha.providers.mock import MockLLMProvider
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


@pytest.mark.asyncio
async def test_completed_turn_persists_events_and_model_run(tmp_path):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    session_id = await store.create_session()
    orchestrator = AISHAOrchestrator(
        store,
        load_persona(settings.character_dir),
        MockLLMProvider(),
    )

    events = [event async for event in orchestrator.stream_user_turn(session_id, "hello")]

    replay = await store.session_events(session_id)
    runs = await store.session_model_runs(session_id)

    assert [event.type for event in events] == [entry["type"] for entry in replay]
    assert runs[0]["status"] == "completed"
    assert runs[0]["total_ms"] is not None
    assert runs[0]["output_chars"] > 0
    assert runs[0]["backend_metrics"]["provider"] == "mock"


class SlowProvider:
    name = "slow"
    model = "slow-test"
    capabilities = ProviderCapabilities(streaming_text=True)

    def __init__(self) -> None:
        self.first_chunk_emitted = asyncio.Event()

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[LLMStreamChunk]:
        del context
        yield LLMStreamChunk(text="first ")
        self.first_chunk_emitted.set()
        await asyncio.sleep(5)
        yield LLMStreamChunk(text="second")
        yield LLMStreamChunk(final=True)


@pytest.mark.asyncio
async def test_cancelled_turn_is_not_committed_as_assistant_message(tmp_path):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    session_id = await store.create_session()
    provider = SlowProvider()
    orchestrator = AISHAOrchestrator(
        store,
        load_persona(settings.character_dir),
        provider,
    )

    async def collect():
        return [event async for event in orchestrator.stream_user_turn(session_id, "hello")]

    task = asyncio.create_task(collect())
    await provider.first_chunk_emitted.wait()
    cancelled_turn_id = await orchestrator.cancel_active_turn(session_id)

    assert cancelled_turn_id is not None

    events = await asyncio.wait_for(task, timeout=6)
    messages = await store.recent_messages(session_id)
    runs = await store.session_model_runs(session_id)

    assert events[-1].type == "aisha.turn.cancelled"
    assert [message.role for message in messages] == ["user"]
    assert runs[0]["status"] == "cancelled"
