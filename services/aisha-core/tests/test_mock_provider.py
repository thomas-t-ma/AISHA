import pytest

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.providers.mock import MockLLMProvider
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


@pytest.mark.asyncio
async def test_mock_vertical_slice(tmp_path):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    session_id = await store.create_session()
    orchestrator = AISHAOrchestrator(store, load_persona(settings.character_dir), MockLLMProvider())

    events = [event async for event in orchestrator.stream_user_turn(session_id, "hello")]
    assert events[0].type == "aisha.turn.started"
    assert events[-1].type == "aisha.turn.finished"
    messages = await store.recent_messages(session_id)
    assert [m.role for m in messages] == ["user", "assistant"]
