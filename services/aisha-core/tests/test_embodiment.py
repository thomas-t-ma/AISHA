from __future__ import annotations

import pytest

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.embodiment.state import EmbodimentDirector
from aisha.providers.mock import MockLLMProvider
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


def test_embodiment_director_transitions_are_deterministic():
    director = EmbodimentDirector()

    initial = director.snapshot()
    assert initial.sequence == 0
    assert initial.activity == "idle"
    assert initial.expression == "neutral"
    assert initial.intensity == 0.20

    thinking = director.transition("thinking")
    assert thinking.sequence == 1
    assert thinking.activity == "thinking"
    assert thinking.expression == "focused"
    assert thinking.intensity == 0.50

    same = director.transition("thinking")
    assert same.sequence == 1

    speaking = director.transition("speaking")
    assert speaking.sequence == 2
    assert speaking.expression == "engaged"

    idle = director.transition("idle")
    assert idle.sequence == 3
    assert idle.expression == "neutral"


@pytest.mark.asyncio
async def test_turn_stream_emits_renderer_independent_embodiment_states(tmp_path):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    session_id = await store.create_session()

    orchestrator = AISHAOrchestrator(
        store,
        load_persona(settings.character_dir),
        MockLLMProvider(),
    )

    events = [
        event
        async for event in orchestrator.stream_user_turn(
            session_id,
            "hello",
        )
    ]

    embodiment = [
        event for event in events if event.type == "aisha.embodiment.state"
    ]
    assert [event.payload["activity"] for event in embodiment] == [
        "thinking",
        "speaking",
        "idle",
    ]
    assert [event.payload["expression"] for event in embodiment] == [
        "focused",
        "engaged",
        "neutral",
    ]
    assert all(event.source == "aisha.embodiment" for event in embodiment)
    assert events[-1].type == "aisha.turn.finished"
    assert orchestrator.embodiment_status()["activity"] == "idle"
