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
    assert initial.affect == "neutral"
    assert initial.affect_intensity == 0.0

    thinking = director.transition("thinking")
    assert thinking.sequence == 1
    assert thinking.activity == "thinking"
    assert thinking.expression == "focused"
    assert thinking.intensity == 0.50

    same = director.transition("thinking")
    assert same.sequence == 1

    amused = director.set_affect("amused", intensity=0.35)
    assert amused.sequence == 2
    assert amused.activity == "thinking"
    assert amused.affect == "amused"
    assert amused.affect_intensity == 0.35

    speaking = director.transition("speaking")
    assert speaking.sequence == 3
    assert speaking.expression == "engaged"
    assert speaking.affect == "amused"
    assert speaking.affect_intensity == 0.35

    same_affect = director.set_affect("amused", intensity=0.35)
    assert same_affect.sequence == 3

    idle = director.transition("idle")
    assert idle.sequence == 4
    assert idle.expression == "neutral"
    assert idle.affect == "amused"

    clamped = director.set_affect("surprised", intensity=2.0)
    assert clamped.sequence == 5
    assert clamped.affect == "surprised"
    assert clamped.affect_intensity == 1.0

    cleared = director.clear_affect()
    assert cleared.sequence == 6
    assert cleared.affect == "neutral"
    assert cleared.affect_intensity == 0.0


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
    assert all(event.payload["affect"] == "neutral" for event in embodiment)
    assert events[-1].type == "aisha.turn.finished"
    assert orchestrator.embodiment_status()["activity"] == "idle"


def test_orchestrator_affect_control_is_independent_of_activity(tmp_path):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    orchestrator = AISHAOrchestrator(
        store,
        load_persona(settings.character_dir),
        MockLLMProvider(),
    )

    thinking = orchestrator.embodiment_director.transition("thinking")
    assert thinking.activity == "thinking"

    affected = orchestrator.set_embodiment_affect("curious", intensity=0.4)
    assert affected["activity"] == "thinking"
    assert affected["expression"] == "focused"
    assert affected["affect"] == "curious"
    assert affected["affect_intensity"] == 0.4

    speaking = orchestrator.embodiment_director.transition("speaking")
    assert speaking.activity == "speaking"
    assert speaking.affect == "curious"
