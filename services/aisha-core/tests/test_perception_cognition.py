from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.contracts.capabilities import ProviderCapabilities
from aisha.contracts.turns import TurnContext
from aisha.memory.ledger import ExperienceLedger
from aisha.perception.base import PerceptionSummary
from aisha.perception.policy import PerceptionPromptPolicy
from aisha.providers.base import LLMStreamChunk
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


class RecordingProvider:
    name = "recording"
    model = "recording-model"
    capabilities = ProviderCapabilities(streaming_text=True)

    def __init__(self) -> None:
        self.contexts: list[TurnContext] = []

    async def warmup(self) -> dict:
        return {"provider": self.name, "preloaded": True}

    async def stream_turn(
        self,
        context: TurnContext,
    ) -> AsyncIterator[LLMStreamChunk]:
        self.contexts.append(context.model_copy(deep=True))
        yield LLMStreamChunk(text="Got it.")
        yield LLMStreamChunk(final=True, metrics={"provider": self.name})


def test_perception_prompt_policy_whitelists_only_observable_summary_facts():
    summary = PerceptionSummary(
        frame_id="frame_SECRET",
        source_id="camera_SECRET",
        person_present=True,
        person_count=2,
        gaze_toward_camera=True,
        head_approximately_frontal=True,
        head_frontal_score=0.91,
        primary_person_x=0.73,
        primary_person_y=0.44,
        observation_kinds=["face", "gaze", "object"],
        visible_objects=["cup", "laptop", "book"],
    )

    rendered = PerceptionPromptPolicy().render(summary)

    assert rendered is not None
    assert "2 people currently visible to the local camera." in rendered
    assert "approximately oriented toward the camera" in rendered
    assert "toward_camera" in rendered
    assert "Visible nearby objects include: cup, laptop, book." in rendered

    # Metadata, coordinates, scores, and provider internals are not cognition input.
    for forbidden in (
        "frame_SECRET",
        "camera_SECRET",
        "0.91",
        "0.73",
        "0.44",
        "observation_kinds",
        "frame_id",
        "source_id",
    ):
        assert forbidden not in rendered

    assert "not user-authored memory" in rendered
    assert "Do not treat these observations as autobiographical memory" in rendered


def test_perception_prompt_policy_emits_nothing_without_useful_observation():
    summary = PerceptionSummary(
        frame_id="frame_SECRET",
        source_id="camera_SECRET",
        person_present=False,
        person_count=0,
        observation_kinds=["background"],
    )

    assert PerceptionPromptPolicy().render(summary) is None


@pytest.mark.asyncio
async def test_transient_perception_reaches_model_but_not_messages_or_memory(tmp_path):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    session_id = await store.create_session()
    provider = RecordingProvider()

    summary = PerceptionSummary(
        frame_id="frame_DO_NOT_PERSIST",
        source_id="camera_DO_NOT_PERSIST",
        person_present=True,
        person_count=1,
        head_approximately_frontal=True,
        head_frontal_score=0.88,
        primary_person_x=0.51,
        primary_person_y=0.49,
        observation_kinds=["face", "object"],
        visible_objects=["cup", "laptop"],
    )

    orchestrator = AISHAOrchestrator(
        store,
        load_persona(settings.character_dir),
        provider,
        ledger=ledger,
        perception_summary_provider=lambda: summary,
    )

    events = [
        event
        async for event in orchestrator.stream_user_turn(
            session_id,
            "Hello AISHA",
        )
    ]

    assert provider.contexts
    context = provider.contexts[-1]
    assert "1 person currently visible to the local camera." in context.system_prompt
    assert "approximately oriented toward the camera" in context.system_prompt
    assert "Visible nearby objects include: cup, laptop." in context.system_prompt

    forbidden = (
        "frame_DO_NOT_PERSIST",
        "camera_DO_NOT_PERSIST",
        "0.88",
        "0.51",
        "0.49",
        "observation_kinds",
    )
    for value in forbidden:
        assert value not in context.system_prompt

    messages = await store.recent_messages(session_id)
    assert [(message.role, message.text) for message in messages] == [
        ("user", "Hello AISHA"),
        ("assistant", "Got it."),
    ]
    serialized_messages = "\n".join(message.text for message in messages)
    assert "visible to the local camera" not in serialized_messages
    assert "frame_DO_NOT_PERSIST" not in serialized_messages
    assert "camera_DO_NOT_PERSIST" not in serialized_messages
    assert "Visible nearby objects" not in serialized_messages

    episodes = await ledger.list_episodes()
    assert len(episodes) == 1
    assert episodes[0]["user_text"] == "Hello AISHA"
    assert episodes[0]["assistant_text"] == "Got it."
    episode_text = episodes[0]["user_text"] + "\n" + episodes[0]["assistant_text"]
    assert "visible to the local camera" not in episode_text
    assert "frame_DO_NOT_PERSIST" not in episode_text
    assert "camera_DO_NOT_PERSIST" not in episode_text
    assert "Visible nearby objects" not in episode_text

    event_text = "\n".join(event.model_dump_json() for event in events)
    assert "frame_DO_NOT_PERSIST" not in event_text
    assert "camera_DO_NOT_PERSIST" not in event_text
