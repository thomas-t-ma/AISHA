from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.contracts.turns import TurnContext
from aisha.memory.ledger import ExperienceLedger
from aisha.memory.retrieval import explain_relevant_beliefs, select_relevant_beliefs
from aisha.providers.base import LLMStreamChunk
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


class PromptProbe:
    name = "mock"
    model = "mock"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[LLMStreamChunk]:
        self.prompts.append(context.system_prompt)
        yield LLMStreamChunk(text="Okay.")


async def add_belief(
    ledger: ExperienceLedger,
    *,
    topic: str,
    text: str,
    quote: str,
    verified: bool,
    question: str | None = None,
) -> dict:
    episode = await ledger.record_episode(
        session_id=f"session_{topic}",
        turn_id=f"turn_{topic}",
        user_message_id=f"user_{topic}",
        assistant_message_id=f"assistant_{topic}",
        user_text=quote,
        assistant_text="Okay.",
    )
    saved, reason = await ledger.apply_with_reason(
        episode=episode,
        action={
            "action": "add",
            "target_belief_id": None,
            "topic_key": topic,
            "text": text,
            "epistemic_status": "uncertain" if question else "stated",
            "source_quote": quote,
            "open_question": question,
        },
        evidence_checked=verified,
    )
    assert reason == "saved"
    assert saved is not None
    return saved


def test_retrieval_only_considers_verified_and_relevant_beliefs():
    beliefs = [
        {
            "topic_key": "job_patient_interaction_level",
            "text": "The user's current job involves little patient interaction.",
            "open_question": None,
            "evidence_status": "verified",
            "updated_at": "2026-09-21T17:52:15",
        },
        {
            "topic_key": "country_residence_decision",
            "text": "The user has decided to live in Japan.",
            "open_question": None,
            "evidence_status": "verified",
            "updated_at": "2026-09-21T17:17:08",
        },
        {
            "topic_key": "current_employment",
            "text": "The user works at Emory and has another position.",
            "open_question": None,
            "evidence_status": "legacy_unchecked",
            "updated_at": "2026-09-21T16:15:38",
        },
    ]

    selected = select_relevant_beliefs(
        "I want more patient interaction in my work.", beliefs
    )
    assert [b["topic_key"] for b in selected] == ["job_patient_interaction_level"]

    assert select_relevant_beliefs("What should I cook tonight?", beliefs) == []
    assert select_relevant_beliefs("Tell me about my Emory employment.", beliefs) == []

    semantic_paraphrase = (
        "I'm getting tired of spending so much of my day away from the people "
        "I'm supposed to be helping. I think I'd enjoy something more hands-on."
    )
    assert select_relevant_beliefs(semantic_paraphrase, beliefs) == []

    details = explain_relevant_beliefs(
        "I want more patient interaction in my work.", beliefs
    )
    assert details[0]["method"] == "lexical"
    assert details[0]["matched_tokens"] == ["interaction", "patient"]
    assert details[0]["score"] > 0


    # Generic conversational overlap must not recall an otherwise unrelated
    # belief. This reproduces the runtime "gett"/"more" false-positive class.
    noisy_belief = [{
        "topic_key": "job_patient_interaction_level",
        "text": "The user is getting more concerned about patient interaction.",
        "open_question": None,
        "evidence_status": "verified",
        "updated_at": "2026-09-21T17:52:15",
    }]
    noisy_prompt = (
        "I'm getting tired and think I'd enjoy something more hands-on."
    )
    assert select_relevant_beliefs(noisy_prompt, noisy_belief) == []


def test_precision_first_tokenization_does_not_create_suffix_artifacts():
    belief = [{
        "topic_key": "preference_for_hands_on_work",
        "text": "The user thinks they would enjoy something more hands-on.",
        "open_question": None,
        "evidence_status": "verified",
        "updated_at": "2026-09-21T18:30:00",
    }]
    details = explain_relevant_beliefs(
        "I think I'd enjoy something more hands-on.", belief
    )
    assert len(details) == 1
    assert details[0]["matched_tokens"] == ["hand"]
    assert "more" in details[0]["ignored_low_information_tokens"]
    assert "something" in details[0]["ignored_low_information_tokens"]
    assert all(token not in {"gett", "someth"} for token in details[0]["matched_tokens"])


def test_retrieval_can_surface_an_open_thread_when_topic_is_relevant():
    beliefs = [{
        "topic_key": "country_residence_decision",
        "text": "The user is deciding which country to live in.",
        "open_question": "Which country will the user choose?",
        "evidence_status": "verified",
        "updated_at": "2026-09-21T17:13:20",
    }]
    selected = select_relevant_beliefs(
        "I've been thinking about my country decision again.", beliefs
    )
    assert [b["topic_key"] for b in selected] == ["country_residence_decision"]


@pytest.mark.asyncio
async def test_prompt_injects_only_relevant_verified_continuity_memory(tmp_path):
    store = AISHAStore(tmp_path / "data.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()

    await add_belief(
        ledger,
        topic="job_patient_interaction_level",
        text="The user's current job involves little patient interaction.",
        quote="My current job doesn't have much patient interaction.",
        verified=True,
    )
    await add_belief(
        ledger,
        topic="country_residence_decision",
        text="The user has decided to live in Japan.",
        quote="I'm going to live in Japan.",
        verified=True,
    )
    await add_belief(
        ledger,
        topic="current_employment",
        text="The user works at Emory and has a Morehouse offer.",
        quote="My employment situation is complicated.",
        verified=False,
    )

    provider = PromptProbe()
    orch = AISHAOrchestrator(
        store,
        load_persona(Settings(aisha_profile="mock").character_dir),
        provider,
        ledger=ledger,
    )
    session = await store.create_session()
    events = [
        event async for event in orch.stream_user_turn(
            session, "I wish I had more patient interaction at work."
        )
    ]

    prompt = provider.prompts[-1]
    assert "RELEVANT VERIFIED CONTINUITY MEMORY" in prompt
    assert "current job involves little patient interaction" in prompt
    assert "decided to live in Japan" not in prompt
    assert "works at Emory and has a Morehouse offer" not in prompt

    started = next(event for event in events if event.type == "aisha.turn.started")
    assert started.payload["memory_recall_count"] == 1
    assert started.payload["memory_recall_topics"] == [
        "job_patient_interaction_level"
    ]
    assert started.payload["memory_recall_details"][0]["method"] == "lexical"
    assert started.payload["memory_recall_details"][0]["matched_tokens"] == [
        "interaction", "patient"
    ]


@pytest.mark.asyncio
async def test_unrelated_turn_receives_no_learned_memory_context(tmp_path):
    store = AISHAStore(tmp_path / "data.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    await add_belief(
        ledger,
        topic="job_patient_interaction_level",
        text="The user's current job involves little patient interaction.",
        quote="My current job doesn't have much patient interaction.",
        verified=True,
    )

    provider = PromptProbe()
    orch = AISHAOrchestrator(
        store,
        load_persona(Settings(aisha_profile="mock").character_dir),
        provider,
        ledger=ledger,
    )
    session = await store.create_session()
    events = [
        event async for event in orch.stream_user_turn(
            session, "What kind of pasta should I make tonight?"
        )
    ]
    assert "RELEVANT VERIFIED CONTINUITY MEMORY" not in provider.prompts[-1]
    started = next(event for event in events if event.type == "aisha.turn.started")
    assert started.payload["memory_recall_count"] == 0
    assert started.payload["memory_recall_topics"] == []
