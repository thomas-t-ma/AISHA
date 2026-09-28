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


class SemanticProbe:
    def __init__(self, topic_key: str, *, reranked: bool = False) -> None:
        self.topic_key = topic_key
        self.reranked = reranked
        self.calls: list[dict] = []

    async def recall(
        self,
        user_text: str,
        beliefs: list[dict],
        *,
        exclude_belief_ids: set[str] | None = None,
        remaining_limit: int | None = None,
    ) -> list[dict]:
        self.calls.append({
            "user_text": user_text,
            "exclude_belief_ids": exclude_belief_ids or set(),
            "remaining_limit": remaining_limit,
        })
        belief = next(b for b in beliefs if b["topic_key"] == self.topic_key)
        if belief["belief_id"] in (exclude_belief_ids or set()):
            return []
        detail = {
            "belief": belief,
            "method": "semantic_reranked" if self.reranked else "semantic",
            "score": 0.84,
            "matched_tokens": [],
            "ignored_low_information_tokens": [],
        }
        if self.reranked:
            detail["reranker_reason"] = "same underlying concern"
        return [detail]

    def status(self) -> dict:
        return {
            "enabled": True,
            "model": "semantic-probe",
            "threshold": 0.72,
            "limit": 2,
            "cached_beliefs": 1,
            "last_error": None,
            "disabled_reason": None,
        }


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


class ReflectionShouldNotRun:
    def __init__(self) -> None:
        self.calls = 0

    async def reflect(self, user_text: str, existing: list[dict]) -> list[dict]:
        self.calls += 1
        return [{
            "action": "add",
            "target_belief_id": None,
            "topic_key": "synthetic_test_memory",
            "text": "The user has a synthetic test preference.",
            "epistemic_status": "stated",
            "source_quote": user_text,
            "open_question": None,
        }]


@pytest.mark.asyncio
async def test_test_mode_allows_recall_but_suppresses_long_term_learning(tmp_path):
    store = AISHAStore(tmp_path / "sandbox.sqlite3")
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
    initial_episode_count = len(await ledger.list_episodes(limit=100))

    provider = PromptProbe()
    reflector = ReflectionShouldNotRun()
    orch = AISHAOrchestrator(
        store,
        load_persona(Settings(aisha_profile="mock").character_dir),
        provider,
        ledger=ledger,
        reflector=reflector,
    )
    session = await store.create_session()
    updated = await store.set_session_memory_mode(session, "test")
    assert updated is not None and updated["memory_mode"] == "test"

    events = [
        event async for event in orch.stream_user_turn(
            session, "I want more patient interaction in my work."
        )
    ]
    await orch.wait_for_reflections()

    prompt = provider.prompts[-1]
    assert "current job involves little patient interaction" in prompt
    started = next(event for event in events if event.type == "aisha.turn.started")
    assert started.payload["memory_recall_count"] == 1
    assert started.payload["memory_mode"] == "test"
    assert started.payload["memory_learning_enabled"] is False

    assert reflector.calls == 0
    assert len(await ledger.list_episodes(limit=100)) == initial_episode_count
    assert not any(
        belief["topic_key"] == "synthetic_test_memory"
        for belief in await ledger.list_beliefs(limit=100)
    )

    persisted = await store.session_events(session, limit=100)
    skipped = [
        event for event in persisted
        if event["type"] == "aisha.memory.learning_skipped"
    ]
    assert len(skipped) == 1
    assert skipped[0]["payload"] == {
        "reason": "test_mode",
        "memory_mode": "test",
    }


@pytest.mark.asyncio
async def test_switching_test_session_back_to_normal_resumes_learning(tmp_path):
    store = AISHAStore(tmp_path / "resume.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    provider = PromptProbe()
    reflector = ReflectionShouldNotRun()
    orch = AISHAOrchestrator(
        store,
        load_persona(Settings(aisha_profile="mock").character_dir),
        provider,
        ledger=ledger,
        reflector=reflector,
    )
    session = await store.create_session()
    await store.set_session_memory_mode(session, "test")
    _ = [event async for event in orch.stream_user_turn(session, "Synthetic test statement.")]
    assert reflector.calls == 0

    await store.set_session_memory_mode(session, "normal")
    _ = [event async for event in orch.stream_user_turn(session, "Synthetic test statement.")]
    await orch.wait_for_reflections()
    assert reflector.calls == 1


@pytest.mark.asyncio
async def test_semantic_recall_fills_gap_after_lexical_retrieval(tmp_path):
    store = AISHAStore(tmp_path / "semantic.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    patient = await add_belief(
        ledger,
        topic="job_patient_interaction_level",
        text="The user's current job involves little patient interaction.",
        quote="My current job doesn't have much patient interaction.",
        verified=True,
    )

    provider = PromptProbe()
    semantic = SemanticProbe("job_patient_interaction_level")
    orch = AISHAOrchestrator(
        store,
        load_persona(Settings(aisha_profile="mock").character_dir),
        provider,
        ledger=ledger,
        semantic_retriever=semantic,
    )
    session = await store.create_session()
    await store.set_session_memory_mode(session, "test")
    events = [
        event async for event in orch.stream_user_turn(
            session,
            "I'm getting tired of spending so much of my day away from the people "
            "I'm supposed to be helping. I think I'd enjoy something more hands-on.",
        )
    ]

    prompt = provider.prompts[-1]
    assert "current job involves little patient interaction" in prompt
    started = next(event for event in events if event.type == "aisha.turn.started")
    assert started.payload["memory_recall_topics"] == [
        "job_patient_interaction_level"
    ]
    assert started.payload["memory_recall_details"] == [{
        "topic_key": "job_patient_interaction_level",
        "method": "semantic",
        "score": 0.84,
        "matched_tokens": [],
        "ignored_low_information_tokens": [],
    }]
    assert started.payload["semantic_recall_status"]["enabled"] is True
    assert started.payload["memory_retrieval_ms"] >= 0
    assert started.payload["lexical_retrieval_ms"] >= 0
    assert started.payload["semantic_retrieval_ms"] >= 0
    assert semantic.calls[0]["exclude_belief_ids"] == set()
    assert patient["belief_id"] not in semantic.calls[0]["exclude_belief_ids"]


@pytest.mark.asyncio
async def test_lexical_match_wins_and_semantic_retrieval_cannot_duplicate_it(tmp_path):
    store = AISHAStore(tmp_path / "hybrid.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    patient = await add_belief(
        ledger,
        topic="job_patient_interaction_level",
        text="The user's current job involves little patient interaction.",
        quote="My current job doesn't have much patient interaction.",
        verified=True,
    )

    provider = PromptProbe()
    semantic = SemanticProbe("job_patient_interaction_level")
    orch = AISHAOrchestrator(
        store,
        load_persona(Settings(aisha_profile="mock").character_dir),
        provider,
        ledger=ledger,
        semantic_retriever=semantic,
    )
    session = await store.create_session()
    events = [
        event async for event in orch.stream_user_turn(
            session, "I want more patient interaction at work."
        )
    ]

    started = next(event for event in events if event.type == "aisha.turn.started")
    assert started.payload["memory_recall_count"] == 1
    assert started.payload["memory_recall_details"][0]["method"] == "lexical"
    assert semantic.calls[0]["exclude_belief_ids"] == {patient["belief_id"]}


@pytest.mark.asyncio
async def test_turn_started_preserves_reranker_reason_in_recall_diagnostics(tmp_path):
    store = AISHAStore(tmp_path / "reranked.sqlite3")
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
    semantic = SemanticProbe("job_patient_interaction_level", reranked=True)
    orch = AISHAOrchestrator(
        store,
        load_persona(Settings(aisha_profile="mock").character_dir),
        provider,
        ledger=ledger,
        semantic_retriever=semantic,
    )
    session = await store.create_session()
    events = [
        event async for event in orch.stream_user_turn(
            session,
            "I want something more hands-on with the people I'm helping.",
        )
    ]

    started = next(event for event in events if event.type == "aisha.turn.started")
    detail = started.payload["memory_recall_details"][0]
    assert detail["method"] == "semantic_reranked"
    assert detail["reranker_reason"] == "same underlying concern"
