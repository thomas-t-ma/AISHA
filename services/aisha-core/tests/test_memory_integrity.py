from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator

import httpx
import pytest

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.contracts.turns import TurnContext
from aisha.memory.evidence import OllamaEvidenceVerifier
from aisha.memory.ledger import ExperienceLedger
from aisha.memory.quotes import original_source_quote
from aisha.providers.base import LLMStreamChunk
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


class ChatProbe:
    name = "mock"
    model = "mock"

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[LLMStreamChunk]:
        yield LLMStreamChunk(text="Okay.")


class CompositeReflector:
    async def reflect(self, user_text: str, existing: list[dict]) -> list[dict]:
        return [{
            "action": "add",
            "target_belief_id": None,
            "topic_key": "current_employment",
            "text": "The user works at Emory, has accepted a Morehouse position, "
                    "and has less clinical exposure at their current job.",
            "epistemic_status": "stated",
            "source_quote": "My current job has less clinical exposure.",
            "open_question": None,
        }]


class SimpleReflector:
    async def reflect(self, user_text: str, existing: list[dict]) -> list[dict]:
        return [{
            "action": "add",
            "target_belief_id": None,
            "topic_key": "country_residence_decision",
            "text": "The user has decided to live in Japan.",
            "epistemic_status": "stated",
            "source_quote": "I'm going to live in Japan.",
            "open_question": None,
        }]


class CheckerProbe:
    def __init__(self, supported: bool) -> None:
        self.supported = supported
        self.queries: list[dict] = []

    async def check(self, action: dict) -> tuple[bool, str]:
        self.queries.append(action)
        if self.supported:
            return True, "supported"
        return False, "evidence_does_not_support_whole_claim"


async def run_reflection(tmp_path, reflector, verifier):
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    orch = AISHAOrchestrator(
        store, load_persona(Settings(aisha_profile="mock").character_dir),
        ChatProbe(), ledger=ledger, reflector=reflector,
        evidence_verifier=verifier,
    )
    session = await store.create_session()
    message = (
        "My current job has less clinical exposure."
        if isinstance(reflector, CompositeReflector)
        else "I'm going to live in Japan."
    )
    _ = [event async for event in orch.stream_user_turn(session, message)]
    await orch.wait_for_reflections()
    return ledger, orch


@pytest.mark.asyncio
async def test_composite_belief_is_rejected_without_erasing_episode(tmp_path):
    checker = CheckerProbe(supported=False)
    ledger, orch = await run_reflection(tmp_path, CompositeReflector(), checker)
    assert len(checker.queries) == 1
    assert await ledger.list_beliefs() == []
    assert len(await ledger.list_episodes()) == 1
    assert orch.memory_status()["last_result"]["rejections"][0]["reason"] == (
        "evidence_does_not_support_whole_claim"
    )


@pytest.mark.asyncio
async def test_grounded_decision_is_verified_and_stored(tmp_path):
    checker = CheckerProbe(supported=True)
    ledger, orch = await run_reflection(tmp_path, SimpleReflector(), checker)
    beliefs = await ledger.list_beliefs()
    assert len(beliefs) == 1
    assert beliefs[0]["evidence_status"] == "verified"
    versions = await ledger.versions(beliefs[0]["belief_id"])
    assert versions[0]["evidence_status"] == "verified"
    assert orch.memory_status()["evidence_check_enabled"] is True


@pytest.mark.asyncio
async def test_unchecked_beliefs_are_not_mislabelled_verified(tmp_path):
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    episode = await ledger.record_episode(
        session_id="session_a", turn_id="turn_a", user_message_id="user_a",
        assistant_message_id="assistant_a",
        user_text="I might change jobs.", assistant_text="Okay.",
    )
    saved = await ledger.apply(episode=episode, action={
        "action": "add", "topic_key": "career_choice",
        "text": "The user may change jobs.", "epistemic_status": "uncertain",
        "source_quote": "I might change jobs.", "open_question": "Will they?",
    })
    assert saved is not None
    assert saved["evidence_status"] == "legacy_unchecked"


@pytest.mark.asyncio
async def test_schema_upgrades_legacy_beliefs_without_changing_content(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""
            CREATE TABLE auto_beliefs (
                belief_id TEXT PRIMARY KEY, topic_key TEXT NOT NULL,
                text TEXT NOT NULL, epistemic_status TEXT NOT NULL,
                source_quote TEXT NOT NULL, source_session_id TEXT NOT NULL,
                source_turn_id TEXT NOT NULL, source_episode_id TEXT NOT NULL,
                open_question TEXT, revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )
        """)
        db.execute("""
            INSERT INTO auto_beliefs VALUES (
              'belief_old', 'old', 'Original untouched text', 'stated', 'quote',
              'session_x', 'turn_x', 'episode_x', NULL, 1,
              '2026-09-01T00:00:00', '2026-09-01T00:00:00'
            )
        """)
        db.execute("""
            CREATE TABLE auto_belief_versions (
                version_id TEXT PRIMARY KEY, belief_id TEXT NOT NULL,
                revision INTEGER NOT NULL, text TEXT NOT NULL,
                epistemic_status TEXT NOT NULL, source_quote TEXT NOT NULL,
                source_session_id TEXT NOT NULL, source_turn_id TEXT NOT NULL,
                source_episode_id TEXT NOT NULL, open_question TEXT,
                recorded_at TEXT NOT NULL
            )
        """)
    ledger = ExperienceLedger(AISHAStore(path))
    await ledger.initialize()
    beliefs = await ledger.list_beliefs()
    assert len(beliefs) == 1
    assert beliefs[0]["text"] == "Original untouched text"
    assert beliefs[0]["evidence_status"] == "legacy_unchecked"


@pytest.mark.asyncio
@pytest.mark.parametrize("supported,expected", [
    (True, "supported"),
    (False, "evidence_does_not_support_whole_claim"),
])
async def test_ollama_checker_uses_unstructured_mlx_compatible_request(
    monkeypatch, supported, expected
):
    captured = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.append(payload)
        if "format" in payload:
            return httpx.Response(501)
        return httpx.Response(
            200, json={"message": {"content": json.dumps(
                {"supported": supported, "reason": "source scope"}
            )}},
        )

    transport = httpx.MockTransport(handle)
    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda *args, **kwargs: client(*args, transport=transport, **kwargs),
    )
    checker = OllamaEvidenceVerifier("qwen3.5:35b-mlx", "http://127.0.0.1:11434")
    result = await checker.check({
        "action": "add", "source_quote": "I'm going to live in Japan.",
        "text": "The user has decided to live in Japan.",
        "epistemic_status": "stated",
    })
    assert result == (supported, expected)
    assert len(captured) == 1
    assert "format" not in captured[0]
    assert captured[0]["think"] is False


@pytest.mark.parametrize(
    ("user_text", "proposed", "expected"),
    [
        (
            "My current job doesn’t offer much patient contact.",
            "My current job doesn't offer much patient contact.",
            "My current job doesn’t offer much patient contact.",
        ),
        (
            'She said “hello” to me.',
            'She said "hello" to me.',
            'She said “hello” to me.',
        ),
        (
            "I might switch jobs, but I have not decided.",
            "I have switched jobs",
            None,
        ),
        (
            "My current job doesn’t offer much patient contact.",
            "My current job doesn't provide much patient contact.",
            None,
        ),
        (
            "My current job doesn’t offer much patient contact.",
            "my current job doesn't offer much patient contact.",
            None,
        ),
    ],
)
def test_source_quote_recovers_only_original_typographic_punctuation(
    user_text, proposed, expected
):
    assert original_source_quote(user_text, proposed) == expected


@pytest.mark.asyncio
async def test_apostrophe_recovery_enables_checked_atomic_memory(tmp_path):
    class TypographicReflection:
        async def reflect(self, user_text: str, existing: list[dict]) -> list[dict]:
            return [{
                "action": "add",
                "topic_key": "patient_contact_at_current_job",
                "target_belief_id": None,
                "text": "The user's current job offers limited patient contact.",
                "epistemic_status": "stated",
                "source_quote": "My current job doesn't offer much patient contact.",
                "open_question": None,
            }]

    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    checker = CheckerProbe(supported=True)
    orch = AISHAOrchestrator(
        store, load_persona(Settings(aisha_profile="mock").character_dir),
        ChatProbe(), ledger=ledger, reflector=TypographicReflection(),
        evidence_verifier=checker,
    )
    session = await store.create_session()
    _ = [event async for event in orch.stream_user_turn(
        session, "My current job doesn’t offer much patient contact."
    )]
    await orch.wait_for_reflections()
    beliefs = await ledger.list_beliefs()
    assert len(beliefs) == 1
    assert beliefs[0]["evidence_status"] == "verified"
    assert beliefs[0]["source_quote"] == (
        "My current job doesn’t offer much patient contact."
    )
    assert checker.queries[0]["source_quote"] == beliefs[0]["source_quote"]
    assert orch.memory_status()["last_result"]["outcome"] == "stored"


def test_reflector_discourages_composite_legacy_revisions():
    from aisha.memory.reflector import REFLECTION_SYSTEM

    assert "patient_contact_at_current_job" in REFLECTION_SYSTEM
    assert "Never revise a composite biography" in REFLECTION_SYSTEM
