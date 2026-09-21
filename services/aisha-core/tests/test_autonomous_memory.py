from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

import pytest
from fastapi.testclient import TestClient

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.contracts.turns import TurnContext
from aisha.main import app
from aisha.memory.ledger import ExperienceLedger
from aisha.memory.reflector import OllamaReflector
from aisha.providers.base import LLMStreamChunk
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


class ConversationProvider:
    name = "mock"
    model = "mock"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[LLMStreamChunk]:
        self.prompts.append(context.system_prompt)
        yield LLMStreamChunk(text="I see.")


class ReflectionProbe:
    def __init__(self) -> None:
        self.seen_user_text: list[str] = []

    async def reflect(self, user_text: str, existing: list[dict]) -> list[dict]:
        self.seen_user_text.append(user_text)
        if "might switch jobs" in user_text:
            return [{
                "action": "add",
                "target_belief_id": None,
                "topic_key": "job-decision",
                "text": "Thomas was considering a job change; no decision was confirmed.",
                "epistemic_status": "uncertain",
                "source_quote": "might switch jobs",
                "open_question": "Did Thomas decide to switch?",
            }]
        if "decided to stay" in user_text:
            target = next(
                belief for belief in existing if belief["topic_key"] == "job-decision"
            )
            return [{
                "action": "revise",
                "target_belief_id": target["belief_id"],
                "topic_key": target["topic_key"],
                "text": "Thomas decided to stay in his current job.",
                "epistemic_status": "stated",
                "source_quote": "decided to stay",
                "open_question": None,
            }]
        return []


@pytest.mark.asyncio
async def test_autonomous_memory_forms_revises_and_forgets_across_sessions(tmp_path):
    store = AISHAStore(tmp_path / "data.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    provider = ConversationProvider()
    reflector = ReflectionProbe()
    persona = load_persona(Settings(aisha_profile="mock").character_dir)
    orch = AISHAOrchestrator(
        store, persona, provider, ledger=ledger, reflector=reflector
    )
    first = await store.create_session()
    events = [
        event async for event in orch.stream_user_turn(
            first, "I might switch jobs, but I'm undecided."
        )
    ]
    assert events[-1].type == "aisha.turn.finished"
    await orch.wait_for_reflections()
    beliefs = await ledger.list_beliefs()
    assert len(beliefs) == 1
    assert beliefs[0]["epistemic_status"] == "uncertain"
    assert beliefs[0]["source_quote"] == "might switch jobs"
    assert len(await ledger.list_episodes()) == 1

    second = await store.create_session()
    _ = [
        event async for event in orch.stream_user_turn(
            second, "I decided to stay at my current job."
        )
    ]
    # She used the provisional belief before processing the correction.
    assert "considering a job change" in provider.prompts[-1]
    await orch.wait_for_reflections()
    beliefs = await ledger.list_beliefs()
    assert len(beliefs) == 1
    assert beliefs[0]["revision"] == 2
    assert beliefs[0]["text"] == "Thomas decided to stay in his current job."
    assert [v["revision"] for v in await ledger.versions(beliefs[0]["belief_id"])] == [1, 2]

    third = await store.create_session()
    _ = [event async for event in orch.stream_user_turn(third, "Hello again.")]
    assert "Thomas decided to stay" in provider.prompts[-1]
    await orch.wait_for_reflections()
    assert reflector.seen_user_text[-1] == "Hello again."

    assert await ledger.forget_belief(beliefs[0]["belief_id"])
    fourth = await store.create_session()
    _ = [event async for event in orch.stream_user_turn(fourth, "Hey there.")]
    await orch.wait_for_reflections()
    assert "Thomas decided to stay" not in provider.prompts[-1]
    assert len(await ledger.list_episodes()) == 4
    assert await ledger.list_beliefs() == []


@pytest.mark.asyncio
async def test_memory_rejects_unsupported_quotes_duplicate_topics_and_bad_revisions(tmp_path):
    store = AISHAStore(tmp_path / "data.sqlite3")
    await store.initialize()
    ledger = ExperienceLedger(store)
    await ledger.initialize()
    episode = await ledger.record_episode(
        session_id="session_x",
        turn_id="turn_x",
        user_message_id="msg_user",
        assistant_message_id="msg_assistant",
        user_text="I might move to another city.",
        assistant_text="You have moved and love it.",  # NOT a source of beliefs
    )
    action = {
        "action": "add",
        "target_belief_id": None,
        "topic_key": "moving",
        "text": "Thomas may move to another city.",
        "epistemic_status": "uncertain",
        "source_quote": "might move",
        "open_question": "Will he move?",
    }
    assert await ledger.apply(episode=episode, action={
        **action, "source_quote": "You have moved",
    }) is None
    saved = await ledger.apply(episode=episode, action=action)
    assert saved is not None
    assert await ledger.apply(episode=episode, action=action) is None
    assert await ledger.apply(episode=episode, action={
        **action, "action": "revise", "target_belief_id": "nonexistent",
    }) is None
    assert len(await ledger.versions(saved["belief_id"])) == 1


def test_autonomous_memory_api_with_mock_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))
    with TestClient(app) as client:
        status = client.get("/v1/memory/status")
        assert status.status_code == 200
        assert status.json()["enabled"] is False
        assert client.get("/v1/memory/beliefs").json() == []
        assert client.get("/v1/memory/episodes").json() == []
        other_origin = client.delete(
            "/v1/memory/beliefs/belief_missing",
            headers={"origin": "https://unrelated.example"},
        )
        assert other_origin.status_code == 403
        assert client.delete("/v1/memory/beliefs/belief_missing").status_code == 404

@pytest.mark.asyncio
@pytest.mark.parametrize('fenced', [False, True])
async def test_mlx_reflector_never_requests_format_and_accepts_strict_json(
    monkeypatch, fenced
):
    seen_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_requests.append(payload)
        # This simulates a backend rejecting an unsupported structured-output
        # flag, without depending on a locally installed MLX model in CI.
        if 'format' in payload:
            return httpx.Response(501, json={'error': 'format not implemented'})
        content = '{"memories":[]}'
        if fenced:
            content = '\x60\x60\x60json\n' + content + '\n\x60\x60\x60'
        return httpx.Response(200, json={'message': {'content': content}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, 'AsyncClient',
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )
    reflector = OllamaReflector('qwen3.5:35b-mlx', 'http://127.0.0.1:11434')
    result = await reflector.reflect('I might switch jobs soon.', [])
    assert result == []
    assert len(seen_requests) == 1
    assert 'format' not in seen_requests[0]
    assert seen_requests[0]['think'] is False


@pytest.mark.asyncio
async def test_reflector_rejects_unstructured_prose(monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={'message': {'content': 'I think: {"memories": []}'}}
        )
    )
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, 'AsyncClient',
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )
    reflector = OllamaReflector('qwen3.5:35b-mlx', 'http://127.0.0.1:11434')
    with pytest.raises(ValueError, match='did not return JSON'):
        await reflector.reflect('I might switch jobs soon.', [])
