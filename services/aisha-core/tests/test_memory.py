from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from aisha.character.loader import load_persona
from aisha.cognition.orchestrator import AISHAOrchestrator
from aisha.contracts.turns import TurnContext
from aisha.main import app
from aisha.providers.base import LLMStreamChunk
from aisha.settings import Settings
from aisha.storage.database import AISHAStore


class InspectingProvider:
    name = "memory-probe"
    model = "mock"
    seen_prompts: list[str]

    def __init__(self) -> None:
        self.seen_prompts = []

    async def stream_turn(self, context: TurnContext) -> AsyncIterator[LLMStreamChunk]:
        self.seen_prompts.append(context.system_prompt)
        yield LLMStreamChunk(text="Acknowledged.")


@pytest.mark.asyncio
async def test_explicit_memory_crosses_session_but_removal_stops_injection(tmp_path):
    settings = Settings(aisha_profile="mock")
    store = AISHAStore(tmp_path / "aisha.sqlite3")
    await store.initialize()
    first = await store.create_session()
    second = await store.create_session()
    provider = InspectingProvider()
    orchestrator = AISHAOrchestrator(store, load_persona(settings.character_dir), provider)

    await _collect(orchestrator, first, "My favorite color is purple.")
    assert await store.list_memories() == []  # No silent memory extraction.

    memory = await store.create_memory("I prefer brief replies.", source_session_id=first)
    await _collect(orchestrator, second, "What should you remember?")
    assert "I prefer brief replies." in provider.seen_prompts[-1]
    assert "My favorite color is purple." not in provider.seen_prompts[-1]
    assert [item.role for item in await store.recent_messages(second)] == [
        "user", "assistant"
    ]

    corrected = await store.update_memory(memory["memory_id"], "I prefer detailed replies.")
    assert corrected is not None
    assert corrected["source_session_id"] == first
    await _collect(orchestrator, await store.create_session(), "Hello")
    assert "I prefer detailed replies." in provider.seen_prompts[-1]
    assert "I prefer brief replies." not in provider.seen_prompts[-1]

    assert await store.delete_memory(memory["memory_id"])
    await _collect(orchestrator, await store.create_session(), "Hello again")
    assert "I prefer detailed replies." not in provider.seen_prompts[-1]
    assert not await store.delete_memory(memory["memory_id"])


async def _collect(orchestrator: AISHAOrchestrator, session_id: str, text: str):
    return [event async for event in orchestrator.stream_user_turn(session_id, text)]


def test_memory_api_crud_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("AISHA_PROFILE", "mock")
    monkeypatch.setenv("AISHA_DATA_DIR", str(tmp_path))
    with TestClient(app) as client:
        session_id = client.post("/v1/sessions").json()["session_id"]
        assert client.get("/v1/memories").json() == []
        assert client.post("/v1/memories", json={"text": "   "}).status_code == 422
        assert client.post(
            "/v1/memories", json={"text": "X" * 501}
        ).status_code == 422
        assert client.post(
            "/v1/memories",
            headers={"origin": "https://unrelated.example"},
            json={"text": "Do not save"},
        ).status_code == 403

        created = client.post(
            "/v1/memories",
            headers={"origin": "http://127.0.0.1:5173"},
            json={"text": "  My nickname is T.  ", "source_session_id": session_id},
        )
        assert created.status_code == 201
        data = created.json()
        assert data["text"] == "My nickname is T."
        assert data["source_session_id"] == session_id
        memory_id = data["memory_id"]
        assert len(client.get("/v1/memories").json()) == 1

        edited = client.patch(
            f"/v1/memories/{memory_id}",
            json={"text": "My nickname is Tom."},
        )
        assert edited.status_code == 200
        assert edited.json()["text"] == "My nickname is Tom."
        assert client.delete(f"/v1/memories/{memory_id}").status_code == 204
        assert client.get("/v1/memories").json() == []
        assert client.patch(
            f"/v1/memories/{memory_id}", json={"text": "Missing"}
        ).status_code == 404
        assert client.delete(f"/v1/memories/{memory_id}").status_code == 404
