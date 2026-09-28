from __future__ import annotations

import json

import httpx
import pytest

from aisha.memory.semantic import OllamaSemanticMemoryRetriever


def verified_belief(
    belief_id: str,
    topic: str,
    text: str,
    *,
    revision: int = 1,
) -> dict:
    return {
        "belief_id": belief_id,
        "topic_key": topic,
        "text": text,
        "open_question": None,
        "evidence_status": "verified",
        "revision": revision,
        "updated_at": "2026-09-28T18:00:00",
    }


@pytest.mark.asyncio
async def test_semantic_retriever_selects_related_verified_belief_and_caches_documents(
    monkeypatch,
):
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        inputs = payload["input"]
        if len(inputs) == 3:
            # Query, patient-contact belief, unrelated Japan belief.
            vectors = [
                [1.0, 0.0],
                [0.92, 0.08],
                [0.10, 0.90],
            ]
        else:
            # Second call should only embed the new query because belief vectors
            # were cached after the first batch.
            assert len(inputs) == 1
            vectors = [[1.0, 0.0]]
        return httpx.Response(200, json={"embeddings": vectors})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    patient = verified_belief(
        "belief_patient",
        "job_patient_interaction_level",
        "The user's current job involves little patient interaction.",
    )
    japan = verified_belief(
        "belief_japan",
        "country_residence_decision",
        "The user has decided to live in Japan.",
    )
    retriever = OllamaSemanticMemoryRetriever(
        model="qwen3-embedding:0.6b",
        base_url="http://127.0.0.1:11434",
        threshold=0.72,
        limit=2,
        keep_alive="30m",
    )

    first = await retriever.recall(
        "I want work that feels more hands-on with the people I'm helping.",
        [patient, japan],
    )
    assert [item["belief"]["belief_id"] for item in first] == ["belief_patient"]
    assert first[0]["method"] == "semantic"
    assert first[0]["score"] == pytest.approx(0.92)
    assert first[0]["matched_tokens"] == []
    assert retriever.status()["last_candidates"] == [
        {"topic_key": "job_patient_interaction_level", "score": 0.92, "selected": True},
        {"topic_key": "country_residence_decision", "score": 0.10, "selected": False},
    ]
    assert "previously stated personal memory" in retriever.status()["query_instruction"]

    second = await retriever.recall(
        "I miss doing something directly useful for people.",
        [patient, japan],
    )
    assert [item["belief"]["belief_id"] for item in second] == ["belief_patient"]
    assert len(requests) == 2
    assert len(requests[0]["input"]) == 3
    assert len(requests[1]["input"]) == 1
    assert requests[0]["model"] == "qwen3-embedding:0.6b"
    assert requests[0]["keep_alive"] == "30m"
    assert requests[0]["input"][0].startswith(
        "Instruct: Given a user's current message"
    )
    assert "\nQuery: I want work that feels more hands-on" in requests[0]["input"][0]
    assert not requests[0]["input"][1].startswith("Instruct:")


@pytest.mark.asyncio
async def test_semantic_retriever_excludes_legacy_and_lexically_selected_beliefs(
    monkeypatch,
):
    seen_inputs: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_inputs.append(payload["input"])
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0], [0.95, 0.05]]},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    lexical = verified_belief(
        "belief_lexical",
        "job_patient_interaction_level",
        "The user's current job involves little patient interaction.",
    )
    other = verified_belief(
        "belief_other",
        "clinical_work_preference",
        "The user prefers hands-on clinical work.",
    )
    legacy = {
        **verified_belief(
            "belief_legacy",
            "current_employment",
            "The user works at two institutions.",
        ),
        "evidence_status": "legacy_unchecked",
    }
    retriever = OllamaSemanticMemoryRetriever(
        model="embed",
        base_url="http://127.0.0.1:11434",
        threshold=0.72,
    )
    result = await retriever.recall(
        "I want something more hands-on.",
        [lexical, other, legacy],
        exclude_belief_ids={"belief_lexical"},
        remaining_limit=1,
    )

    assert [item["belief"]["belief_id"] for item in result] == ["belief_other"]
    assert len(seen_inputs) == 1
    # Query + only the one eligible semantic candidate.
    assert len(seen_inputs[0]) == 2


@pytest.mark.asyncio
async def test_missing_embedding_model_fails_open_and_disables_repeated_calls(monkeypatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"error": "model not found"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    retriever = OllamaSemanticMemoryRetriever(
        model="missing-embedding-model",
        base_url="http://127.0.0.1:11434",
    )
    belief = verified_belief(
        "belief_patient",
        "job_patient_interaction_level",
        "The user's current job involves little patient interaction.",
    )

    assert await retriever.recall("hands-on helping work", [belief]) == []
    assert retriever.status()["disabled_reason"] == "embedding_model_unavailable"
    assert await retriever.recall("another prompt", [belief]) == []
    assert calls == 1


@pytest.mark.asyncio
async def test_semantic_status_exposes_near_miss_below_threshold(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0], [0.69, 0.31]]},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    retriever = OllamaSemanticMemoryRetriever(
        model="embed",
        base_url="http://127.0.0.1:11434",
        threshold=0.72,
    )
    belief = verified_belief(
        "belief_patient",
        "job_patient_interaction_level",
        "The user's current job involves little patient interaction.",
    )
    assert await retriever.recall("hands-on helping work", [belief]) == []
    assert retriever.status()["last_candidates"] == [{
        "topic_key": "job_patient_interaction_level",
        "score": 0.69,
        "selected": False,
    }]
