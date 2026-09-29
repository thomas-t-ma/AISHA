from __future__ import annotations

import json

import httpx
import pytest

from aisha.memory.relevance import OllamaMemoryRelevanceGate
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
        {
            "topic_key": "job_patient_interaction_level",
            "score": 0.92,
            "selected": True,
            "decision": "direct_accept",
            "reason": "embedding_score_above_direct_threshold",
        },
        {
            "topic_key": "country_residence_decision",
            "score": 0.10,
            "selected": False,
            "decision": "below_candidate_floor",
            "reason": "embedding_score_below_candidate_floor",
        },
    ]
    assert "previously stated personal memory" in retriever.status()["query_instruction"]
    assert retriever.status()["pipeline_version"] == "hybrid-final-gate-v4"

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
        "decision": "below_direct_threshold",
        "reason": "no_relevance_gate_configured",
    }]


class FakeRelevanceGate:
    def __init__(self, relevant: bool) -> None:
        self.relevant = relevant
        self.calls: list[tuple[str, list[dict]]] = []
        self.last_error: str | None = None
        self.last_decisions: list[dict] = []

    def reset(self) -> None:
        self.last_error = None
        self.last_decisions = []

    async def judge(self, user_text: str, candidates: list[dict]) -> list[dict]:
        self.calls.append((user_text, candidates))
        return [
            {
                "index": index,
                "relevant": self.relevant,
                "reason": "same underlying work concern" if self.relevant else "generic association",
            }
            for index, _candidate in enumerate(candidates)
        ]

    def status(self) -> dict:
        return {
            "enabled": True,
            "model": "fake-reranker",
            "last_error": self.last_error,
            "last_decisions": self.last_decisions,
        }


@pytest.mark.asyncio
async def test_midrange_semantic_candidate_can_be_accepted_by_relevance_gate(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0], [0.4344, 0.5656]]},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    gate = FakeRelevanceGate(relevant=True)
    retriever = OllamaSemanticMemoryRetriever(
        model="embed",
        base_url="http://127.0.0.1:11434",
        threshold=0.72,
        candidate_floor=0.30,
        relevance_gate=gate,
    )
    belief = verified_belief(
        "belief_patient",
        "job_patient_interaction_level",
        "The user's current job involves little patient interaction.",
    )

    result = await retriever.recall(
        "I want something more hands-on with the people I'm helping.",
        [belief],
    )
    assert len(gate.calls) == 1
    assert [item["belief"]["belief_id"] for item in result] == ["belief_patient"]
    assert result[0]["method"] == "semantic_reranked"
    assert result[0]["reranker_reason"] == "same underlying work concern"
    assert retriever.status()["last_candidates"][0]["decision"] == "reranker_accept"


@pytest.mark.asyncio
async def test_midrange_semantic_candidate_can_be_rejected_by_relevance_gate(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0], [0.4344, 0.5656]]},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    gate = FakeRelevanceGate(relevant=False)
    retriever = OllamaSemanticMemoryRetriever(
        model="embed",
        base_url="http://127.0.0.1:11434",
        threshold=0.72,
        candidate_floor=0.30,
        relevance_gate=gate,
    )
    belief = verified_belief(
        "belief_patient",
        "job_patient_interaction_level",
        "The user's current job involves little patient interaction.",
    )

    assert await retriever.recall("A vaguely related work comment.", [belief]) == []
    assert retriever.status()["last_candidates"][0]["decision"] == "reranker_reject"


@pytest.mark.asyncio
async def test_below_candidate_floor_does_not_call_relevance_gate(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0], [0.1634, 0.8366]]},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    gate = FakeRelevanceGate(relevant=True)
    retriever = OllamaSemanticMemoryRetriever(
        model="embed",
        base_url="http://127.0.0.1:11434",
        threshold=0.72,
        candidate_floor=0.30,
        relevance_gate=gate,
    )
    belief = verified_belief(
        "belief_patient",
        "job_patient_interaction_level",
        "The user's current job involves little patient interaction.",
    )

    assert await retriever.recall("What is a good dessert to make tonight?", [belief]) == []
    assert gate.calls == []
    assert retriever.status()["last_candidates"][0]["decision"] == "below_candidate_floor"


@pytest.mark.asyncio
async def test_ollama_relevance_gate_parses_v3_scope_and_batch_json(monkeypatch):
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps({
                        "message_scope": "personal_anchored",
                        "scope_reason": "explicit personal patient-contact concern",
                        "decisions": [{
                            "index": 0,
                            "relevant": True,
                            "reason": "same underlying patient-contact concern",
                        }],
                    })
                }
            },
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    gate = OllamaMemoryRelevanceGate(
        model="qwen3.5:35b-mlx",
        base_url="http://127.0.0.1:11434",
        keep_alive="30m",
    )
    candidate = {
        "belief": verified_belief(
            "belief_patient",
            "job_patient_interaction_level",
            "The user's current job involves little patient interaction.",
        ),
        "score": 0.4344,
    }
    decisions = await gate.judge(
        "I want work that feels more hands-on with the people I'm helping.",
        [candidate],
    )

    assert decisions[0]["relevant"] is True
    assert requests[0]["think"] is False
    assert requests[0]["options"]["temperature"] == 0
    assert requests[0]["options"]["num_predict"] == 320
    assert "format" not in requests[0]
    assert requests[0]["keep_alive"] == "30m"
    system_prompt = requests[0]["messages"][0]["content"]
    assert "GLOBAL SCOPE CLASSIFICATION" in system_prompt
    assert "personal_anchored" in system_prompt
    assert "MINIMAL SUFFICIENT SET" in system_prompt
    assert "UNIQUE VALUE" in system_prompt
    assert "MEMORY-BLIND TEST" in system_prompt
    assert gate.status()["message_scope"] == "personal_anchored"
    assert gate.status()["last_metrics"]["total_ms"] == 0.0


@pytest.mark.asyncio
async def test_relevance_gate_failure_rejects_candidate_without_breaking_chat(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "not json"}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    gate = OllamaMemoryRelevanceGate(
        model="qwen3.5:35b-mlx",
        base_url="http://127.0.0.1:11434",
    )
    decisions = await gate.judge(
        "current message",
        [{"belief": verified_belief("b1", "topic", "memory"), "score": 0.4}],
    )
    assert decisions == [{
        "index": 0,
        "relevant": False,
        "reason": "relevance_gate_unavailable",
    }]
    assert "relevance_gate_invalid_response" in (gate.last_error or "")


@pytest.mark.asyncio
async def test_relevance_gate_diagnostics_are_cleared_when_next_turn_is_below_floor(monkeypatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        score = 0.4344 if calls == 1 else 0.1634
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0], [score, 1.0 - score]]},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: real_client(*args, transport=transport, **kwargs),
    )

    gate = FakeRelevanceGate(relevant=True)
    retriever = OllamaSemanticMemoryRetriever(
        model="embed",
        base_url="http://127.0.0.1:11434",
        threshold=0.72,
        candidate_floor=0.30,
        relevance_gate=gate,
    )
    belief = verified_belief(
        "belief_patient",
        "job_patient_interaction_level",
        "The user's current job involves little patient interaction.",
    )

    first = await retriever.recall("hands-on helping work", [belief])
    assert first
    gate.last_decisions = [{"index": 0, "relevant": True, "reason": "stale"}]

    second = await retriever.recall("dessert tonight", [belief])
    assert second == []
    assert retriever.status()["relevance_gate"]["last_decisions"] == []


class TopicRelevanceGate:
    def __init__(self, relevant_topics: set[str]) -> None:
        self.relevant_topics = relevant_topics
        self.calls: list[list[dict]] = []
        self.last_error: str | None = None
        self.last_decisions: list[dict] = []

    def reset(self) -> None:
        self.last_error = None
        self.last_decisions = []

    async def judge(self, user_text: str, candidates: list[dict]) -> list[dict]:
        self.calls.append(candidates)
        decisions = []
        for index, candidate in enumerate(candidates):
            topic = str(candidate["belief"]["topic_key"])
            relevant = topic in self.relevant_topics
            decisions.append({
                "index": index,
                "relevant": relevant,
                "reason": "same personal thread" if relevant else "only generic overlap",
            })
        self.last_decisions = decisions
        return decisions

    def status(self) -> dict:
        return {
            "enabled": True,
            "model": "topic-gate",
            "last_error": self.last_error,
            "last_decisions": self.last_decisions,
        }


@pytest.mark.asyncio
async def test_hybrid_final_gate_can_reject_lexical_false_positive(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    [1.0, 0.0],
                    [0.45, 0.55],
                    [0.35, 0.65],
                ]
            },
        )

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
    remote = verified_belief(
        "belief_remote",
        "remote_work_preference",
        "The user prefers jobs that allow some work from home.",
    )
    gate = TopicRelevanceGate({"job_patient_interaction_level"})
    retriever = OllamaSemanticMemoryRetriever(
        model="embed",
        base_url="http://127.0.0.1:11434",
        candidate_floor=0.30,
        relevance_gate=gate,
    )

    lexical_false_positive = [{
        "belief": remote,
        "method": "lexical",
        "score": 3.5,
        "matched_tokens": ["work"],
        "ignored_low_information_tokens": [],
    }]
    result = await retriever.recall_hybrid(
        "I wish I spent more time directly helping patients.",
        [patient, remote],
        lexical_candidates=lexical_false_positive,
        total_limit=4,
    )

    assert [item["belief"]["topic_key"] for item in result] == [
        "job_patient_interaction_level"
    ]
    assert result[0]["method"] == "semantic_reranked"
    diagnostics = {
        row["topic_key"]: row for row in retriever.status()["last_candidates"]
    }
    assert diagnostics["remote_work_preference"]["decision"] == "reranker_reject"
    assert diagnostics["job_patient_interaction_level"]["decision"] == "reranker_accept"


@pytest.mark.asyncio
async def test_hybrid_final_gate_deduplicates_lexical_and_semantic_same_belief(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0], [0.50, 0.50]]},
        )

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
    gate = TopicRelevanceGate({"job_patient_interaction_level"})
    retriever = OllamaSemanticMemoryRetriever(
        model="embed",
        base_url="http://127.0.0.1:11434",
        candidate_floor=0.30,
        relevance_gate=gate,
    )
    lexical = [{
        "belief": patient,
        "method": "lexical",
        "score": 8.5,
        "matched_tokens": ["interaction", "patient"],
        "ignored_low_information_tokens": [],
    }]

    result = await retriever.recall_hybrid(
        "I want more patient interaction.",
        [patient],
        lexical_candidates=lexical,
    )

    assert len(result) == 1
    assert result[0]["method"] == "hybrid_reranked"
    assert result[0]["candidate_sources"] == ["lexical", "semantic"]
    assert len(gate.calls) == 1
    assert len(gate.calls[0]) == 1
