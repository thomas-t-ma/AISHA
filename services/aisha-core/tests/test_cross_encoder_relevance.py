from __future__ import annotations

import pytest

from aisha.memory.cross_encoder_relevance import CrossEncoderMemoryRelevanceGate


def candidate(topic: str, text: str, question: str | None = None) -> dict:
    return {
        "belief": {
            "belief_id": f"belief_{topic}",
            "topic_key": topic,
            "text": text,
            "open_question": question,
            "evidence_status": "verified",
        }
    }


@pytest.mark.asyncio
async def test_cross_encoder_gate_applies_probability_threshold_without_optional_runtime():
    seen: list[tuple[str, list[str]]] = []

    def predictor(query: str, documents: list[str]) -> list[float]:
        seen.append((query, documents))
        return [0.91, 0.18]

    gate = CrossEncoderMemoryRelevanceGate(
        model="fake-cross-encoder",
        threshold=0.5,
        predictor=predictor,
    )
    decisions = await gate.judge(
        "I want more direct time with patients.",
        [
            candidate(
                "job_patient_interaction_level",
                "The user's job involves little direct patient interaction.",
            ),
            candidate(
                "pottery_hobby",
                "The user enjoys making pottery.",
                "Whether they will return to the ceramics studio is unresolved.",
            ),
        ],
    )

    assert [row["relevant"] for row in decisions] == [True, False]
    assert decisions[0]["reason"] == "cross_encoder_score=0.9100"
    assert decisions[1]["reason"] == "cross_encoder_score=0.1800"
    assert seen[0][0] == "I want more direct time with patients."
    assert seen[0][1][0] == "The user's job involves little direct patient interaction."
    assert "Open thread:" in seen[0][1][1]
    assert gate.status()["last_scores"] == [0.91, 0.18]
    assert gate.status()["last_metrics"]["pairs"] == 2
    assert gate.status()["kind"] == "cross_encoder"


@pytest.mark.asyncio
async def test_cross_encoder_gate_threshold_is_configurable():
    gate = CrossEncoderMemoryRelevanceGate(
        model="fake",
        threshold=0.8,
        predictor=lambda _query, _documents: [0.79, 0.81],
    )
    decisions = await gate.judge(
        "current message",
        [candidate("a", "memory a"), candidate("b", "memory b")],
    )
    assert [row["relevant"] for row in decisions] == [False, True]


@pytest.mark.asyncio
async def test_cross_encoder_gate_fails_closed_on_predictor_error():
    def predictor(_query: str, _documents: list[str]) -> list[float]:
        raise RuntimeError("boom")

    gate = CrossEncoderMemoryRelevanceGate(
        model="fake",
        predictor=predictor,
    )
    decisions = await gate.judge(
        "current message",
        [candidate("a", "memory a")],
    )

    assert decisions == [{
        "index": 0,
        "relevant": False,
        "reason": "cross_encoder_unavailable",
    }]
    assert "RuntimeError: boom" == gate.status()["last_error"]


@pytest.mark.asyncio
async def test_cross_encoder_gate_rejects_score_count_mismatch():
    gate = CrossEncoderMemoryRelevanceGate(
        model="fake",
        predictor=lambda _query, _documents: [],
    )
    decisions = await gate.judge(
        "current message",
        [candidate("a", "memory a")],
    )
    assert decisions[0]["relevant"] is False
    assert "cross_encoder_score_count_mismatch" in (gate.status()["last_error"] or "")
