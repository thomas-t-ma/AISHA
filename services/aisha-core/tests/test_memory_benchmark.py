from __future__ import annotations

import pytest

from aisha.memory.benchmark import (
    RecallCase,
    benchmark_cases,
    run_case,
    summarize_results,
    synthetic_beliefs,
)


class FakeSemanticRetriever:
    def __init__(self, topic: str | None) -> None:
        self.topic = topic
        self.last_candidates: list[dict] = []

    async def recall(
        self,
        user_text: str,
        beliefs: list[dict],
        *,
        exclude_belief_ids: set[str] | None = None,
        remaining_limit: int | None = None,
    ) -> list[dict]:
        if self.topic is None:
            self.last_candidates = []
            return []
        belief = next(row for row in beliefs if row["topic_key"] == self.topic)
        if str(belief["belief_id"]) in (exclude_belief_ids or set()):
            return []
        self.last_candidates = [{
            "topic_key": self.topic,
            "score": 0.44,
            "selected": True,
            "decision": "reranker_accept",
            "reason": "same synthetic thread",
        }]
        return [{
            "belief": belief,
            "method": "semantic_reranked",
            "score": 0.44,
            "matched_tokens": [],
            "ignored_low_information_tokens": [],
            "reranker_reason": "same synthetic thread",
        }]

    def status(self) -> dict:
        return {
            "last_candidates": self.last_candidates,
            "relevance_gate": {"enabled": True},
        }


def test_synthetic_benchmark_has_unique_verified_topics_and_mixed_cases():
    beliefs = synthetic_beliefs()
    cases = benchmark_cases()

    topics = [belief["topic_key"] for belief in beliefs]
    assert len(topics) == len(set(topics))
    assert all(belief["evidence_status"] == "verified" for belief in beliefs)

    categories = {case.category for case in cases}
    assert "semantic_positive" in categories
    assert "lexical_positive" in categories
    assert "hard_negative" in categories
    assert "multi_memory" in categories
    assert any(not case.expected_topics for case in cases)


@pytest.mark.asyncio
async def test_run_case_combines_lexical_and_semantic_without_database():
    beliefs = synthetic_beliefs()
    case = RecallCase(
        case_id="combined",
        text=(
            "I want more patient interaction, and I don't want my next role "
            "to keep me onsite every weekday."
        ),
        expected_topics=("job_patient_interaction_level", "remote_work_preference"),
        category="test",
    )
    semantic = FakeSemanticRetriever("remote_work_preference")

    result = await run_case(case, beliefs, semantic)

    assert result["exact"] is True
    assert set(result["actual_topics"]) == {
        "job_patient_interaction_level",
        "remote_work_preference",
    }
    assert {row["method"] for row in result["methods"]} == {
        "lexical",
        "semantic_reranked",
    }
    assert result["latency_ms"]["total"] >= 0


def test_summary_reports_precision_recall_negatives_and_latency():
    results = [
        {
            "expected_topics": ["a"],
            "actual_topics": ["a"],
            "exact": True,
            "true_positive": 1,
            "false_positive": 0,
            "false_negative": 0,
            "methods": [{"method": "lexical"}],
            "semantic_candidates": [{"decision": "direct_accept"}],
            "latency_ms": {"total": 10.0, "semantic": 5.0},
        },
        {
            "expected_topics": [],
            "actual_topics": ["b"],
            "exact": False,
            "true_positive": 0,
            "false_positive": 1,
            "false_negative": 0,
            "methods": [{"method": "semantic_reranked"}],
            "semantic_candidates": [{"decision": "reranker_accept"}],
            "latency_ms": {"total": 20.0, "semantic": 15.0},
        },
        {
            "expected_topics": ["c"],
            "actual_topics": [],
            "exact": False,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 1,
            "methods": [],
            "semantic_candidates": [{"decision": "below_candidate_floor"}],
            "latency_ms": {"total": 30.0, "semantic": 10.0},
        },
    ]

    summary = summarize_results(results)

    assert summary["exact_case_accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["negative_control_accuracy"] == 0.0
    assert summary["method_counts"] == {
        "lexical": 1,
        "semantic_reranked": 1,
    }
    assert summary["candidate_decisions"]["reranker_accept"] == 1
    assert summary["latency_ms"]["median_total"] == 20.0
