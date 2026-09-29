from __future__ import annotations

import pytest

from aisha.memory.benchmark import synthetic_beliefs
from aisha.memory.relevance import RELEVANCE_SYSTEM
from aisha.memory.reranker_benchmark import (
    SMOKE_RERANKER_CASE_IDS,
    reranker_benchmark_cases,
    reranker_holdout_cases,
    run_reranker_case,
    summarize_reranker_results,
)


class FakeGate:
    def __init__(self, accepted_topics: set[str]) -> None:
        self.accepted_topics = accepted_topics
        self.last_metrics = {"total_ms": 12.0}

    async def judge(self, user_text: str, candidates: list[dict]) -> list[dict]:
        return [
            {
                "index": index,
                "relevant": candidate["belief"]["topic_key"] in self.accepted_topics,
                "reason": "synthetic decision",
            }
            for index, candidate in enumerate(candidates)
        ]

    def status(self) -> dict:
        return {
            "enabled": True,
            "last_metrics": self.last_metrics,
        }


def test_hard_reranker_benchmark_has_40_cases_and_required_categories():
    cases = reranker_benchmark_cases()

    assert len(cases) == 40
    assert len({case.case_id for case in cases}) == 40
    assert SMOKE_RERANKER_CASE_IDS <= {case.case_id for case in cases}

    categories = {case.category for case in cases}
    assert "minimal_pair" in categories
    assert "multi_memory" in categories
    assert "general_question_negative" in categories
    assert "ambiguous_negative" in categories
    assert "same_thread_update" in categories

    assert any(len(case.expected_topics) == 0 for case in cases)
    assert any(len(case.expected_topics) > 1 for case in cases)
    assert all(len(case.candidate_topics) == 6 for case in cases)


def test_every_reranker_candidate_topic_exists_in_synthetic_beliefs():
    known = {belief["topic_key"] for belief in synthetic_beliefs()}

    for case in reranker_benchmark_cases():
        assert set(case.candidate_topics) <= known
        assert set(case.expected_topics) <= set(case.candidate_topics)



def test_holdout_reranker_suite_is_balanced_unique_and_prompt_unseen():
    cases = reranker_holdout_cases()
    normalized_prompt = " ".join(RELEVANCE_SYSTEM.lower().split())

    assert len(cases) == 32
    assert len({case.case_id for case in cases}) == 32
    assert sum(1 for case in cases if case.expected_topics) == 16
    assert sum(1 for case in cases if not case.expected_topics) == 16
    assert all(case.case_id.startswith("holdout_") for case in cases)
    assert all(len(case.candidate_topics) == 6 for case in cases)

    for case in cases:
        normalized_text = " ".join(case.text.lower().split())
        assert normalized_text not in normalized_prompt


def test_every_holdout_candidate_and_expected_topic_exists():
    known = {belief["topic_key"] for belief in synthetic_beliefs()}

    for case in reranker_holdout_cases():
        assert set(case.candidate_topics) <= known
        assert set(case.expected_topics) <= set(case.candidate_topics)


@pytest.mark.asyncio
async def test_run_reranker_case_uses_fixed_candidates_and_scores_exactness():
    case = next(
        case
        for case in reranker_benchmark_cases()
        if case.case_id == "patient_contact_paraphrase"
    )
    gate = FakeGate({"job_patient_interaction_level"})

    result = await run_reranker_case(case, synthetic_beliefs(), gate)

    assert result["exact"] is True
    assert result["actual_topics"] == ["job_patient_interaction_level"]
    assert result["true_positive"] == 1
    assert result["false_positive"] == 0
    assert result["false_negative"] == 0
    assert result["latency_ms"]["gate"] == 12.0
    assert len(result["candidate_decisions"]) == 6


def test_reranker_summary_reports_over_retrieval_and_clean_queries():
    results = [
        {
            "category": "positive",
            "expected_topics": ["a"],
            "actual_topics": ["a", "b"],
            "exact": False,
            "true_positive": 1,
            "false_positive": 1,
            "false_negative": 0,
            "latency_ms": {"total": 10.0, "gate": 8.0},
        },
        {
            "category": "negative",
            "expected_topics": [],
            "actual_topics": [],
            "exact": True,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "latency_ms": {"total": 20.0, "gate": 18.0},
        },
        {
            "category": "positive",
            "expected_topics": ["c"],
            "actual_topics": [],
            "exact": False,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 1,
            "latency_ms": {"total": 30.0, "gate": 28.0},
        },
    ]

    summary = summarize_reranker_results(results)

    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["negative_control_accuracy"] == 1.0
    assert summary["expected_memories"] == 2
    assert summary["over_retrieval_rate"] == 0.5
    assert summary["clean_queries"] == 2
    assert summary["clean_query_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert summary["false_positive_cases"] == 1
    assert summary["protocol_failures"] == 0
    assert summary["judged_cases"] == 3
    assert summary["judged_exact_accuracy"] == pytest.approx(1 / 3, rel=1e-3)
    assert summary["judged_precision"] == 0.5
    assert summary["judged_recall"] == 0.5
    assert summary["latency_ms"]["median_gate"] == 18.0
