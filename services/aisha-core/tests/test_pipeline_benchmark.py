from __future__ import annotations

import pytest

from aisha.memory.pipeline_benchmark import (
    noisy_synthetic_beliefs,
    pipeline_benchmark_cases,
    run_pipeline_case,
    summarize_pipeline_results,
)


class FakePipelineRetriever:
    def __init__(
        self,
        *,
        selected_topics: set[str],
        evaluated: list[tuple[str, bool]],
        gate_error: str | None = None,
    ) -> None:
        self.selected_topics = selected_topics
        self.evaluated = evaluated
        self.gate_error = gate_error
        self._beliefs: dict[str, dict] = {}

    async def recall_hybrid(
        self,
        user_text: str,
        beliefs: list[dict],
        *,
        lexical_candidates: list[dict],
        total_limit: int = 4,
        candidate_limit: int = 8,
    ) -> list[dict]:
        self._beliefs = {str(row["topic_key"]): row for row in beliefs}
        return [
            {
                "belief": self._beliefs[topic],
                "method": "semantic_reranked",
                "score": 0.5,
                "lexical_score": None,
                "semantic_score": 0.5,
                "candidate_sources": ["semantic"],
            }
            for topic in self.selected_topics
        ][:total_limit]

    def status(self) -> dict:
        return {
            "last_candidates": [
                {
                    "topic_key": topic,
                    "lexical_score": None,
                    "semantic_score": 0.5,
                    "candidate_sources": ["semantic"],
                    "selected": selected,
                    "decision": "reranker_accept" if selected else "reranker_reject",
                    "reason": "synthetic pipeline decision",
                }
                for topic, selected in self.evaluated
            ],
            "relevance_gate": {
                "last_error": self.gate_error,
                "last_metrics": {"total_ms": 15.0},
            },
        }


def test_noisy_store_has_12_targets_plus_120_unique_distractors():
    beliefs = noisy_synthetic_beliefs()

    assert len(beliefs) == 132
    assert len({row["belief_id"] for row in beliefs}) == 132
    assert len({row["topic_key"] for row in beliefs}) == 132
    assert all(row["evidence_status"] == "verified" for row in beliefs)

    smaller = noisy_synthetic_beliefs(distractor_limit=20)
    assert len(smaller) == 32


def test_pipeline_suite_has_expected_mix():
    cases = pipeline_benchmark_cases()

    assert len(cases) == 30
    assert len({case.case_id for case in cases}) == 30
    assert sum(1 for case in cases if not case.expected_topics) == 12
    assert sum(1 for case in cases if len(case.expected_topics) > 1) == 4
    assert {case.category for case in cases} == {
        "positive",
        "multi_memory",
        "update",
        "general_negative",
        "ambiguous_negative",
    }


@pytest.mark.asyncio
async def test_pipeline_case_distinguishes_retrieval_and_gate_failures():
    beliefs = noisy_synthetic_beliefs(distractor_limit=0)
    case = next(
        row
        for row in pipeline_benchmark_cases()
        if row.case_id == "pipeline_patient_and_school"
    )
    retriever = FakePipelineRetriever(
        selected_topics={
            "job_patient_interaction_level",
            "remote_work_preference",
        },
        evaluated=[
            ("job_patient_interaction_level", True),
            ("remote_work_preference", True),
            ("professional_school_goal", False),
        ],
    )

    result = await run_pipeline_case(case, beliefs, retriever)

    assert result["retrieval_misses"] == []
    assert result["gate_false_negatives"] == ["professional_school_goal"]
    assert result["gate_false_positives"] == ["remote_work_preference"]
    assert result["candidate_recall_numerator"] == 2
    assert result["candidate_recall_denominator"] == 2



@pytest.mark.asyncio
async def test_pipeline_case_separates_gate_accept_from_final_limit_displacement():
    beliefs = noisy_synthetic_beliefs(distractor_limit=0)
    case = next(
        row
        for row in pipeline_benchmark_cases()
        if row.case_id == "pipeline_patient_and_school"
    )
    retriever = FakePipelineRetriever(
        selected_topics={"job_patient_interaction_level"},
        evaluated=[
            ("job_patient_interaction_level", True),
            ("professional_school_goal", True),
            ("remote_work_preference", True),
        ],
    )

    result = await run_pipeline_case(case, beliefs, retriever)

    assert result["gate_false_negatives"] == []
    assert result["final_limit_displacements"] == ["professional_school_goal"]
    assert result["gate_false_positives"] == ["remote_work_preference"]
    assert result["final_false_positives"] == []


@pytest.mark.asyncio
async def test_pipeline_case_marks_expected_memory_absent_from_pool_as_retrieval_miss():
    beliefs = noisy_synthetic_beliefs(distractor_limit=0)
    case = next(
        row for row in pipeline_benchmark_cases() if row.case_id == "pipeline_keyboard"
    )
    retriever = FakePipelineRetriever(
        selected_topics=set(),
        evaluated=[
            ("computer_build_priority", False),
            ("computer_performance_priority", False),
        ],
    )
    # The second synthetic topic is not in the zero-distractor store, so use a
    # core topic while still leaving the expected keyboard memory out of pool.
    retriever.evaluated = [
        ("computer_build_priority", False),
        ("horror_game_project", False),
    ]

    result = await run_pipeline_case(case, beliefs, retriever)

    assert result["retrieval_misses"] == ["keyboard_preference"]
    assert result["gate_false_negatives"] == []
    assert result["candidate_recall_numerator"] == 0


def test_pipeline_summary_reports_stage_specific_error_counts():
    results = [
        {
            "category": "positive",
            "expected_topics": ["a"],
            "actual_topics": [],
            "exact": False,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 1,
            "candidate_recall_numerator": 0,
            "retrieval_misses": ["a"],
            "gate_false_negatives": [],
            "final_limit_displacements": [],
            "gate_false_positives": [],
            "final_false_positives": [],
            "correct_rejections": ["x"],
            "candidate_pool_topics": ["x"],
            "gate_error": None,
            "latency_ms": {"total": 20.0, "gate": 10.0},
        },
        {
            "category": "positive",
            "expected_topics": ["b"],
            "actual_topics": [],
            "exact": False,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 1,
            "candidate_recall_numerator": 1,
            "retrieval_misses": [],
            "gate_false_negatives": ["b"],
            "final_limit_displacements": [],
            "gate_false_positives": [],
            "final_false_positives": [],
            "correct_rejections": ["x", "y"],
            "candidate_pool_topics": ["b", "x", "y"],
            "gate_error": None,
            "latency_ms": {"total": 30.0, "gate": 20.0},
        },
        {
            "category": "negative",
            "expected_topics": [],
            "actual_topics": ["c"],
            "exact": False,
            "true_positive": 0,
            "false_positive": 1,
            "false_negative": 0,
            "candidate_recall_numerator": 0,
            "retrieval_misses": [],
            "gate_false_negatives": [],
            "final_limit_displacements": [],
            "gate_false_positives": ["c"],
            "final_false_positives": ["c"],
            "correct_rejections": ["x"],
            "candidate_pool_topics": ["c", "x"],
            "gate_error": None,
            "latency_ms": {"total": 40.0, "gate": 30.0},
        },
    ]

    summary = summarize_pipeline_results(results)

    assert summary["candidate_pool_recall"] == 0.5
    assert summary["retrieval_misses"] == 1
    assert summary["gate_false_negatives"] == 1
    assert summary["final_limit_displacements"] == 0
    assert summary["gate_false_positives"] == 1
    assert summary["final_false_positives"] == 1
    assert summary["correct_rejections"] == 4
    assert summary["protocol_failures"] == 0
    assert summary["median_candidate_pool_size"] == 2.0
    assert summary["latency_ms"]["median_gate"] == 20.0
