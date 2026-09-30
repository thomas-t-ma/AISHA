from __future__ import annotations

from aisha.memory.benchmark import synthetic_beliefs
from aisha.memory.pipeline_benchmark import (
    NOISY_MEMORY_GROUPS,
    pipeline_benchmark_cases,
)
from aisha.memory.validation_benchmark import (
    VALIDATION_DISTRACTOR_GROUPS,
    VALIDATION_SUITE_VERSION,
    validation_beliefs,
    validation_cases,
)


def test_validation_suite_shape_is_frozen_and_balanced():
    cases = validation_cases()
    assert VALIDATION_SUITE_VERSION == "unseen-v1-2026-09-30"
    assert len(cases) == 48

    category_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1

    assert category_counts == {
        "ambiguous_negative": 12,
        "general_negative": 12,
        "multi_memory": 8,
        "positive": 12,
        "update": 4,
    }


def test_validation_case_ids_do_not_overlap_development_suite():
    validation_ids = [case.case_id for case in validation_cases()]
    dev_ids = {case.case_id for case in pipeline_benchmark_cases()}

    assert len(validation_ids) == len(set(validation_ids))
    assert not (set(validation_ids) & dev_ids)


def test_validation_expected_topics_are_frozen_core_memories():
    core_topics = {str(row["topic_key"]) for row in synthetic_beliefs()}
    expected_topics = {
        topic
        for case in validation_cases()
        for topic in case.expected_topics
    }

    assert expected_topics <= core_topics
    assert len(core_topics) == 12
    assert expected_topics == core_topics


def test_validation_distractors_are_new_and_noncolliding():
    validation_rows = [
        row
        for group in VALIDATION_DISTRACTOR_GROUPS.values()
        for row in group
    ]
    validation_topics = [topic for topic, _text in validation_rows]
    dev_topics = {
        topic
        for group in NOISY_MEMORY_GROUPS.values()
        for topic, _text in group
    }
    core_topics = {str(row["topic_key"]) for row in synthetic_beliefs()}

    assert len(validation_topics) == 72
    assert len(validation_topics) == len(set(validation_topics))
    assert not (set(validation_topics) & dev_topics)
    assert not (set(validation_topics) & core_topics)


def test_validation_memory_store_has_expected_size_and_verified_rows():
    beliefs = validation_beliefs()

    assert len(beliefs) == 84
    assert len({str(row["topic_key"]) for row in beliefs}) == 84
    assert all(row["evidence_status"] == "verified" for row in beliefs)
