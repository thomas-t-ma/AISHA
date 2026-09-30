import json

from aisha.memory.relevance import (
    CANDIDATE_SELECTION_SYSTEM,
    MESSAGE_ANALYSIS_SYSTEM,
    RELEVANCE_GATE_PROMPT_VERSION,
    OllamaMemoryRelevanceGate,
)


def test_v9_physically_separates_message_analysis_from_candidate_selection():
    analysis_prompt = " ".join(MESSAGE_ANALYSIS_SYSTEM.split())
    selection_prompt = " ".join(CANDIDATE_SELECTION_SYSTEM.split())

    assert RELEVANCE_GATE_PROMPT_VERSION == "personal-continuity-v9-thread-anchors"
    assert "current message before any long-term memories are visible" in analysis_prompt
    assert "personal_anchored" in analysis_prompt
    assert "general_informational" in analysis_prompt
    assert "ambiguous_unanchored" in analysis_prompt
    assert "propositions=[]" in analysis_prompt

    assert "frozen personal propositions" in selection_prompt
    assert "Do NOT choose a winner" in selection_prompt
    assert "thread_core" in analysis_prompt
    assert "required_anchors" in analysis_prompt
    assert "turn_modifiers" in analysis_prompt
    assert "thread_match" in selection_prompt
    assert "anchor_coverage" in selection_prompt
    assert "predicate_compatibility" in selection_prompt
    assert "Candidate retrieval order and scores are not semantic evidence" in selection_prompt
    assert "MEMORY-ADDRESSABLE" in analysis_prompt


def test_analysis_parser_requires_no_propositions_for_nonpersonal_scope():
    raw = json.dumps(
        {
            "message_scope": "general_informational",
            "scope_reason": "general keyboard question",
            "propositions": [],
        }
    )

    parsed = OllamaMemoryRelevanceGate._parse_analysis(raw)

    assert parsed is not None
    assert parsed["message_scope"] == "general_informational"
    assert parsed["propositions"] == []

    invalid = json.dumps(
        {
            "message_scope": "ambiguous_unanchored",
            "scope_reason": "missing referent",
            "propositions": [{"index": 0, "text": "guessed hidden topic"}],
        }
    )
    assert OllamaMemoryRelevanceGate._parse_analysis(invalid) is None


def test_analysis_parser_requires_personal_proposition():
    raw = json.dumps(
        {
            "message_scope": "personal_anchored",
            "scope_reason": "specific work preference",
            "propositions": [
                {
                    "index": 0,
                    "text": "wants more patient interaction at work",
                    "thread_core": "direct patient interaction at work",
                    "required_anchors": ["direct patient interaction", "work"],
                    "turn_modifiers": [],
                }
            ],
        }
    )

    parsed = OllamaMemoryRelevanceGate._parse_analysis(raw)

    assert parsed is not None
    assert parsed["propositions"][0]["index"] == 0
    assert "patient interaction" in parsed["propositions"][0]["text"]

    invalid = json.dumps(
        {
            "message_scope": "personal_anchored",
            "scope_reason": "specific work preference",
            "propositions": [],
        }
    )
    assert OllamaMemoryRelevanceGate._parse_analysis(invalid) is None


def test_selection_parser_applies_thread_anchor_hard_rules():
    raw = json.dumps(
        {
            "evaluations": [
                {
                    "proposition_index": 0,
                    "candidate_index": 0,
                    "candidate_topic_key": "patient_education_interest",
                    "relation": "adjacent",
                    "thread_match": "different",
                    "anchor_coverage": "partial",
                    "predicate_compatibility": "different",
                    "reason": "different patient activity",
                },
                {
                    "proposition_index": 0,
                    "candidate_index": 1,
                    "candidate_topic_key": "job_patient_interaction_level",
                    "relation": "background_state",
                    "thread_match": "exact",
                    "anchor_coverage": "full",
                    "predicate_compatibility": "compatible",
                    "reason": "same direct-patient-contact thread",
                },
            ]
        }
    )

    parsed = OllamaMemoryRelevanceGate._parse_selection(
        raw,
        expected=2,
        proposition_count=1,
        candidate_topics=["patient_education_interest", "job_patient_interaction_level"],
    )

    assert parsed is not None
    assert parsed[0]["relevant"] is False
    assert parsed[1]["relevant"] is True
    assert parsed[1]["proposition_index"] == 0
    assert "background_state" in parsed[1]["reason"]


def test_selection_parser_verifies_topic_index_and_prioritizes_direct_thread_match():
    wrong_topic_echo = json.dumps(
        {
            "evaluations": [
                {
                    "proposition_index": 0,
                    "candidate_index": 1,
                    "candidate_topic_key": "remote_work_preference",
                    "relation": "same_thread",
                    "thread_match": "exact",
                    "anchor_coverage": "full",
                    "predicate_compatibility": "exact",
                    "reason": "wrong echoed topic",
                }
            ]
        }
    )
    assert (
        OllamaMemoryRelevanceGate._parse_selection(
            wrong_topic_echo,
            expected=2,
            proposition_count=1,
            candidate_topics=["remote_work_preference", "job_patient_interaction_level"],
        )
        is None
    )

    tie = json.dumps(
        {
            "evaluations": [
                {
                    "proposition_index": 0,
                    "candidate_index": 1,
                    "candidate_topic_key": "second",
                    "relation": "same_thread",
                    "thread_match": "exact",
                    "anchor_coverage": "full",
                    "predicate_compatibility": "exact",
                    "reason": "direct thread match",
                },
                {
                    "proposition_index": 0,
                    "candidate_index": 0,
                    "candidate_topic_key": "first",
                    "relation": "background_state",
                    "thread_match": "exact",
                    "anchor_coverage": "full",
                    "predicate_compatibility": "exact",
                    "reason": "eligible background state",
                },
            ]
        }
    )
    parsed = OllamaMemoryRelevanceGate._parse_selection(
        tie,
        expected=2,
        proposition_count=1,
        candidate_topics=["first", "second"],
    )
    assert parsed is not None
    assert parsed[0]["relevant"] is False
    assert parsed[1]["relevant"] is True


def test_selection_parser_rejects_partial_required_anchor_coverage():
    raw = json.dumps(
        {
            "evaluations": [
                {
                    "proposition_index": 0,
                    "candidate_index": 0,
                    "candidate_topic_key": "thai_food_interest",
                    "relation": "same_thread",
                    "thread_match": "broader",
                    "anchor_coverage": "partial",
                    "predicate_compatibility": "compatible",
                    "reason": "missing spicy-hot anchor",
                }
            ]
        }
    )
    parsed = OllamaMemoryRelevanceGate._parse_selection(
        raw,
        expected=1,
        proposition_count=1,
        candidate_topics=["thai_food_interest"],
    )
    assert parsed is not None
    assert parsed[0]["relevant"] is False
    assert "anchors=partial" in parsed[0]["reason"]


def test_selection_parser_allows_one_candidate_to_cover_multiple_propositions():
    raw = json.dumps(
        {
            "evaluations": [
                {
                    "proposition_index": 0,
                    "candidate_index": 1,
                    "candidate_topic_key": "keyboard_preference",
                    "relation": "same_thread",
                    "thread_match": "exact",
                    "anchor_coverage": "full",
                    "predicate_compatibility": "exact",
                    "reason": "smooth linear switches",
                },
                {
                    "proposition_index": 1,
                    "candidate_index": 1,
                    "candidate_topic_key": "keyboard_preference",
                    "relation": "same_thread",
                    "thread_match": "exact",
                    "anchor_coverage": "full",
                    "predicate_compatibility": "exact",
                    "reason": "quiet keyboard preference",
                },
            ]
        }
    )

    parsed = OllamaMemoryRelevanceGate._parse_selection(
        raw,
        expected=2,
        proposition_count=2,
        candidate_topics=["computer_build_priority", "keyboard_preference"],
    )

    assert parsed is not None
    assert parsed[1]["relevant"] is True
    assert parsed[1]["proposition_index"] == 0
    assert parsed[1]["proposition_indices"] == [0, 1]


def test_relevance_status_reports_v9_fields():
    gate = OllamaMemoryRelevanceGate(
        model="test-model",
        base_url="http://127.0.0.1:11434",
    )

    status = gate.status()

    assert status["model"] == "test-model"
    assert status["prompt_version"] == "personal-continuity-v9-thread-anchors"
    assert status["message_scope"] is None
    assert status["scope_reason"] is None
    assert status["propositions"] == []
