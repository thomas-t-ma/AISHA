import json

from aisha.memory.relevance import (
    CANDIDATE_SELECTION_SYSTEM,
    MESSAGE_ANALYSIS_SYSTEM,
    RELEVANCE_GATE_PROMPT_VERSION,
    OllamaMemoryRelevanceGate,
)


def test_v7_physically_separates_message_analysis_from_candidate_selection():
    analysis_prompt = " ".join(MESSAGE_ANALYSIS_SYSTEM.split())
    selection_prompt = " ".join(CANDIDATE_SELECTION_SYSTEM.split())

    assert RELEVANCE_GATE_PROMPT_VERSION == "personal-continuity-v7-structured-axis"
    assert "current message before any long-term memories are visible" in analysis_prompt
    assert "personal_anchored" in analysis_prompt
    assert "general_informational" in analysis_prompt
    assert "ambiguous_unanchored" in analysis_prompt
    assert "propositions=[]" in analysis_prompt

    assert "frozen personal propositions" in selection_prompt
    assert "Do NOT choose a winner" in selection_prompt
    assert "axis_match" in selection_prompt
    assert "qualifier_fidelity" in selection_prompt
    assert "Candidate retrieval order and score are not semantic evidence" in selection_prompt
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
            "propositions": [{"index": 0, "text": "wants more patient interaction at work"}],
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


def test_selection_parser_applies_structured_axis_hard_rules():
    raw = json.dumps(
        {
            "evaluations": [
                {
                    "proposition_index": 0,
                    "candidate_index": 0,
                    "candidate_topic_key": "patient_education_interest",
                    "relation": "same_fact",
                    "axis_match": "different",
                    "qualifier_fidelity": "preserved",
                    "reason": "different patient activity",
                },
                {
                    "proposition_index": 0,
                    "candidate_index": 1,
                    "candidate_topic_key": "job_patient_interaction_level",
                    "relation": "same_axis_state",
                    "axis_match": "exact",
                    "qualifier_fidelity": "preserved",
                    "reason": "same direct-patient-contact axis",
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
    assert "same_axis_state" in parsed[1]["reason"]


def test_selection_parser_verifies_topic_index_and_uses_order_only_after_hard_checks():
    wrong_topic_echo = json.dumps(
        {
            "evaluations": [
                {
                    "proposition_index": 0,
                    "candidate_index": 1,
                    "candidate_topic_key": "remote_work_preference",
                    "relation": "same_fact",
                    "axis_match": "exact",
                    "qualifier_fidelity": "preserved",
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
                    "relation": "same_fact",
                    "axis_match": "exact",
                    "qualifier_fidelity": "preserved",
                    "reason": "eligible second",
                },
                {
                    "proposition_index": 0,
                    "candidate_index": 0,
                    "candidate_topic_key": "first",
                    "relation": "same_axis_state",
                    "axis_match": "exact",
                    "qualifier_fidelity": "preserved",
                    "reason": "eligible first",
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
    assert parsed[0]["relevant"] is True
    assert parsed[1]["relevant"] is False


def test_selection_parser_allows_one_candidate_to_cover_multiple_propositions():
    raw = json.dumps(
        {
            "evaluations": [
                {
                    "proposition_index": 0,
                    "candidate_index": 1,
                    "candidate_topic_key": "keyboard_preference",
                    "relation": "same_fact",
                    "axis_match": "exact",
                    "qualifier_fidelity": "preserved",
                    "reason": "smooth linear switches",
                },
                {
                    "proposition_index": 1,
                    "candidate_index": 1,
                    "candidate_topic_key": "keyboard_preference",
                    "relation": "same_fact",
                    "axis_match": "exact",
                    "qualifier_fidelity": "preserved",
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


def test_relevance_status_reports_v7_fields():
    gate = OllamaMemoryRelevanceGate(
        model="test-model",
        base_url="http://127.0.0.1:11434",
    )

    status = gate.status()

    assert status["model"] == "test-model"
    assert status["prompt_version"] == "personal-continuity-v7-structured-axis"
    assert status["message_scope"] is None
    assert status["scope_reason"] is None
    assert status["propositions"] == []
