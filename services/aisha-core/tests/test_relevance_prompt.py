import json

from aisha.memory.relevance import (
    CANDIDATE_SELECTION_SYSTEM,
    MESSAGE_ANALYSIS_SYSTEM,
    RELEVANCE_GATE_PROMPT_VERSION,
    OllamaMemoryRelevanceGate,
)


def test_v6_physically_separates_message_analysis_from_candidate_selection():
    analysis_prompt = " ".join(MESSAGE_ANALYSIS_SYSTEM.split())
    selection_prompt = " ".join(CANDIDATE_SELECTION_SYSTEM.split())

    assert RELEVANCE_GATE_PROMPT_VERSION == "personal-continuity-v6-two-stage"
    assert "current message before any long-term memories are visible" in analysis_prompt
    assert "personal_anchored" in analysis_prompt
    assert "general_informational" in analysis_prompt
    assert "ambiguous_unanchored" in analysis_prompt
    assert "propositions=[]" in analysis_prompt
    assert "candidate" not in analysis_prompt.lower()

    assert "frozen list of personal propositions" in selection_prompt
    assert "AT MOST ONE candidate" in selection_prompt
    assert "QUALIFIER FIDELITY" in selection_prompt
    assert "Candidate retrieval order and score are not evidence" in selection_prompt


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


def test_selection_parser_allows_at_most_one_memory_per_proposition():
    raw = json.dumps(
        {
            "decisions": [
                {
                    "index": 0,
                    "relevant": True,
                    "proposition_index": 0,
                    "reason": "closest match",
                },
                {
                    "index": 1,
                    "relevant": False,
                    "proposition_index": None,
                    "reason": "broader adjacent memory",
                },
            ]
        }
    )

    parsed = OllamaMemoryRelevanceGate._parse_selection(
        raw,
        expected=2,
        proposition_count=1,
    )

    assert parsed is not None
    assert parsed[0]["relevant"] is True
    assert parsed[0]["proposition_index"] == 0
    assert parsed[1]["relevant"] is False

    duplicate = json.dumps(
        {
            "decisions": [
                {
                    "index": 0,
                    "relevant": True,
                    "proposition_index": 0,
                    "reason": "first match",
                },
                {
                    "index": 1,
                    "relevant": True,
                    "proposition_index": 0,
                    "reason": "second match",
                },
            ]
        }
    )
    assert (
        OllamaMemoryRelevanceGate._parse_selection(
            duplicate,
            expected=2,
            proposition_count=1,
        )
        is None
    )


def test_relevance_status_reports_v6_fields():
    gate = OllamaMemoryRelevanceGate(
        model="test-model",
        base_url="http://127.0.0.1:11434",
    )

    status = gate.status()

    assert status["model"] == "test-model"
    assert status["prompt_version"] == "personal-continuity-v6-two-stage"
    assert status["message_scope"] is None
    assert status["scope_reason"] is None
    assert status["propositions"] == []
