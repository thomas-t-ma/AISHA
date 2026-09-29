import json

from aisha.memory.relevance import (
    RELEVANCE_GATE_PROMPT_VERSION,
    RELEVANCE_SYSTEM,
    OllamaMemoryRelevanceGate,
)


def test_relevance_prompt_requires_global_scope_before_candidate_judgment():
    normalized_prompt = " ".join(RELEVANCE_SYSTEM.split())

    assert RELEVANCE_GATE_PROMPT_VERSION == "personal-continuity-v3"
    assert "GLOBAL SCOPE CLASSIFICATION" in normalized_prompt
    assert "personal_anchored" in normalized_prompt
    assert "general_informational" in normalized_prompt
    assert "ambiguous_unanchored" in normalized_prompt
    assert "EVERY decision MUST be false" in normalized_prompt
    assert "candidate supplies missing referent" in normalized_prompt


def test_relevance_parser_enforces_nonpersonal_scope():
    raw = json.dumps(
        {
            "message_scope": "general_informational",
            "scope_reason": "asks for general keyboard advice",
            "decisions": [
                {"index": 0, "relevant": True, "reason": "topic matches keyboard preference"},
                {"index": 1, "relevant": False, "reason": "unrelated"},
            ],
        }
    )

    parsed = OllamaMemoryRelevanceGate._parse(raw, 2)

    assert parsed is not None
    assert parsed["message_scope"] == "general_informational"
    assert all(not row["relevant"] for row in parsed["decisions"])


def test_relevance_parser_preserves_personal_scope_decisions():
    raw = json.dumps(
        {
            "message_scope": "personal_anchored",
            "scope_reason": "explicit current-job patient contact concern",
            "decisions": [
                {"index": 0, "relevant": True, "reason": "same patient-contact thread"},
                {"index": 1, "relevant": False, "reason": "adjacent volunteering thread"},
            ],
        }
    )

    parsed = OllamaMemoryRelevanceGate._parse(raw, 2)

    assert parsed is not None
    assert parsed["decisions"][0]["relevant"] is True
    assert parsed["decisions"][1]["relevant"] is False


def test_relevance_status_reports_prompt_version_and_scope_fields():
    gate = OllamaMemoryRelevanceGate(
        model="test-model",
        base_url="http://127.0.0.1:11434",
    )

    status = gate.status()

    assert status["model"] == "test-model"
    assert status["prompt_version"] == "personal-continuity-v3"
    assert status["message_scope"] is None
    assert status["scope_reason"] is None
