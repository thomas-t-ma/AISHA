from aisha.memory.relevance import (
    RELEVANCE_GATE_PROMPT_VERSION,
    RELEVANCE_SYSTEM,
    OllamaMemoryRelevanceGate,
)


def test_relevance_prompt_requires_memory_blind_thread_evidence():
    normalized_prompt = " ".join(RELEVANCE_SYSTEM.split())

    assert RELEVANCE_GATE_PROMPT_VERSION == "personal-continuity-v2"
    assert "MEMORY-BLIND TEST" in normalized_prompt
    assert "candidate supplies missing referent" in normalized_prompt
    assert "general question, not personal continuity" in normalized_prompt
    assert "BROAD motive" not in normalized_prompt
    assert "broad motive" in normalized_prompt


def test_relevance_status_reports_prompt_version():
    gate = OllamaMemoryRelevanceGate(
        model="test-model",
        base_url="http://127.0.0.1:11434",
    )

    status = gate.status()

    assert status["model"] == "test-model"
    assert status["prompt_version"] == "personal-continuity-v2"
