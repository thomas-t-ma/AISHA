from aisha.memory.relevance import (
    RELEVANCE_GATE_PROMPT_VERSION,
    RELEVANCE_SYSTEM,
    OllamaMemoryRelevanceGate,
)


def test_relevance_prompt_requires_memory_blind_thread_evidence():
    assert RELEVANCE_GATE_PROMPT_VERSION == "personal-continuity-v2"
    assert "MEMORY-BLIND TEST" in RELEVANCE_SYSTEM
    assert "candidate supplies missing referent" in RELEVANCE_SYSTEM
    assert "general question, not personal continuity" in RELEVANCE_SYSTEM
    assert "BROAD motive" not in RELEVANCE_SYSTEM
    assert "broad motive" in RELEVANCE_SYSTEM


def test_relevance_status_reports_prompt_version():
    gate = OllamaMemoryRelevanceGate(
        model="test-model",
        base_url="http://127.0.0.1:11434",
    )

    status = gate.status()

    assert status["model"] == "test-model"
    assert status["prompt_version"] == "personal-continuity-v2"
