from aisha.memory.mlx_reranker import (
    DEFAULT_MEMORY_RERANK_INSTRUCTION,
    MLXQwen3MemoryReranker,
)


def test_mlx_memory_reranker_is_lazy_and_uses_personal_continuity_instruction():
    gate = MLXQwen3MemoryReranker(threshold=0.61)

    assert gate.model_name == "mlx-community/Qwen3-Reranker-0.6B-4bit"
    assert gate.threshold == 0.61
    assert gate._model is None
    assert "specific personal proposition" in DEFAULT_MEMORY_RERANK_INSTRUCTION
    assert "general informational questions" in DEFAULT_MEMORY_RERANK_INSTRUCTION


def test_mlx_memory_reranker_document_preserves_specific_memory_proposition():
    candidate = {
        "belief": {
            "topic_key": "job_patient_interaction_level",
            "text": "The user's current job involves little patient interaction.",
            "open_question": "Will the user seek a more patient-facing role?",
        }
    }

    document = MLXQwen3MemoryReranker._document(candidate)

    assert "Topic: job patient interaction level" in document
    assert "Personal memory proposition:" in document
    assert "little patient interaction" in document
    assert "Open question:" in document


def test_mlx_memory_reranker_reset_clears_turn_diagnostics():
    gate = MLXQwen3MemoryReranker()
    gate.last_error = "old"
    gate.last_decisions = [{"index": 0}]
    gate.last_metrics = {"total_ms": 1}

    gate.reset()

    assert gate.last_error is None
    assert gate.last_decisions == []
    assert gate.last_metrics == {}
