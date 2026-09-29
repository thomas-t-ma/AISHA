from __future__ import annotations

import asyncio
from collections.abc import Callable
from time import perf_counter

MEMORY_RERANK_INSTRUCTION = (
    "Decide whether the candidate memory's specific personal proposition is "
    "actively relevant to the user's current message. Approve only the same "
    "personal situation, decision, preference, goal, relationship, problem, "
    "or unresolved thread. Reject merely topical, adjacent, generic, or "
    "associative similarity, and reject general informational questions that "
    "only share vocabulary with the memory."
)


class CrossEncoderMemoryRelevanceGate:
    """Optional dedicated cross-encoder gate for memory-candidate benchmarking."""

    def __init__(
        self,
        *,
        model: str,
        threshold: float = 0.5,
        instruction: str = MEMORY_RERANK_INSTRUCTION,
        device: str | None = None,
        predictor: Callable[[str, list[str]], list[float]] | None = None,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.instruction = instruction.strip()
        self.device = device
        self._predictor = predictor
        self._cross_encoder = None
        self._torch = None
        self.last_error: str | None = None
        self.last_decisions: list[dict] = []
        self.last_metrics: dict = {}
        self.last_scores: list[float] = []

    def reset(self) -> None:
        self.last_error = None
        self.last_decisions = []
        self.last_metrics = {}
        self.last_scores = []

    @staticmethod
    def _document(candidate: dict) -> str:
        belief = candidate["belief"]
        text = str(belief.get("text", "")).strip()
        question = str(belief.get("open_question") or "").strip()
        return text if not question else f"{text}\nOpen thread: {question}"

    def _ensure_model(self) -> None:
        if self._predictor is not None or self._cross_encoder is not None:
            return

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Dedicated reranker dependencies are not installed. "
                "Install AISHA Core with the 'rerank' extra."
            ) from exc

        device = self.device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self._torch = torch
        self._cross_encoder = CrossEncoder(
            self.model,
            prompts={"memory_relevance": self.instruction},
            default_prompt_name="memory_relevance",
            device=device,
            max_length=1024,
        )
        self.device = device

    def _predict(self, query: str, documents: list[str]) -> list[float]:
        if self._predictor is not None:
            return [float(score) for score in self._predictor(query, documents)]

        self._ensure_model()
        assert self._cross_encoder is not None
        assert self._torch is not None
        scores = self._cross_encoder.predict(
            [(query, document) for document in documents],
            activation_fn=self._torch.nn.Sigmoid(),
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    async def judge(self, user_text: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            self.reset()
            return []

        documents = [self._document(candidate) for candidate in candidates]
        started = perf_counter()
        try:
            scores = await asyncio.to_thread(self._predict, user_text, documents)
            elapsed_ms = round((perf_counter() - started) * 1000, 3)
            if len(scores) != len(candidates):
                raise ValueError("cross_encoder_score_count_mismatch")

            self.last_scores = scores
            self.last_metrics = {
                "total_ms": elapsed_ms,
                "pairs": len(candidates),
                "device": self.device,
            }
            self.last_decisions = [
                {
                    "index": index,
                    "relevant": score >= self.threshold,
                    "reason": f"cross_encoder_score={score:.4f}",
                }
                for index, score in enumerate(scores)
            ]
            self.last_error = None
            return self.last_decisions
        except Exception as exc:  # noqa: BLE001 - benchmark gate fails closed
            self.last_scores = []
            self.last_metrics = {}
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_decisions = [
                {
                    "index": index,
                    "relevant": False,
                    "reason": "cross_encoder_unavailable",
                }
                for index in range(len(candidates))
            ]
            return self.last_decisions

    def status(self) -> dict:
        return {
            "enabled": True,
            "kind": "cross_encoder",
            "model": self.model,
            "threshold": self.threshold,
            "device": self.device,
            "last_error": self.last_error,
            "last_decisions": self.last_decisions,
            "last_scores": self.last_scores,
            "last_metrics": self.last_metrics,
        }
