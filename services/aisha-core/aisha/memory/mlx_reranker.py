from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any


DEFAULT_MEMORY_RERANK_INSTRUCTION = (
    "Given a current user message, determine whether a candidate memory's "
    "specific personal proposition is directly active in the message. Relevant "
    "only if the user states, updates, contradicts, questions, or strongly "
    "paraphrases the same personal fact, goal, preference, problem, or unresolved "
    "thread. Reject broad topical similarity, adjacent motivations, and general "
    "informational questions that merely share terms."
)


class MLXQwen3MemoryReranker:
    """Apple-Silicon Qwen3 reranker using yes/no logit scoring."""

    def __init__(
        self,
        *,
        model: str = "mlx-community/Qwen3-Reranker-0.6B-4bit",
        threshold: float = 0.5,
        instruction: str = DEFAULT_MEMORY_RERANK_INSTRUCTION,
    ) -> None:
        self.model_name = model
        self.threshold = threshold
        self.instruction = instruction.strip()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._mx: Any | None = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None
        self._true_id: int | None = None
        self._false_id: int | None = None
        self.last_error: str | None = None
        self.last_decisions: list[dict] = []
        self.last_metrics: dict = {}

    def reset(self) -> None:
        self.last_error = None
        self.last_decisions = []
        self.last_metrics = {}

    def _ensure_loaded(self) -> float:
        if self._model is not None:
            return 0.0

        started = perf_counter()
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "MLX memory reranker requires the 'mac-memory' optional dependency. "
                "Install with: pip install -e '.[dev,mac-memory]'"
            ) from exc

        model, tokenizer = load(self.model_name)
        hf = getattr(tokenizer, "_tokenizer", tokenizer)

        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            "and the Instruct provided. Note that the answer can only be "
            '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self._model = model
        self._tokenizer = hf
        self._mx = mx
        self._prefix_tokens = hf.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = hf.encode(suffix, add_special_tokens=False)
        self._true_id = int(hf.convert_tokens_to_ids("yes"))
        self._false_id = int(hf.convert_tokens_to_ids("no"))
        return (perf_counter() - started) * 1000

    @staticmethod
    def _document(candidate: dict) -> str:
        belief = candidate["belief"]
        topic = str(belief.get("topic_key", "")).replace("_", " ")
        text = str(belief.get("text", "")).strip()
        question = str(belief.get("open_question") or "").strip()
        parts = [f"Topic: {topic}", f"Personal memory proposition: {text}"]
        if question:
            parts.append(f"Open question: {question}")
        return "\n".join(parts)

    def _score_sync(self, query: str, document: str) -> float:
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._mx is not None
        assert self._prefix_tokens is not None
        assert self._suffix_tokens is not None
        assert self._true_id is not None
        assert self._false_id is not None

        content = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        ids = (
            self._prefix_tokens
            + self._tokenizer.encode(content, add_special_tokens=False)
            + self._suffix_tokens
        )
        logits = self._model(self._mx.array([ids]))[:, -1, :]
        pair = self._mx.stack(
            [logits[0, self._false_id], logits[0, self._true_id]]
        )
        probability = self._mx.exp(
            (pair - self._mx.logsumexp(pair))[1]
        )
        self._mx.eval(probability)
        return float(probability.item())

    async def judge(self, user_text: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            self.reset()
            return []

        started = perf_counter()
        try:
            load_ms = await asyncio.to_thread(self._ensure_loaded)
            score_started = perf_counter()
            scores: list[float] = []
            for candidate in candidates:
                score = await asyncio.to_thread(
                    self._score_sync,
                    user_text,
                    self._document(candidate),
                )
                scores.append(score)
            score_ms = (perf_counter() - score_started) * 1000

            decisions = [
                {
                    "index": index,
                    "relevant": score >= self.threshold,
                    "reason": (
                        f"reranker_score={score:.4f} threshold={self.threshold:.2f}"
                    ),
                    "reranker_score": round(score, 6),
                }
                for index, score in enumerate(scores)
            ]
            self.last_decisions = decisions
            self.last_error = None
            self.last_metrics = {
                "total_ms": round((perf_counter() - started) * 1000, 3),
                "load_ms": round(load_ms, 3),
                "score_ms": round(score_ms, 3),
                "pairs": len(candidates),
                "threshold": self.threshold,
            }
            return decisions
        except Exception as exc:  # noqa: BLE001 - recall must never break chat
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_decisions = [
                {
                    "index": index,
                    "relevant": False,
                    "reason": "mlx_reranker_unavailable",
                }
                for index in range(len(candidates))
            ]
            self.last_metrics = {
                "total_ms": round((perf_counter() - started) * 1000, 3),
                "pairs": len(candidates),
                "threshold": self.threshold,
            }
            return self.last_decisions

    def status(self) -> dict:
        return {
            "enabled": True,
            "backend": "mlx_qwen3_reranker",
            "model": self.model_name,
            "threshold": self.threshold,
            "last_error": self.last_error,
            "last_decisions": self.last_decisions,
            "last_metrics": self.last_metrics,
        }
