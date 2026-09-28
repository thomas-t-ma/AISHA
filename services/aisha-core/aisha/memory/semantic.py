from __future__ import annotations

import logging
from math import fsum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OllamaSemanticMemoryRetriever:
    """Small local embedding retriever for verified learned beliefs."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        threshold: float = 0.72,
        limit: int = 2,
        keep_alive: str | int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.threshold = threshold
        self.limit = max(1, limit)
        self.keep_alive = keep_alive
        self._belief_vectors: dict[str, list[float]] = {}
        self.last_error: str | None = None
        self.disabled_reason: str | None = None

    @staticmethod
    def _cache_key(belief: dict) -> str:
        return "|".join(
            [
                str(belief.get("belief_id", "")),
                str(belief.get("revision", "")),
                str(belief.get("topic_key", "")),
                str(belief.get("text", "")),
                str(belief.get("open_question") or ""),
            ]
        )

    @staticmethod
    def _document_text(belief: dict) -> str:
        topic = str(belief.get("topic_key", "")).replace("_", " ").replace("-", " ")
        text = str(belief.get("text", "")).strip()
        question = str(belief.get("open_question") or "").strip()
        pieces = [piece for piece in (topic, text, question) if piece]
        return ". ".join(pieces)

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            raise ValueError("Embedding dimension mismatch")
        # Ollama's /api/embed vectors are L2-normalized, so dot product is cosine.
        return float(fsum(a * b for a, b in zip(left, right, strict=True)))

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": inputs,
            "truncate": True,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        timeout = httpx.Timeout(connect=5.0, read=30.0, write=15.0, pool=15.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}/api/embed", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                self.disabled_reason = "embedding_model_unavailable"
            raise
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise ValueError("Ollama embed response has unexpected shape")

        parsed: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise ValueError("Ollama embed response contains an invalid vector")
            parsed.append([float(value) for value in vector])
        return parsed

    async def recall(
        self,
        user_text: str,
        beliefs: list[dict],
        *,
        exclude_belief_ids: set[str] | None = None,
        remaining_limit: int | None = None,
    ) -> list[dict]:
        if self.disabled_reason is not None:
            return []

        excluded = exclude_belief_ids or set()
        candidates = [
            belief
            for belief in beliefs
            if belief.get("evidence_status") == "verified"
            and str(belief.get("belief_id", "")) not in excluded
        ]
        if not candidates or not user_text.strip():
            return []

        limit = min(self.limit, remaining_limit or self.limit)
        if limit <= 0:
            return []

        missing: list[tuple[str, dict, str]] = []
        for belief in candidates:
            key = self._cache_key(belief)
            if key not in self._belief_vectors:
                missing.append((key, belief, self._document_text(belief)))

        try:
            inputs = [user_text]
            inputs.extend(document for _key, _belief, document in missing)
            vectors = await self._embed(inputs)
            query_vector = vectors[0]
            for (key, _belief, _document), vector in zip(
                missing, vectors[1:], strict=True
            ):
                self._belief_vectors[key] = vector

            ranked: list[tuple[float, str, dict]] = []
            for belief in candidates:
                score = self._dot(query_vector, self._belief_vectors[self._cache_key(belief)])
                if score < self.threshold:
                    continue
                ranked.append((
                    score,
                    str(belief.get("updated_at", "")),
                    {
                        "belief": belief,
                        "method": "semantic",
                        "score": round(score, 4),
                        "matched_tokens": [],
                        "ignored_low_information_tokens": [],
                    },
                ))

            ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
            self.last_error = None
            return [detail for _score, _updated, detail in ranked[:limit]]
        except Exception as exc:  # noqa: BLE001 - semantic recall must fail open
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Semantic memory recall unavailable: %s", self.last_error)
            return []

    def status(self) -> dict:
        return {
            "enabled": self.disabled_reason is None,
            "model": self.model,
            "threshold": self.threshold,
            "limit": self.limit,
            "cached_beliefs": len(self._belief_vectors),
            "last_error": self.last_error,
            "disabled_reason": self.disabled_reason,
        }
