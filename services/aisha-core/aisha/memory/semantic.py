from __future__ import annotations

import logging
from math import fsum
from typing import Any

import httpx

from aisha.memory.relevance import OllamaMemoryRelevanceGate

logger = logging.getLogger(__name__)


class OllamaSemanticMemoryRetriever:
    """Small local embedding retriever for verified learned beliefs."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        threshold: float = 0.72,
        candidate_floor: float = 0.30,
        limit: int = 2,
        keep_alive: str | int | None = None,
        relevance_gate: OllamaMemoryRelevanceGate | None = None,
        query_instruction: str = (
            "Given a user's current message, retrieve a previously stated personal "
            "memory that is directly relevant and useful for responding. Prefer the "
            "same situation or underlying concern even when phrased differently; "
            "avoid merely topical or generic associations."
        ),
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.threshold = threshold
        self.candidate_floor = candidate_floor
        self.limit = max(1, limit)
        self.keep_alive = keep_alive
        self.relevance_gate = relevance_gate
        self.query_instruction = query_instruction.strip()
        self._belief_vectors: dict[str, list[float]] = {}
        self.last_error: str | None = None
        self.last_candidates: list[dict] = []
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

    def _query_text(self, user_text: str) -> str:
        # Qwen3-Embedding recommends an instruction on retrieval queries only;
        # documents remain unprefixed.
        return f"Instruct: {self.query_instruction}\nQuery: {user_text}"

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
            self.last_candidates = []
            return []

        excluded = exclude_belief_ids or set()
        candidates = [
            belief
            for belief in beliefs
            if belief.get("evidence_status") == "verified"
            and str(belief.get("belief_id", "")) not in excluded
        ]
        if not candidates or not user_text.strip():
            self.last_candidates = []
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
            inputs = [self._query_text(user_text)]
            inputs.extend(document for _key, _belief, document in missing)
            vectors = await self._embed(inputs)
            query_vector = vectors[0]
            for (key, _belief, _document), vector in zip(
                missing, vectors[1:], strict=True
            ):
                self._belief_vectors[key] = vector

            scored: list[tuple[float, str, dict]] = []
            for belief in candidates:
                score = self._dot(query_vector, self._belief_vectors[self._cache_key(belief)])
                scored.append((
                    score,
                    str(belief.get("updated_at", "")),
                    belief,
                ))

            scored.sort(key=lambda row: (row[0], row[1]), reverse=True)

            selected: list[dict] = []
            decisions_by_topic: dict[str, dict] = {}

            # Very strong embedding matches can be accepted directly. Mid-range
            # candidates need a second-stage relevance judgment. Low scores are
            # discarded before any extra model call.
            for score, _updated, belief in scored:
                if len(selected) >= limit:
                    break
                if score < self.threshold:
                    continue
                detail = {
                    "belief": belief,
                    "method": "semantic",
                    "score": round(score, 4),
                    "matched_tokens": [],
                    "ignored_low_information_tokens": [],
                }
                selected.append(detail)
                decisions_by_topic[str(belief.get("topic_key", ""))] = {
                    "selected": True,
                    "decision": "direct_accept",
                    "reason": "embedding_score_above_direct_threshold",
                }

            remaining = max(0, limit - len(selected))
            borderline: list[dict] = []
            if remaining:
                directly_selected_ids = {
                    str(detail["belief"].get("belief_id", ""))
                    for detail in selected
                }
                for score, _updated, belief in scored:
                    if len(borderline) >= remaining:
                        break
                    if score < self.candidate_floor or score >= self.threshold:
                        continue
                    if str(belief.get("belief_id", "")) in directly_selected_ids:
                        continue
                    borderline.append({
                        "belief": belief,
                        "method": "semantic_reranked",
                        "score": round(score, 4),
                        "matched_tokens": [],
                        "ignored_low_information_tokens": [],
                    })

            if borderline and self.relevance_gate is not None:
                decisions = await self.relevance_gate.judge(user_text, borderline)
                for candidate, decision in zip(borderline, decisions, strict=True):
                    topic = str(candidate["belief"].get("topic_key", ""))
                    decisions_by_topic[topic] = {
                        "selected": bool(decision["relevant"]),
                        "decision": (
                            "reranker_accept"
                            if decision["relevant"]
                            else "reranker_reject"
                        ),
                        "reason": decision["reason"],
                    }
                    if decision["relevant"] and len(selected) < limit:
                        selected.append({
                            **candidate,
                            "reranker_reason": decision["reason"],
                        })
            elif borderline:
                for candidate in borderline:
                    topic = str(candidate["belief"].get("topic_key", ""))
                    decisions_by_topic[topic] = {
                        "selected": False,
                        "decision": "below_direct_threshold",
                        "reason": "no_relevance_gate_configured",
                    }

            self.last_candidates = []
            for score, _updated, belief in scored[:3]:
                topic = str(belief.get("topic_key", ""))
                decision = decisions_by_topic.get(topic)
                if decision is None:
                    if score < self.candidate_floor:
                        decision = {
                            "selected": False,
                            "decision": "below_candidate_floor",
                            "reason": "embedding_score_below_candidate_floor",
                        }
                    else:
                        decision = {
                            "selected": False,
                            "decision": "not_evaluated",
                            "reason": "semantic_limit_reached",
                        }
                self.last_candidates.append({
                    "topic_key": belief.get("topic_key"),
                    "score": round(score, 4),
                    **decision,
                })

            self.last_error = None
            return selected[:limit]
        except Exception as exc:  # noqa: BLE001 - semantic recall must fail open
            self.last_candidates = []
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Semantic memory recall unavailable: %s", self.last_error)
            return []

    def status(self) -> dict:
        return {
            "enabled": self.disabled_reason is None,
            "model": self.model,
            "threshold": self.threshold,
            "candidate_floor": self.candidate_floor,
            "limit": self.limit,
            "query_instruction": self.query_instruction,
            "cached_beliefs": len(self._belief_vectors),
            "last_error": self.last_error,
            "last_candidates": self.last_candidates,
            "relevance_gate": (
                self.relevance_gate.status()
                if self.relevance_gate is not None
                else {"enabled": False}
            ),
            "disabled_reason": self.disabled_reason,
        }
