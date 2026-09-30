from __future__ import annotations

import logging
from math import fsum
from typing import Any

import httpx

from aisha.memory.relevance import OllamaMemoryRelevanceGate

logger = logging.getLogger(__name__)
SEMANTIC_PIPELINE_VERSION = "hybrid-final-gate-v8-predicate-preserving"


class OllamaSemanticMemoryRetriever:
    """Local embedding retrieval plus optional final conversational relevance gate."""

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

    async def _score_beliefs(
        self,
        user_text: str,
        beliefs: list[dict],
        *,
        exclude_belief_ids: set[str] | None = None,
    ) -> list[tuple[float, str, dict]]:
        excluded = exclude_belief_ids or set()
        candidates = [
            belief
            for belief in beliefs
            if belief.get("evidence_status") == "verified"
            and str(belief.get("belief_id", "")) not in excluded
        ]
        if not candidates or not user_text.strip():
            return []

        missing: list[tuple[str, dict, str]] = []
        for belief in candidates:
            key = self._cache_key(belief)
            if key not in self._belief_vectors:
                missing.append((key, belief, self._document_text(belief)))

        inputs = [self._query_text(user_text)]
        inputs.extend(document for _key, _belief, document in missing)
        vectors = await self._embed(inputs)
        query_vector = vectors[0]
        for (key, _belief, _document), vector in zip(missing, vectors[1:], strict=True):
            self._belief_vectors[key] = vector

        scored = [
            (
                self._dot(query_vector, self._belief_vectors[self._cache_key(belief)]),
                str(belief.get("updated_at", "")),
                belief,
            )
            for belief in candidates
        ]
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return scored

    async def recall_hybrid(
        self,
        user_text: str,
        beliefs: list[dict],
        *,
        lexical_candidates: list[dict],
        total_limit: int = 4,
        candidate_limit: int = 8,
    ) -> list[dict]:
        """Gate the union of lexical and semantic candidates in one batch."""
        if self.relevance_gate is not None:
            self.relevance_gate.reset()

        scored: list[tuple[float, str, dict]] = []
        if self.disabled_reason is None:
            try:
                scored = await self._score_beliefs(user_text, beliefs)
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 - lexical candidates can still work
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Semantic memory scoring unavailable: %s", self.last_error)

        merged: dict[str, dict] = {}
        for detail in lexical_candidates:
            belief = detail["belief"]
            belief_id = str(belief.get("belief_id", ""))
            merged[belief_id] = {
                "belief": belief,
                "candidate_sources": ["lexical"],
                "lexical_score": float(detail.get("score", 0.0)),
                "semantic_score": None,
                "matched_tokens": list(detail.get("matched_tokens", [])),
                "ignored_low_information_tokens": list(
                    detail.get("ignored_low_information_tokens", [])
                ),
            }

        semantic_rows = [
            row for row in scored if row[0] >= self.candidate_floor
        ][:candidate_limit]
        for score, _updated, belief in semantic_rows:
            belief_id = str(belief.get("belief_id", ""))
            existing = merged.get(belief_id)
            if existing is None:
                merged[belief_id] = {
                    "belief": belief,
                    "candidate_sources": ["semantic"],
                    "lexical_score": None,
                    "semantic_score": round(score, 4),
                    "matched_tokens": [],
                    "ignored_low_information_tokens": [],
                }
            else:
                existing["candidate_sources"].append("semantic")
                existing["semantic_score"] = round(score, 4)

        pool = list(merged.values())
        pool.sort(
            key=lambda item: (
                len(item["candidate_sources"]),
                float(item["semantic_score"] or 0.0),
                float(item["lexical_score"] or 0.0),
            ),
            reverse=True,
        )
        # Always preserve every lexical candidate in the final judgment.
        # candidate_limit caps semantic expansion, not lexical evidence.
        pool = pool[: candidate_limit + len(lexical_candidates)]

        decisions: list[dict]
        if pool and self.relevance_gate is not None:
            decisions = await self.relevance_gate.judge(user_text, pool)
        else:
            decisions = [
                {
                    "index": index,
                    "relevant": False,
                    "reason": "relevance_gate_unavailable",
                }
                for index in range(len(pool))
            ]

        selected: list[dict] = []
        candidate_diagnostics: list[dict] = []
        for candidate, decision in zip(pool, decisions, strict=True):
            sources = candidate["candidate_sources"]
            if sources == ["lexical"]:
                method = "lexical_reranked"
            elif sources == ["semantic"]:
                method = "semantic_reranked"
            else:
                method = "hybrid_reranked"

            selected_flag = bool(decision["relevant"])
            diagnostic = {
                "topic_key": candidate["belief"].get("topic_key"),
                "lexical_score": candidate["lexical_score"],
                "semantic_score": candidate["semantic_score"],
                "candidate_sources": sources,
                "selected": selected_flag,
                "decision": "reranker_accept" if selected_flag else "reranker_reject",
                "reason": decision["reason"],
            }
            candidate_diagnostics.append(diagnostic)

            if selected_flag and len(selected) < total_limit:
                score = (
                    candidate["semantic_score"]
                    if candidate["semantic_score"] is not None
                    else candidate["lexical_score"]
                )
                selected.append({
                    "belief": candidate["belief"],
                    "method": method,
                    "score": score,
                    "lexical_score": candidate["lexical_score"],
                    "semantic_score": candidate["semantic_score"],
                    "candidate_sources": sources,
                    "matched_tokens": candidate["matched_tokens"],
                    "ignored_low_information_tokens": candidate[
                        "ignored_low_information_tokens"
                    ],
                    "reranker_reason": decision["reason"],
                })

        seen_topics = {
            str(candidate["topic_key"])
            for candidate in candidate_diagnostics
        }
        for score, _updated, belief in scored:
            topic = str(belief.get("topic_key", ""))
            if topic in seen_topics:
                continue
            if len(candidate_diagnostics) >= candidate_limit + len(lexical_candidates) + 3:
                break
            candidate_diagnostics.append({
                "topic_key": belief.get("topic_key"),
                "lexical_score": None,
                "semantic_score": round(score, 4),
                "candidate_sources": ["semantic"],
                "selected": False,
                "decision": (
                    "below_candidate_floor"
                    if score < self.candidate_floor
                    else "not_evaluated"
                ),
                "reason": (
                    "embedding_score_below_candidate_floor"
                    if score < self.candidate_floor
                    else "candidate_pool_limit"
                ),
            })

        self.last_candidates = candidate_diagnostics
        return selected

    async def recall(
        self,
        user_text: str,
        beliefs: list[dict],
        *,
        exclude_belief_ids: set[str] | None = None,
        remaining_limit: int | None = None,
    ) -> list[dict]:
        """Compatibility semantic-only path retained for focused unit tests."""
        if self.relevance_gate is not None:
            self.relevance_gate.reset()
        if self.disabled_reason is not None:
            self.last_candidates = []
            return []

        limit = min(self.limit, remaining_limit or self.limit)
        if limit <= 0:
            return []

        try:
            scored = await self._score_beliefs(
                user_text,
                beliefs,
                exclude_belief_ids=exclude_belief_ids,
            )
        except Exception as exc:  # noqa: BLE001 - semantic recall must fail open
            self.last_candidates = []
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Semantic memory recall unavailable: %s", self.last_error)
            return []

        selected: list[dict] = []
        decisions_by_topic: dict[str, dict] = {}
        for score, _updated, belief in scored:
            if len(selected) >= limit:
                break
            if score < self.threshold:
                continue
            selected.append({
                "belief": belief,
                "method": "semantic",
                "score": round(score, 4),
                "matched_tokens": [],
                "ignored_low_information_tokens": [],
            })
            decisions_by_topic[str(belief.get("topic_key", ""))] = {
                "selected": True,
                "decision": "direct_accept",
                "reason": "embedding_score_above_direct_threshold",
            }

        remaining = max(0, limit - len(selected))
        borderline = []
        if remaining:
            selected_ids = {
                str(detail["belief"].get("belief_id", ""))
                for detail in selected
            }
            for score, _updated, belief in scored:
                if len(borderline) >= remaining:
                    break
                if score < self.candidate_floor or score >= self.threshold:
                    continue
                if str(belief.get("belief_id", "")) in selected_ids:
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
                        "reranker_accept" if decision["relevant"] else "reranker_reject"
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
                decision = {
                    "selected": False,
                    "decision": (
                        "below_candidate_floor"
                        if score < self.candidate_floor
                        else "not_evaluated"
                    ),
                    "reason": (
                        "embedding_score_below_candidate_floor"
                        if score < self.candidate_floor
                        else "semantic_limit_reached"
                    ),
                }
            self.last_candidates.append({
                "topic_key": belief.get("topic_key"),
                "score": round(score, 4),
                **decision,
            })

        self.last_error = None
        return selected[:limit]

    def status(self) -> dict:
        return {
            "enabled": self.disabled_reason is None,
            "pipeline_version": SEMANTIC_PIPELINE_VERSION,
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
