"""Second-stage semantic relevance gate for memory recall."""
from __future__ import annotations

import json

import httpx

RELEVANCE_SYSTEM = """You are a conservative relevance judge for conversational memory.
Decide whether each candidate personal memory is directly useful for responding
to the user's CURRENT message.

Return ONLY JSON:
{"decisions":[{"index":0,"relevant":true|false,"reason":"short explanation"}]}

Relevant means the current message and memory concern the same underlying
personal situation, decision, preference, goal, relationship, problem, or
unresolved thread. They may use very different wording. A current message may
also contradict, update, question, or indirectly describe the same situation.

Reject merely topical, generic, or associative similarity. Shared ideas such as
work, people, food, feelings, time, or wanting something are not enough by
themselves. Do not force a callback merely because a memory could be mentioned.
If recalling the memory would not materially improve the response, reject it.

Judge RELEVANCE only. Do not decide whether the memory is true, current, or
factually verified; another subsystem handles evidence integrity. If uncertain,
prefer false.
"""


class OllamaMemoryRelevanceGate:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        keep_alive: str | int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.last_error: str | None = None
        self.last_decisions: list[dict] = []

    def reset(self) -> None:
        self.last_error = None
        self.last_decisions = []

    @staticmethod
    def _parse(raw: str, expected: int) -> list[dict] | None:
        raw = raw.strip()
        fence = chr(96) * 3
        if raw.startswith(fence) and raw.endswith(fence):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[0].lower() in {fence, fence + "json"}:
                raw = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        decisions = payload.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != expected:
            return None

        parsed: list[dict] = []
        seen: set[int] = set()
        for decision in decisions:
            if not isinstance(decision, dict):
                return None
            index = decision.get("index")
            relevant = decision.get("relevant")
            reason = decision.get("reason")
            if (
                not isinstance(index, int)
                or index < 0
                or index >= expected
                or index in seen
                or type(relevant) is not bool
                or not isinstance(reason, str)
            ):
                return None
            seen.add(index)
            parsed.append({
                "index": index,
                "relevant": relevant,
                "reason": reason.strip()[:240],
            })

        parsed.sort(key=lambda row: row["index"])
        return parsed if [row["index"] for row in parsed] == list(range(expected)) else None

    async def judge(self, user_text: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            self.last_decisions = []
            self.last_error = None
            return []

        candidate_payload = [
            {
                "index": index,
                "topic_key": candidate["belief"].get("topic_key"),
                "memory": candidate["belief"].get("text"),
                "open_question": candidate["belief"].get("open_question"),
                "embedding_score": candidate.get("score"),
            }
            for index, candidate in enumerate(candidates)
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 220,
                "num_ctx": 4096,
            },
            "messages": [
                {"role": "system", "content": RELEVANCE_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_user_message": user_text,
                            "candidates": candidate_payload,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        try:
            timeout = httpx.Timeout(connect=5.0, read=30.0, write=15.0, pool=15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
            raw = data.get("message", {}).get("content")
            if not isinstance(raw, str):
                raise TypeError("relevance_gate_missing_response")
            decisions = self._parse(raw, len(candidates))
            if decisions is None:
                raise ValueError("relevance_gate_invalid_response")
            self.last_decisions = decisions
            self.last_error = None
            return decisions
        except Exception as exc:  # noqa: BLE001 - recall must never break chat
            self.last_decisions = [
                {
                    "index": index,
                    "relevant": False,
                    "reason": "relevance_gate_unavailable",
                }
                for index in range(len(candidates))
            ]
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.last_decisions

    def status(self) -> dict:
        return {
            "enabled": True,
            "model": self.model,
            "last_error": self.last_error,
            "last_decisions": self.last_decisions,
        }
