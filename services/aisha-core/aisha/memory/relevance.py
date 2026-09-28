"""Second-stage semantic relevance gate for memory recall."""
from __future__ import annotations

import json

import httpx

RELEVANCE_SYSTEM = """You are a conservative relevance judge for conversational memory.
Decide whether each candidate personal memory is directly useful for responding
to the user's CURRENT message.

Return ONLY JSON:
{"decisions":[{"index":0,"relevant":true|false,"reason":"<=12 words"}]}

Relevant means the CURRENT message is actually about the user's same underlying
personal situation, decision, preference, goal, relationship, problem, or
unresolved thread. Different wording is fine. A current message may contradict,
update, question, or indirectly describe that same personal thread.

Be strict about PERSONAL CONTINUITY:
- First identify the candidate memory's DEFINING PROPOSITION: the specific
  relation it claims about the user, not merely its broad topic.
- Approve only when the CURRENT message states, updates, contradicts, questions,
  or strongly paraphrases that defining proposition. If a required relation
  would have to be invented, inferred from a broad motive, or imported from the
  memory itself, reject it.
- Do not turn one personal thread into a second adjacent thread. Wanting more
  direct human contact does not by itself imply a volunteering constraint;
  wanting a quieter keyboard does not by itself invoke a general computer-part
  quality preference.
- A general informational question is not personal continuity merely because it
  shares words with a memory. Example: asking how medical-school accreditation
  works does NOT make a personal medical-school application goal relevant.
- A broader category is not enough. A healthcare or AI question does NOT
  automatically make every healthcare, school, or research memory relevant.
- For multi-topic messages, approve each memory only if that specific defining
  proposition is independently present in the current message.

Reject merely topical, generic, adjacent, motivational, or associative
similarity. Shared ideas such as work, helping people, food, computers,
healthcare, school, projects, feelings, time, or wanting something are not
enough by themselves. Do not force a callback merely because a memory could be
mentioned. Ask two questions: (1) what exact personal proposition does this
memory add, and (2) is that proposition actually active in the CURRENT message?
If the answer to (2) is not clearly yes, reject it.

Keep each reason extremely short (12 words maximum). The reason is diagnostic,
not an explanation for the user. Examples: "same patient-contact problem",
"adjacent work issue only", "general question, not personal continuity".

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
        self.last_metrics: dict = {}

    def reset(self) -> None:
        self.last_error = None
        self.last_decisions = []
        self.last_metrics = {}

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
                "embedding_score": candidate.get("semantic_score", candidate.get("score")),
                "lexical_score": candidate.get("lexical_score"),
                "candidate_sources": candidate.get("candidate_sources"),
            }
            for index, candidate in enumerate(candidates)
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 180,
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
            self.last_metrics = {
                "total_ms": round(float(data.get("total_duration", 0)) / 1_000_000, 3),
                "load_ms": round(float(data.get("load_duration", 0)) / 1_000_000, 3),
                "prompt_eval_ms": round(
                    float(data.get("prompt_eval_duration", 0)) / 1_000_000, 3
                ),
                "eval_ms": round(float(data.get("eval_duration", 0)) / 1_000_000, 3),
                "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
                "output_tokens": int(data.get("eval_count", 0) or 0),
            }
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
            self.last_metrics = {}
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.last_decisions

    def status(self) -> dict:
        return {
            "enabled": True,
            "model": self.model,
            "last_error": self.last_error,
            "last_decisions": self.last_decisions,
            "last_metrics": self.last_metrics,
        }
