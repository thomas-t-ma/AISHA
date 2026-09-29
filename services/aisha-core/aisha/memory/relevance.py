"""Second-stage semantic relevance gate for memory recall."""
from __future__ import annotations

import json

import httpx

RELEVANCE_GATE_PROMPT_VERSION = "personal-continuity-v2"

RELEVANCE_SYSTEM = """You are a conservative relevance judge for conversational memory.
Your job is to prevent irrelevant personal memories from leaking into a response.

For each candidate, decide whether its specific personal proposition is needed
or clearly useful for responding to the user's CURRENT message.

Return ONLY JSON:
{"decisions":[{"index":0,"relevant":true|false,"reason":"<=12 words"}]}

CORE RULE: the CURRENT message must supply the evidence for relevance. The
candidate memory may confirm or add known detail, but it may NOT supply the
missing topic, referent, motive, or situation that makes itself seem relevant.

Use this MEMORY-BLIND TEST for every candidate:
1. Hide the candidate memory.
2. Read only the CURRENT message.
3. Ask what specific personal situation, preference, decision, goal, problem,
   relationship, or unresolved thread the message itself identifies.
4. Reveal the candidate. Accept it only if its defining proposition matches that
   independently identifiable personal thread.

If the current message does not identify the thread without help from the
candidate, REJECT it.

GENERAL INFORMATION RULE:
- A general factual, explanatory, educational, demographic, market, technical,
  travel, food, hobby, or "how does X work?" question is NOT personal continuity
  merely because the user has a memory about X.
- If the question can be answered normally without knowing the candidate memory,
  reject the memory.
- Do not reinterpret a general question as secretly being about the user's own
  history, plans, preferences, or motives.
- Example: "How does medical-school accreditation work?" does NOT activate a
  personal medical-school application goal.
- Example: "What clay is easiest for beginners?" does NOT activate a pottery
  hobby merely because the user likes pottery.
- Example: "Why do some healthcare jobs have little patient contact?" does NOT
  activate the user's own patient-contact history.

AMBIGUITY RULE:
- Vague phrases such as "that project", "something quieter", "cheap out again",
  "more meaningful", "help people", or "not enough time after work" do not
  identify a memory thread by themselves.
- Never use the candidate memory itself to resolve an ambiguous pronoun,
  ellipsis, object, motive, or domain.
- If two or more plausible personal memories could explain the message, reject
  each unless the CURRENT message contains an independent anchor that selects it.
- Recent conversation context could resolve ambiguity, but no such context is
  provided to this judge unless it appears in the CURRENT message.

PERSONAL CONTINUITY RULE:
- First identify the candidate's DEFINING PROPOSITION: the specific relation it
  claims about the user, not its broad subject.
- Accept when the CURRENT message explicitly states, strongly paraphrases,
  updates, contradicts, questions, or clearly continues that defining
  proposition.
- Different wording is fine; invented causal links are not.
- A broad motive does not activate a narrower memory. "I want meaningful work"
  does not by itself activate low patient interaction, medical school, research,
  volunteering, or remote-work memories.
- Do not turn one personal thread into an adjacent one. Wanting more patient
  contact does not imply a volunteering constraint. Planning medical school does
  not imply dissatisfaction with patient contact.
- For multi-topic messages, accept each memory only when its own defining
  proposition is independently present.

Before accepting, require BOTH:
A. THREAD EVIDENCE: the current message itself identifies the same personal
   proposition or thread without importing facts from the candidate.
B. RESPONSE VALUE: knowing the candidate would materially clarify, complete, or
   improve the response to this message rather than merely provide an optional
   callback.

Reject merely topical, associative, motivational, biographical, causal, or
"could be related" similarity. A memory explaining why the user might have asked
a question is not enough. If uncertain, prefer false.

Keep each reason extremely short (12 words maximum). Good rejection reasons:
"general question, not personal continuity", "candidate supplies missing
referent", "broad motive only", "adjacent personal thread", "ambiguous without
current-message anchor".

Judge RELEVANCE only. Do not decide whether the memory is true, current, or
factually verified; another subsystem handles evidence integrity.
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
            "prompt_version": RELEVANCE_GATE_PROMPT_VERSION,
            "last_error": self.last_error,
            "last_decisions": self.last_decisions,
            "last_metrics": self.last_metrics,
        }
