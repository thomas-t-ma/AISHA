"""Second-stage semantic relevance gate for memory recall."""
from __future__ import annotations

import json

import httpx

RELEVANCE_GATE_PROMPT_VERSION = "personal-continuity-v3"

RELEVANCE_SYSTEM = """You are a conservative relevance judge for conversational memory.
Your job is to prevent irrelevant personal memories from leaking into a response.

FIRST classify the CURRENT message itself, without using candidate memories to
fill in missing context. Only after that may you judge individual candidates.

Return ONLY JSON:
{"message_scope":"personal_anchored|general_informational|ambiguous_unanchored",
 "scope_reason":"<=12 words",
 "decisions":[{"index":0,"relevant":true|false,"reason":"<=12 words"}]}

GLOBAL SCOPE CLASSIFICATION

personal_anchored:
- The CURRENT message itself identifies a specific personal situation,
  preference, decision, goal, relationship, problem, or ongoing thread.
- It can name it directly or strongly paraphrase it.
- Examples: "I want more patient interaction in my current job"; "My weekday
  schedule leaves no room for volunteering"; "I still plan to apply to medical
  school"; "For my PC I don't want mystery components."

general_informational:
- The message asks for general facts, explanation, advice, definitions,
  recommendations, statistics, or category-level information without grounding
  the question in the user's own specific situation.
- Topic overlap with a candidate memory NEVER makes a general question personal.
- Examples: "Why do some healthcare jobs have little patient contact?"; "What
  clay is easiest for beginners?"; "What switch types are best for an office
  keyboard?"; "How is AI used in clinical research?"
- A general question remains general even when the user happens to have a
  personal memory about exactly that topic.

ambiguous_unanchored:
- The message is first-person or continuity-sounding but does not independently
  identify which personal thread it refers to.
- Vague phrases such as "that project", "something quieter", "cheap out again",
  "more meaningful", "help people", or "not enough time after work" are not
  enough by themselves.
- Words such as "again", "still", "this time", "that", or "the usual" do NOT
  permit you to use candidate memory as the missing referent.

HARD GLOBAL RULE:
- If message_scope is general_informational, EVERY decision MUST be false.
- If message_scope is ambiguous_unanchored, EVERY decision MUST be false.
- Only personal_anchored messages may have any true decisions.
- Determine message_scope from the CURRENT message BEFORE considering which
  memories happen to be available.

CANDIDATE RULES FOR personal_anchored MESSAGES

Use this MEMORY-BLIND TEST for each candidate:
1. Hide the candidate memory.
2. Read only the CURRENT message.
3. Identify the specific personal proposition or thread present in the message.
4. Reveal the candidate.
5. Accept only if the candidate's DEFINING PROPOSITION matches that independently
   identified thread and materially improves the response.

The candidate may confirm or add known detail, but it may NOT supply the missing
topic, referent, motive, domain, or situation that makes itself seem relevant.

Reject:
- merely topical or lexical similarity;
- adjacent personal threads;
- broad motives used to infer narrower memories;
- a memory that merely explains why the user might have asked;
- optional callbacks that do not materially help answer the message.

A broad motive does not activate a narrower memory. "I want meaningful work"
does not by itself activate low patient interaction, medical school, research,
volunteering, or remote-work memories.

For multi-topic personal messages, accept each memory only when its own defining
proposition is independently present. Do not collapse two explicit clauses into
one: a message can legitimately activate multiple memories.

Keep each reason extremely short (12 words maximum). Good rejection reasons:
"general question, not personal continuity", "candidate supplies missing
referent", "broad motive only", "adjacent personal thread", "ambiguous without
current-message anchor".

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
        self.last_message_scope: str | None = None
        self.last_scope_reason: str | None = None
        self.last_metrics: dict = {}

    def reset(self) -> None:
        self.last_error = None
        self.last_decisions = []
        self.last_message_scope = None
        self.last_scope_reason = None
        self.last_metrics = {}

    @staticmethod
    def _parse(raw: str, expected: int) -> dict | None:
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

        message_scope = payload.get("message_scope")
        scope_reason = payload.get("scope_reason")
        valid_scopes = {
            "personal_anchored",
            "general_informational",
            "ambiguous_unanchored",
        }
        if message_scope not in valid_scopes or not isinstance(scope_reason, str):
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
        if [row["index"] for row in parsed] != list(range(expected)):
            return None

        # Enforce the model's global scope classification. Candidate-level
        # topical attraction cannot override a general or ambiguous message.
        if message_scope != "personal_anchored":
            parsed = [
                {
                    "index": row["index"],
                    "relevant": False,
                    "reason": (
                        "general question, not personal continuity"
                        if message_scope == "general_informational"
                        else "ambiguous without current-message anchor"
                    ),
                }
                for row in parsed
            ]

        return {
            "message_scope": message_scope,
            "scope_reason": scope_reason.strip()[:240],
            "decisions": parsed,
        }

    async def judge(self, user_text: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            self.last_decisions = []
            self.last_message_scope = None
            self.last_scope_reason = None
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
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 320,
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
            parsed = self._parse(raw, len(candidates))
            if parsed is None:
                raise ValueError("relevance_gate_invalid_response")
            decisions = parsed["decisions"]
            self.last_message_scope = str(parsed["message_scope"])
            self.last_scope_reason = str(parsed["scope_reason"])
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
            self.last_message_scope = None
            self.last_scope_reason = None
            self.last_metrics = {}
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.last_decisions

    def status(self) -> dict:
        return {
            "enabled": True,
            "model": self.model,
            "prompt_version": RELEVANCE_GATE_PROMPT_VERSION,
            "message_scope": self.last_message_scope,
            "scope_reason": self.last_scope_reason,
            "last_error": self.last_error,
            "last_decisions": self.last_decisions,
            "last_metrics": self.last_metrics,
        }
