"""Second-stage semantic relevance gate for memory recall."""
from __future__ import annotations

import json

import httpx

RELEVANCE_GATE_PROMPT_VERSION = "personal-continuity-v5"

RELEVANCE_SYSTEM = """You are a conservative semantic judge for conversational memory.
Your job is to select the MINIMAL, MAXIMALLY FAITHFUL SET of personal memories
that directly matches the user's CURRENT message.

Judge candidates JOINTLY. Candidate retrieval rank or score is not evidence of
relevance; judge only the meaning of the current message and memory proposition.

Return ONLY JSON:
{"message_scope":"personal_anchored|general_informational|ambiguous_unanchored",
 "scope_reason":"<=12 words",
 "decisions":[{"index":0,"relevant":true|false,"reason":"<=12 words"}]}

STEP 1 — CLASSIFY THE CURRENT MESSAGE WITHOUT MEMORY

personal_anchored:
- The message itself expresses a specific personal situation, preference,
  interest, value, decision, goal, relationship, problem, update, or ongoing
  thread.
- Specific first-person declarative preferences and interests count as personal
  anchors even when the user is not asking for advice.

general_informational:
- The message asks for general facts, explanations, definitions,
  recommendations, statistics, or category-level information without grounding
  the request in the user's own specific situation.
- Topic overlap with personal memory never turns a general question personal.

ambiguous_unanchored:
- The message sounds personal or continuous but does not independently identify
  which personal thread it refers to.
- Vague references such as "that", "it", "something", "more time", "a change",
  "again", or "this time" cannot be resolved from candidate memory.

HARD SCOPE RULE:
- general_informational => every decision false.
- ambiguous_unanchored => every decision false.
- Only personal_anchored may select memories.

STEP 2 — ATOMIZE THE PERSONAL MESSAGE

Before looking at candidates, split the CURRENT message into the smallest
independent personal propositions actually expressed. Treat coordinated clauses
as separate propositions when they express distinct goals or preferences.

Examples of proposition structure:
- object + relation: "apply to medical school"
- preference + qualifier: "likes spicy Thai food"
- situation + qualifier: "little direct patient interaction at work"
- preference + object: "quiet linear keyboard"
- decision/update + object: "reconsidering moving to a country"

Do not invent a proposition that is merely implied, plausible, causal, or
helpful background.

STEP 3 — ALIGN CANDIDATES TO PROPOSITIONS

A candidate may be relevant=true only when its DEFINING PROPOSITION maps
directly to one of the explicit personal propositions from Step 2.

Require QUALIFIER FIDELITY:
- Preserve the user's distinguishing qualifiers whenever a candidate exists that
  does so.
- Qualifiers include the object, subtype, target, domain, time/status, modality,
  and relation expressed by the user.
- A broader memory that drops a meaningful qualifier is weaker than a memory
  that preserves it.
- A narrower memory that adds an unexpressed qualifier is also weaker.
- Do not substitute a nearby goal, value, consequence, explanation, or context
  for the proposition actually expressed.

Prefer semantic equivalence or the closest faithful paraphrase, not the memory
with the most overlapping words.

STEP 4 — SELECT THE MINIMAL MAXIMALLY FAITHFUL SET

For each explicit personal proposition, select at most the best candidate unless
multiple memories contribute genuinely non-overlapping information required by
that same proposition.

Reject candidates that are:
- merely compatible with the message;
- broader background facts about the same domain;
- narrower details not stated by the user;
- causes, consequences, motivations, or likely context;
- duplicates, subsets, supersets, restatements, or redundant companions;
- adjacent goals, interests, preferences, constraints, or biography.

A selected memory should survive this counterfactual:
"If this candidate were removed while the better-aligned candidate remained,
would any explicit proposition in the CURRENT message lose its best personal
continuity match?" If no, reject it as redundant.

For multi-topic messages, preserve each genuinely independent proposition.
Minimal does not mean one memory; it means one best match per expressed thread,
with no inferred or redundant extras.

MEMORY-BLIND SAFETY TEST

1. Hide all candidate memories.
2. Identify the personal propositions solely from the CURRENT message.
3. Reveal candidates.
4. Match candidates to those pre-existing propositions.
5. Reject any candidate that creates a new proposition rather than matching one.

Keep each reason extremely short (12 words maximum). Useful rejection reasons:
"general question, not personal continuity", "ambiguous without message anchor",
"broader than expressed proposition", "adds unexpressed qualifier",
"adjacent personal thread", "redundant with closer match",
"proposition not expressed".

Judge relevance only. Do not decide whether a memory is true or current; another
subsystem handles evidence integrity. If uncertain, prefer false.
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
            }
            for index, candidate in enumerate(candidates)
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            # The local MLX Ollama-compatible /api/chat endpoint used by AISHA
            # does not implement Ollama's structured-output "format" option.
            # The system prompt still requires raw JSON and _parse validates it.
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

        timeout = httpx.Timeout(connect=5.0, read=30.0, write=15.0, pool=15.0)
        final_error: Exception | None = None

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()

                metrics = {
                    "total_ms": round(float(data.get("total_duration", 0)) / 1_000_000, 3),
                    "load_ms": round(float(data.get("load_duration", 0)) / 1_000_000, 3),
                    "prompt_eval_ms": round(
                        float(data.get("prompt_eval_duration", 0)) / 1_000_000, 3
                    ),
                    "eval_ms": round(float(data.get("eval_duration", 0)) / 1_000_000, 3),
                    "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
                    "output_tokens": int(data.get("eval_count", 0) or 0),
                    "attempts": attempt + 1,
                }

                raw = data.get("message", {}).get("content")
                if not isinstance(raw, str):
                    final_error = TypeError("relevance_gate_missing_response")
                    if attempt == 0:
                        continue
                    break

                parsed = self._parse(raw, len(candidates))
                if parsed is None:
                    final_error = ValueError("relevance_gate_invalid_response")
                    if attempt == 0:
                        continue
                    break

                decisions = parsed["decisions"]
                self.last_message_scope = str(parsed["message_scope"])
                self.last_scope_reason = str(parsed["scope_reason"])
                self.last_decisions = decisions
                self.last_metrics = metrics
                self.last_error = None
                return decisions
            except Exception as exc:  # noqa: BLE001 - recall must never break chat
                final_error = exc
                break

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
        error = final_error or RuntimeError("relevance_gate_unknown_failure")
        self.last_error = f"{type(error).__name__}: {error}"
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
