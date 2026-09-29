"""Two-stage semantic relevance gate for memory recall."""
from __future__ import annotations

import json
from typing import Callable

import httpx

RELEVANCE_GATE_PROMPT_VERSION = "personal-continuity-v6.1-two-stage-sparse"

MESSAGE_ANALYSIS_SYSTEM = """You analyze ONLY the user's current message before any
long-term memories are visible.

Return ONLY JSON:
{"message_scope":"personal_anchored|general_informational|ambiguous_unanchored",
 "scope_reason":"<=12 words",
 "propositions":[{"index":0,"text":"explicit personal proposition"}]}

Classify the message from its own words.

personal_anchored:
- The message itself expresses one or more specific personal situations,
  preferences, interests, values, decisions, goals, relationships, problems,
  updates, or ongoing threads.
- Specific first-person declarative preferences and interests count.
- For this scope, atomize the message into the smallest independent personal
  propositions actually expressed. Preserve meaningful qualifiers such as
  object, subtype, target, domain, time/status, modality, and relation.

general_informational:
- The message asks for general facts, explanations, definitions,
  recommendations, statistics, or category-level information without grounding
  the request in the user's own specific situation.
- Return propositions=[].

ambiguous_unanchored:
- The message sounds personal or continuous but does not independently identify
  the object, topic, goal, preference, situation, or referent.
- Vague references such as "that", "it", "something", "more time", "a change",
  "again", or "this time" do not identify a thread by themselves.
- Do not guess what missing context might be.
- Return propositions=[].

For personal_anchored, include only propositions stated by the message itself.
Do not add likely causes, consequences, motives, background, or inferred goals.
Use at most four propositions. If uncertain whether a specific personal thread
is independently identifiable, prefer ambiguous_unanchored.
"""

CANDIDATE_SELECTION_SYSTEM = """You map candidate memories onto a frozen list of
personal propositions extracted earlier WITHOUT access to any memories.

Return ONLY sparse JSON:
{"matches":[
  {"proposition_index":0,"candidate_index":3,
   "candidate_topic_key":"example_topic","reason":"<=12 words"}
]}

Return ZERO or ONE match per proposition. If no candidate faithfully matches a
proposition, omit that proposition from matches.

The proposition list is authoritative. Candidate memories may NEVER create,
reinterpret, broaden, narrow, or add a proposition.

For each proposition, choose AT MOST ONE candidate: the memory whose DEFINING
PROPOSITION is the closest faithful semantic match.

Require QUALIFIER FIDELITY:
- Preserve meaningful object, subtype, target, domain, time/status, modality,
  and relation qualifiers in the frozen proposition.
- A broader memory that drops a qualifier is weaker.
- A narrower memory that adds an unexpressed qualifier is weaker.
- A nearby goal, value, consequence, explanation, cause, preference, or context
  is not a match merely because it would make sense.
- Prefer semantic equivalence over word overlap.

Reject candidates that are merely compatible, adjacent, redundant, background,
a subset/superset, a plausible cause/consequence, or an inferred companion.

IMPORTANT OUTPUT CONTRACT:
- candidate_index MUST be the exact index shown on the chosen candidate.
- candidate_topic_key MUST exactly echo that candidate's topic_key.
- Never renumber candidates.
- Never output rejected candidates.
- Never output two matches for the same proposition.
- Never output one candidate for two propositions.
- Candidate retrieval order and score are not evidence of semantic relevance.

If uncertain, return no match for that proposition.
"""

# Backward-compatible inspection surface for tests/status tooling. Production
# inference uses the two prompts in separate requests.
RELEVANCE_SYSTEM = MESSAGE_ANALYSIS_SYSTEM + "\n\n" + CANDIDATE_SELECTION_SYSTEM


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
        self.last_propositions: list[dict] = []
        self.last_metrics: dict = {}
        self.last_protocol_phase: str | None = None
        self.last_protocol_response_preview: str | None = None

    def reset(self) -> None:
        self.last_error = None
        self.last_decisions = []
        self.last_message_scope = None
        self.last_scope_reason = None
        self.last_propositions = []
        self.last_metrics = {}
        self.last_protocol_phase = None
        self.last_protocol_response_preview = None

    @staticmethod
    def _strip_fence(raw: str) -> str:
        raw = raw.strip()
        fence = chr(96) * 3
        if raw.startswith(fence) and raw.endswith(fence):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[0].lower() in {fence, fence + "json"}:
                return "\n".join(lines[1:-1]).strip()
        return raw

    @classmethod
    def _parse_analysis(cls, raw: str) -> dict | None:
        try:
            payload = json.loads(cls._strip_fence(raw))
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None

        scope = payload.get("message_scope")
        reason = payload.get("scope_reason")
        valid_scopes = {
            "personal_anchored",
            "general_informational",
            "ambiguous_unanchored",
        }
        if scope not in valid_scopes or not isinstance(reason, str):
            return None

        propositions = payload.get("propositions")
        if not isinstance(propositions, list) or len(propositions) > 4:
            return None

        parsed: list[dict] = []
        seen: set[int] = set()
        for proposition in propositions:
            if not isinstance(proposition, dict):
                return None
            index = proposition.get("index")
            text = proposition.get("text")
            if (
                not isinstance(index, int)
                or index < 0
                or index in seen
                or not isinstance(text, str)
                or not text.strip()
            ):
                return None
            seen.add(index)
            parsed.append({"index": index, "text": text.strip()[:320]})

        parsed.sort(key=lambda row: row["index"])
        if [row["index"] for row in parsed] != list(range(len(parsed))):
            return None
        if scope == "personal_anchored" and not parsed:
            return None
        if scope != "personal_anchored" and parsed:
            return None

        return {
            "message_scope": scope,
            "scope_reason": reason.strip()[:240],
            "propositions": parsed,
        }

    @classmethod
    def _parse_selection(
        cls,
        raw: str,
        *,
        expected: int,
        proposition_count: int,
        candidate_topics: list[str],
    ) -> list[dict] | None:
        try:
            payload = json.loads(cls._strip_fence(raw))
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None

        matches = payload.get("matches")
        if not isinstance(matches, list) or len(matches) > proposition_count:
            return None

        selected_candidates: set[int] = set()
        selected_propositions: set[int] = set()
        selected_by_candidate: dict[int, dict] = {}

        for match in matches:
            if not isinstance(match, dict):
                return None
            proposition_index = match.get("proposition_index")
            candidate_index = match.get("candidate_index")
            candidate_topic_key = match.get("candidate_topic_key")
            reason = match.get("reason")

            if (
                not isinstance(proposition_index, int)
                or proposition_index < 0
                or proposition_index >= proposition_count
                or proposition_index in selected_propositions
                or not isinstance(candidate_index, int)
                or candidate_index < 0
                or candidate_index >= expected
                or candidate_index in selected_candidates
                or not isinstance(candidate_topic_key, str)
                or candidate_topic_key != candidate_topics[candidate_index]
                or not isinstance(reason, str)
            ):
                return None

            selected_propositions.add(proposition_index)
            selected_candidates.add(candidate_index)
            selected_by_candidate[candidate_index] = {
                "index": candidate_index,
                "relevant": True,
                "proposition_index": proposition_index,
                "reason": reason.strip()[:240],
            }

        return [
            selected_by_candidate.get(
                index,
                {
                    "index": index,
                    "relevant": False,
                    "proposition_index": None,
                    "reason": "not selected for any frozen proposition",
                },
            )
            for index in range(expected)
        ]

    @staticmethod
    def _metrics(data: dict, attempts: int) -> dict:
        return {
            "total_ms": round(float(data.get("total_duration", 0)) / 1_000_000, 3),
            "load_ms": round(float(data.get("load_duration", 0)) / 1_000_000, 3),
            "prompt_eval_ms": round(
                float(data.get("prompt_eval_duration", 0)) / 1_000_000, 3
            ),
            "eval_ms": round(float(data.get("eval_duration", 0)) / 1_000_000, 3),
            "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
            "output_tokens": int(data.get("eval_count", 0) or 0),
            "attempts": attempts,
        }

    async def _run_phase(
        self,
        *,
        phase: str,
        system_prompt: str,
        user_payload: dict,
        num_predict: int,
        parser: Callable[[str], dict | list[dict] | None],
    ) -> tuple[dict | list[dict], dict]:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": num_predict,
                "num_ctx": 4096,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
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

                raw = data.get("message", {}).get("content")
                if not isinstance(raw, str):
                    self.last_protocol_phase = phase
                    self.last_protocol_response_preview = None
                    final_error = TypeError(
                        f"relevance_gate_missing_response:{phase}"
                    )
                    continue

                parsed = parser(raw)
                if parsed is None:
                    self.last_protocol_phase = phase
                    self.last_protocol_response_preview = raw.strip()[:1000]
                    final_error = ValueError(
                        f"relevance_gate_invalid_response:{phase}"
                    )
                    continue

                self.last_protocol_phase = None
                self.last_protocol_response_preview = None
                return parsed, self._metrics(data, attempt + 1)
            except Exception as exc:  # noqa: BLE001 - recall must never break chat
                final_error = exc
                break

        if final_error is None:
            final_error = RuntimeError("relevance_gate_unknown_failure")
        raise final_error

    @staticmethod
    def _combine_metrics(analysis: dict, selection: dict | None) -> dict:
        selection = selection or {}
        additive = (
            "total_ms",
            "load_ms",
            "prompt_eval_ms",
            "eval_ms",
            "prompt_tokens",
            "output_tokens",
        )
        combined = {
            key: round(
                float(analysis.get(key, 0) or 0)
                + float(selection.get(key, 0) or 0),
                3,
            )
            for key in additive
        }
        combined.update(
            {
                "analysis_ms": float(analysis.get("total_ms", 0) or 0),
                "selection_ms": float(selection.get("total_ms", 0) or 0),
                "analysis_attempts": int(analysis.get("attempts", 0) or 0),
                "selection_attempts": int(selection.get("attempts", 0) or 0),
            }
        )
        combined["prompt_tokens"] = int(combined["prompt_tokens"])
        combined["output_tokens"] = int(combined["output_tokens"])
        return combined

    def _fail_closed(self, candidates: list[dict], exc: Exception) -> list[dict]:
        self.last_decisions = [
            {
                "index": index,
                "relevant": False,
                "proposition_index": None,
                "reason": "relevance_gate_unavailable",
            }
            for index in range(len(candidates))
        ]
        self.last_metrics = {}
        self.last_error = f"{type(exc).__name__}: {exc}"
        return self.last_decisions

    async def judge(self, user_text: str, candidates: list[dict]) -> list[dict]:
        self.reset()
        if not candidates:
            return []

        # Phase 1 is physically memory-blind: no candidate content is included.
        try:
            analysis, analysis_metrics = await self._run_phase(
                phase="analysis",
                system_prompt=MESSAGE_ANALYSIS_SYSTEM,
                user_payload={"current_user_message": user_text},
                num_predict=180,
                parser=self._parse_analysis,
            )
        except Exception as exc:  # noqa: BLE001 - recall must never break chat
            return self._fail_closed(candidates, exc)

        assert isinstance(analysis, dict)
        self.last_message_scope = str(analysis["message_scope"])
        self.last_scope_reason = str(analysis["scope_reason"])
        self.last_propositions = list(analysis["propositions"])

        if self.last_message_scope != "personal_anchored":
            reason = (
                "general question, not personal continuity"
                if self.last_message_scope == "general_informational"
                else "ambiguous without message anchor"
            )
            self.last_decisions = [
                {
                    "index": index,
                    "relevant": False,
                    "proposition_index": None,
                    "reason": reason,
                }
                for index in range(len(candidates))
            ]
            self.last_metrics = self._combine_metrics(analysis_metrics, None)
            self.last_error = None
            return self.last_decisions

        candidate_payload = [
            {
                "index": index,
                "topic_key": candidate["belief"].get("topic_key"),
                "memory": candidate["belief"].get("text"),
                "open_question": candidate["belief"].get("open_question"),
            }
            for index, candidate in enumerate(candidates)
        ]

        try:
            selection, selection_metrics = await self._run_phase(
                phase="selection",
                system_prompt=CANDIDATE_SELECTION_SYSTEM,
                user_payload={
                    "propositions": self.last_propositions,
                    "candidates": candidate_payload,
                },
                num_predict=320,
                parser=lambda raw: self._parse_selection(
                    raw,
                    expected=len(candidates),
                    proposition_count=len(self.last_propositions),
                    candidate_topics=[
                        str(candidate["belief"].get("topic_key", ""))
                        for candidate in candidates
                    ],
                ),
            )
        except Exception as exc:  # noqa: BLE001 - recall must never break chat
            return self._fail_closed(candidates, exc)

        assert isinstance(selection, list)
        self.last_decisions = selection
        self.last_metrics = self._combine_metrics(
            analysis_metrics,
            selection_metrics,
        )
        self.last_error = None
        return self.last_decisions

    def status(self) -> dict:
        return {
            "enabled": True,
            "model": self.model,
            "prompt_version": RELEVANCE_GATE_PROMPT_VERSION,
            "message_scope": self.last_message_scope,
            "scope_reason": self.last_scope_reason,
            "propositions": self.last_propositions,
            "last_error": self.last_error,
            "protocol_phase": self.last_protocol_phase,
            "protocol_response_preview": self.last_protocol_response_preview,
            "last_decisions": self.last_decisions,
            "last_metrics": self.last_metrics,
        }
