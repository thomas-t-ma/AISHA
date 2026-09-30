"""Two-stage semantic relevance gate for memory recall."""
from __future__ import annotations

import json
from typing import Callable

import httpx

RELEVANCE_GATE_PROMPT_VERSION = "personal-continuity-v7-structured-axis"

MESSAGE_ANALYSIS_SYSTEM = """You analyze ONLY the user's current message before any
long-term memories are visible.

Return ONLY JSON:
{"message_scope":"personal_anchored|general_informational|ambiguous_unanchored",
 "scope_reason":"<=12 words",
 "propositions":[{"index":0,"text":"explicit personal proposition"}]}

Classify the message from its own words.

personal_anchored:
- The message itself expresses one or more specific, MEMORY-ADDRESSABLE personal
  situations, preferences, interests, values, decisions, goals, relationships,
  problems, updates, or ongoing threads.
- Specific first-person declarative preferences and interests count only when
  the object/domain/thread is identifiable from the message itself.
- MEMORY-ADDRESSABLE means an unfamiliar reader could describe what kind of
  personal memory would match WITHOUT seeing any candidate memories.
- If the message explicitly names the subject or domain, continuity language
  such as "still", "used to", "anymore", or "less than before" remains anchored
  to that named thread even when the previous state is implicit.
- A generic personal desire or state is NOT enough when its object/domain is
  missing or generic (for example: wanting "more outside work", "a change",
  "more time", "something better", or "to get back to it").
- For this scope, extract ONE proposition per independently storable personal
  fact. A proposition should correspond to one memory-addressable claim.
- SPLIT clauses when they express different objects, predicates, goals,
  decisions, or preferences and each clause could stand alone as a meaningful
  long-term memory.
- Do NOT merge distinct facts merely because one explains, motivates, enables,
  constrains, or gives the timing/purpose of the other.
- Purpose/time clauses can contain a second explicit personal fact. If both the
  main clause and that clause are independently memory-addressable, emit both.
- KEEP clauses together when they are only multiple qualifiers or attributes of
  the SAME object and relation, so splitting them would create fragments rather
  than distinct memories.
- Preserve meaningful qualifiers such as object, subtype, target, domain,
  time/status, modality, and relation.

general_informational:
- The message asks for general facts, explanations, definitions,
  recommendations, statistics, or category-level information without grounding
  the request in the user's own specific situation.
- Return propositions=[].

ambiguous_unanchored:
- The message sounds personal or continuous but does not independently identify
  a MEMORY-ADDRESSABLE object, topic, goal, preference, situation, or referent.
- This includes generic personal wishes or states whose missing domain could
  plausibly be completed by several unrelated memories.
- Vague references such as "that", "it", "something", "more time", "a change",
  "more outside work", "again", or "this time" do not identify a thread by
  themselves.
- Do not convert a generic wish into a specific topic such as hobbies, service,
  career, school, relationships, or health.
- Do not guess what missing context might be.
- Return propositions=[].

For personal_anchored, include only propositions stated by the message itself.
Do not add likely causes, consequences, motives, background, or inferred goals.
Use at most four propositions. If uncertain whether a specific personal thread
is independently identifiable, prefer ambiguous_unanchored.
"""

CANDIDATE_SELECTION_SYSTEM = """You evaluate candidate memories against frozen
personal propositions extracted earlier WITHOUT access to any memories.

Do NOT choose a winner. Return a sparse shortlist of plausible candidates with
structured labels so code can apply the final hard rules.

Return ONLY JSON:
{"evaluations":[
  {"proposition_index":0,"candidate_index":3,
   "candidate_topic_key":"example_topic",
   "relation":"same_fact|same_axis_state|direct_constraint|prior_state_update|adjacent",
   "axis_match":"exact|broader|narrower|different",
   "qualifier_fidelity":"preserved|dropped|added|conflict",
   "reason":"<=12 words"}
]}

For each proposition, return every candidate that is plausibly on the same
subject/domain, up to FOUR candidates. Omit clearly unrelated candidates.

RELATION:
- same_fact: same state, preference, goal, decision, interest, or situation.
- same_axis_state: stored state directly explains the current gap/desire on the
  same exact attribute.
- direct_constraint: stored circumstance directly corresponds to an obstacle
  explicitly present in the proposition.
- prior_state_update: current proposition revises/questions a prior state on the
  same explicitly named attribute.
- adjacent: related topic but not one of the four relations above.

AXIS MATCH:
- exact: same distinguishing attribute/activity/constraint.
- broader: candidate loses a distinguishing attribute.
- narrower: candidate adds a new distinguishing attribute.
- different: candidate concerns another activity, value, or attribute.

QUALIFIER FIDELITY:
- preserved: all important object, target, domain, timing/status, and relation
  qualifiers are retained.
- dropped: candidate omits an important qualifier.
- added: candidate adds an unsupported qualifier.
- conflict: candidate contradicts a qualifier.

Examples of distinctions:
- direct patient interaction is a different axis from patient education.
- schedule room for volunteering is not the same axis as merely wanting regular
  volunteering.
- remote-work preference is a different axis from general schedule flexibility.
- direct patient experience is narrower than general clinical experience.

Candidate retrieval order and score are not semantic evidence. Label candidates
independently and conservatively. If uncertain, use adjacent/broader/different
rather than upgrading a candidate to an exact match.
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

        evaluations = payload.get("evaluations")
        if not isinstance(evaluations, list):
            return None
        if len(candidate_topics) != expected:
            return None
        if len(evaluations) > proposition_count * min(expected, 4):
            return None

        valid_relations = {
            "same_fact",
            "same_axis_state",
            "direct_constraint",
            "prior_state_update",
            "adjacent",
        }
        valid_axis = {"exact", "broader", "narrower", "different"}
        valid_fidelity = {"preserved", "dropped", "added", "conflict"}

        seen_pairs: set[tuple[int, int]] = set()
        per_proposition_counts: dict[int, int] = {}
        parsed_evaluations: list[dict] = []
        for row in evaluations:
            if not isinstance(row, dict):
                return None
            proposition_index = row.get("proposition_index")
            candidate_index = row.get("candidate_index")
            topic_key = row.get("candidate_topic_key")
            relation = row.get("relation")
            axis_match = row.get("axis_match")
            qualifier_fidelity = row.get("qualifier_fidelity")
            reason = row.get("reason")

            if (
                not isinstance(proposition_index, int)
                or proposition_index < 0
                or proposition_index >= proposition_count
                or not isinstance(candidate_index, int)
                or candidate_index < 0
                or candidate_index >= expected
                or not isinstance(topic_key, str)
                or topic_key != candidate_topics[candidate_index]
                or relation not in valid_relations
                or axis_match not in valid_axis
                or qualifier_fidelity not in valid_fidelity
                or not isinstance(reason, str)
            ):
                return None

            pair = (proposition_index, candidate_index)
            if pair in seen_pairs:
                return None
            count = per_proposition_counts.get(proposition_index, 0) + 1
            if count > min(expected, 4):
                return None
            per_proposition_counts[proposition_index] = count

            seen_pairs.add(pair)
            parsed_evaluations.append(
                {
                    "proposition_index": proposition_index,
                    "candidate_index": candidate_index,
                    "relation": relation,
                    "axis_match": axis_match,
                    "qualifier_fidelity": qualifier_fidelity,
                    "reason": reason.strip()[:240],
                }
            )

        allowed_relations = {
            "same_fact",
            "same_axis_state",
            "direct_constraint",
            "prior_state_update",
        }
        winners: dict[int, dict] = {}
        for proposition_index in range(proposition_count):
            eligible = [
                row
                for row in parsed_evaluations
                if row["proposition_index"] == proposition_index
                and row["relation"] in allowed_relations
                and row["axis_match"] == "exact"
                and row["qualifier_fidelity"] == "preserved"
            ]
            if not eligible:
                continue
            # Candidate order is only a deterministic tie-breaker after every
            # semantic hard check above has passed.
            winner = min(eligible, key=lambda row: row["candidate_index"])
            winners[proposition_index] = winner

        selected_by_candidate: dict[int, dict] = {}
        for proposition_index, winner in winners.items():
            candidate_index = int(winner["candidate_index"])
            existing = selected_by_candidate.get(candidate_index)
            if existing is None:
                selected_by_candidate[candidate_index] = {
                    "index": candidate_index,
                    "relevant": True,
                    "proposition_index": proposition_index,
                    "proposition_indices": [proposition_index],
                    "reason": (
                        f"{winner['relation']}; axis={winner['axis_match']}; "
                        f"qualifiers={winner['qualifier_fidelity']}: "
                        f"{winner['reason']}"
                    )[:240],
                }
            else:
                existing["proposition_indices"].append(proposition_index)

        return [
            selected_by_candidate.get(
                index,
                {
                    "index": index,
                    "relevant": False,
                    "proposition_index": None,
                    "proposition_indices": [],
                    "reason": "not eligible under structured continuity rules",
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
                num_predict=700,
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
