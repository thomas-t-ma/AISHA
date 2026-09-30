"""Two-stage semantic relevance gate for memory recall."""
from __future__ import annotations

import json
from typing import Callable

import httpx

RELEVANCE_GATE_PROMPT_VERSION = "personal-continuity-v10-mode-aware"

MESSAGE_ANALYSIS_SYSTEM = """You analyze ONLY the user's current message before any
long-term memories are visible.

Return ONLY JSON:
{"message_scope":"personal_anchored|general_informational|ambiguous_unanchored",
 "scope_reason":"<=12 words",
 "propositions":[
   {"index":0,
    "text":"faithful explicit personal proposition",
    "thread_core":"enduring personal thread expressed by this proposition",
    "continuity_mode":"direct|gap|constraint|update",
    "required_anchors":["identity-defining concept"],
    "turn_modifiers":["current-turn detail not required in an older memory"]}
 ]}

Classify the message from its own words.

personal_anchored:
- The message itself expresses one or more specific, MEMORY-ADDRESSABLE personal
  situations, preferences, interests, values, decisions, goals, relationships,
  problems, updates, or ongoing threads.
- The object/domain/thread must be identifiable from the current message itself.
- MEMORY-ADDRESSABLE means an unfamiliar reader could describe what kind of
  personal memory would match WITHOUT seeing any candidate memories.
- If the message explicitly names the subject or domain, continuity language
  such as "still", "used to", "anymore", or "less than before" remains anchored
  to that named thread even when the previous state is implicit.
- A generic personal desire or state is NOT enough when its object/domain is
  missing or generic.

For personal_anchored, extract ONE proposition per independently storable
personal fact. For each proposition assign continuity_mode:
- direct: a stable preference, goal, project, interest, state, or a current
  instance of one without an expressed gap, obstacle, or revision.
- gap: the user misses, lacks, wants more/less of, or expresses a deficit on the
  thread. An older state on that exact thread may explain the current gap.
- constraint: the proposition explicitly states an obstacle or circumstance
  preventing/delaying the thread.
- update: the proposition questions, strengthens, weakens, reverses, or revises
  a prior preference/decision/state.

text:
- Preserve the meaning of the user's explicit statement faithfully.
- Preserve coordinated descriptors and negation; do not silently drop them.

thread_core:
- State the enduring personal thread that an older memory would need to identify.
- Remove only episode-specific wording that does not define which memory thread
  this is.
- Do NOT add facts, causes, motives, or goals that were not stated.

required_anchors:
- Include every concept needed to distinguish the intended personal thread from
  nearby but different memories.
- Anchors may be semantic concepts rather than exact words.
- Coordinated descriptors that change which preference/thread is identified
  belong here.
- Examples: "spicy/hot" AND "Thai food" distinguish a spicy-Thai preference;
  "direct patient interaction" differs from generic clinical experience or
  patient education; "remote work" differs from general schedule flexibility.

turn_modifiers:
- Put explicit details here only when an older memory need not contain them to
  still identify the same personal thread.
- Typical examples: tonight/today, "keep moving toward", intensity/emphasis,
  incidental stylistic adjectives, or the fact that the thread is being
  mentioned in this particular episode.
- A turn modifier may refine the present situation, but its absence from an
  older memory must not by itself make that memory irrelevant.

Split propositions when clauses express independently storable threads.
- Apply this especially to coordination with words such as "and", "while",
  "but", "plus", or "alongside": if each side independently names a personal
  thread that could be stored as its own memory, emit separate propositions.
- Never combine required_anchors from two independent domains merely because
  they appear in one sentence.
- Example: wanting more direct patient experience WHILE preparing for medical
  school is TWO propositions: the patient-experience thread and the
  medical-school-preparation/application thread.
Keep multiple attributes together only when they jointly define ONE thread.
Use at most four propositions.

general_informational:
- The message asks for general facts, explanations, definitions,
  recommendations, statistics, or category-level information without grounding
  the request in the user's own specific situation.
- Return propositions=[].

ambiguous_unanchored:
- The message sounds personal or continuous but does not independently identify
  a memory-addressable object, topic, goal, preference, situation, or referent.
- Vague references such as "that", "it", "something", "more time", "a change",
  "more outside work", "again", or "this time" do not identify a thread alone.
- Do not guess missing context from possible memories.
- Return propositions=[].

If uncertain whether a specific personal thread is independently identifiable,
prefer ambiguous_unanchored.
"""

CANDIDATE_SELECTION_SYSTEM = """You evaluate candidate memories against frozen
personal propositions extracted earlier. Phase 1 has already decided that the
message is personal and memory-addressable.

Do NOT reinterpret the message or create new propositions. Use thread_core and
required_anchors as authoritative. turn_modifiers are informative current-turn
details but are NOT required to appear in an older memory.

Do NOT choose a winner. Return structured evaluations for every plausibly
related candidate, up to SIX candidates per proposition.

Return ONLY JSON:
{"evaluations":[
  {"proposition_index":0,"candidate_index":3,
   "candidate_topic_key":"example_topic",
   "relation":"same_thread|background_state|direct_constraint|prior_state_update|adjacent",
   "thread_match":"exact|broader|narrower|different",
   "anchor_coverage":"full|partial|conflict",
   "predicate_compatibility":"exact|compatible|different|conflict",
   "reason":"<=12 words"}
]}

THREAD MATCH asks whether this is the SAME enduring personal thread:
- exact: same distinguishing preference/goal/project/constraint/state thread.
- broader: candidate collapses the thread into a more general category.
- narrower: candidate changes it to a more specific sub-thread/activity.
- different: another thread despite topical overlap.

ANCHOR COVERAGE compares ONLY required_anchors:
- full: every required anchor is represented semantically.
- partial: at least one required anchor is missing.
- conflict: a required anchor is contradicted.
Do NOT penalize a candidate for omitting turn_modifiers.

PREDICATE COMPATIBILITY:
- exact: same enduring assertion.
- compatible: wording/state differs in a way that directly supports continuity.
  Examples include "crowds out" vs "makes difficult", a stable preference
  explaining a current instance, or a stored low level explaining desire for
  more on the same axis.
- different: different assertion about the same topic/thread.
- conflict: incompatible assertion.

RELATION:
- same_thread: direct restatement or stable preference/goal/project corresponding
  to the current proposition.
- background_state: stored state directly explains the current expressed gap or
  desire on the exact same thread.
- direct_constraint: stored constraint directly corresponds to the obstacle in
  the proposition.
- prior_state_update: current proposition revises/questions a prior state on the
  exact same thread.
- adjacent: topically related but not direct continuity.

Important distinctions:
- Intending to apply to medical school is not the same thread as uncertainty
  about the exact application date.
- Direct patient interaction is not generic clinical experience or patient
  education.
- A volunteering schedule constraint is not merely a desire for consistency.
- Remote work is not generic schedule flexibility.
- A stable preference may match a specific current instance even if an episode
  detail such as "tonight" is absent from the memory.
- Project/style adjectives need not match unless Phase 1 made them a
  required_anchor.

Candidate retrieval order and scores are not semantic evidence. Be conservative:
when a required anchor is missing, mark partial rather than forcing a match.
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
            thread_core = proposition.get("thread_core")
            continuity_mode = proposition.get("continuity_mode")
            required_anchors = proposition.get("required_anchors")
            turn_modifiers = proposition.get("turn_modifiers")
            if (
                not isinstance(index, int)
                or index < 0
                or index in seen
                or not isinstance(text, str)
                or not text.strip()
                or not isinstance(thread_core, str)
                or not thread_core.strip()
                or continuity_mode not in {"direct", "gap", "constraint", "update"}
                or not isinstance(required_anchors, list)
                or not required_anchors
                or len(required_anchors) > 8
                or not all(
                    isinstance(item, str) and item.strip()
                    for item in required_anchors
                )
                or not isinstance(turn_modifiers, list)
                or len(turn_modifiers) > 8
                or not all(
                    isinstance(item, str) and item.strip()
                    for item in turn_modifiers
                )
            ):
                return None

            seen.add(index)
            parsed.append(
                {
                    "index": index,
                    "text": text.strip()[:320],
                    "thread_core": thread_core.strip()[:320],
                    "continuity_mode": continuity_mode,
                    "required_anchors": [
                        item.strip()[:120] for item in required_anchors
                    ],
                    "turn_modifiers": [
                        item.strip()[:120] for item in turn_modifiers
                    ],
                }
            )

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
        proposition_modes: list[str],
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
        if len(proposition_modes) != proposition_count:
            return None
        if any(
            mode not in {"direct", "gap", "constraint", "update"}
            for mode in proposition_modes
        ):
            return None
        if len(evaluations) > proposition_count * min(expected, 6):
            return None

        valid_relations = {
            "same_thread",
            "background_state",
            "direct_constraint",
            "prior_state_update",
            "adjacent",
        }
        valid_thread = {"exact", "broader", "narrower", "different"}
        valid_coverage = {"full", "partial", "conflict"}
        valid_predicate = {"exact", "compatible", "different", "conflict"}

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
            thread_match = row.get("thread_match")
            anchor_coverage = row.get("anchor_coverage")
            predicate_compatibility = row.get("predicate_compatibility")
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
                or thread_match not in valid_thread
                or anchor_coverage not in valid_coverage
                or predicate_compatibility not in valid_predicate
                or not isinstance(reason, str)
            ):
                return None

            pair = (proposition_index, candidate_index)
            if pair in seen_pairs:
                return None
            count = per_proposition_counts.get(proposition_index, 0) + 1
            if count > min(expected, 6):
                return None
            per_proposition_counts[proposition_index] = count
            seen_pairs.add(pair)
            parsed_evaluations.append(
                {
                    "proposition_index": proposition_index,
                    "candidate_index": candidate_index,
                    "relation": relation,
                    "thread_match": thread_match,
                    "anchor_coverage": anchor_coverage,
                    "predicate_compatibility": predicate_compatibility,
                    "reason": reason.strip()[:240],
                }
            )

        allowed_relations = {
            "same_thread",
            "background_state",
            "direct_constraint",
            "prior_state_update",
        }
        mode_relation_priority = {
            "direct": {
                "same_thread": 5,
                "prior_state_update": 3,
                "direct_constraint": 2,
                "background_state": 1,
            },
            "gap": {
                "background_state": 5,
                "direct_constraint": 4,
                "same_thread": 3,
                "prior_state_update": 1,
            },
            "constraint": {
                "direct_constraint": 5,
                "same_thread": 4,
                "background_state": 3,
                "prior_state_update": 1,
            },
            "update": {
                "prior_state_update": 5,
                "same_thread": 4,
                "background_state": 2,
                "direct_constraint": 1,
            },
        }
        predicate_priority = {"exact": 2, "compatible": 1}
        thread_priority = {"exact": 2, "broader": 1}

        winners: dict[int, dict] = {}
        for proposition_index in range(proposition_count):
            mode = proposition_modes[proposition_index]
            eligible: list[dict] = []
            for row in parsed_evaluations:
                if row["proposition_index"] != proposition_index:
                    continue
                if row["relation"] not in allowed_relations:
                    continue
                if row["anchor_coverage"] != "full":
                    continue
                if row["predicate_compatibility"] not in {"exact", "compatible"}:
                    continue

                exact_thread = row["thread_match"] == "exact"
                stable_broader_thread = (
                    mode == "direct"
                    and row["relation"] == "same_thread"
                    and row["thread_match"] == "broader"
                )
                if not (exact_thread or stable_broader_thread):
                    continue
                eligible.append(row)

            if not eligible:
                continue

            relation_priority = mode_relation_priority[mode]
            winner = max(
                eligible,
                key=lambda row: (
                    relation_priority[row["relation"]],
                    thread_priority[row["thread_match"]],
                    predicate_priority[row["predicate_compatibility"]],
                    -int(row["candidate_index"]),
                ),
            )
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
                        f"{winner['relation']}; thread={winner['thread_match']}; "
                        f"anchors={winner['anchor_coverage']}; "
                        f"predicate={winner['predicate_compatibility']}: "
                        f"{winner['reason']}"
                    )[:240],
                }
            else:
                existing["proposition_indices"].append(proposition_index)

        evaluated_by_candidate: dict[int, list[dict]] = {}
        for row in parsed_evaluations:
            evaluated_by_candidate.setdefault(int(row["candidate_index"]), []).append(row)

        decisions: list[dict] = []
        for index in range(expected):
            selected = selected_by_candidate.get(index)
            if selected is not None:
                decisions.append(selected)
                continue

            rows = evaluated_by_candidate.get(index, [])
            if rows:
                summary = rows[0]
                reason = (
                    f"rejected: {summary['relation']}; "
                    f"thread={summary['thread_match']}; "
                    f"anchors={summary['anchor_coverage']}; "
                    f"predicate={summary['predicate_compatibility']}: "
                    f"{summary['reason']}"
                )[:240]
            else:
                reason = "not evaluated as plausibly related"

            decisions.append(
                {
                    "index": index,
                    "relevant": False,
                    "proposition_index": None,
                    "proposition_indices": [],
                    "reason": reason,
                }
            )

        return decisions

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
                num_predict=320,
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
                num_predict=900,
                parser=lambda raw: self._parse_selection(
                    raw,
                    expected=len(candidates),
                    proposition_count=len(self.last_propositions),
                    proposition_modes=[
                        str(proposition.get("continuity_mode", ""))
                        for proposition in self.last_propositions
                    ],
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
