from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Protocol


@dataclass(frozen=True)
class RerankerCase:
    case_id: str
    text: str
    candidate_topics: tuple[str, ...]
    expected_topics: tuple[str, ...]
    category: str


class MemoryRelevanceGate(Protocol):
    async def judge(self, user_text: str, candidates: list[dict]) -> list[dict]: ...

    def status(self) -> dict: ...


SMOKE_RERANKER_CASE_IDS = {
    "patient_contact_paraphrase",
    "work_and_volunteer_both",
    "medical_school_general_question",
}

WORK_CANDIDATES = (
    "job_patient_interaction_level",
    "volunteering_schedule_constraint",
    "remote_work_preference",
    "professional_school_goal",
    "research_interest",
    "pottery_hobby",
)
SCHOOL_RESEARCH_CANDIDATES = (
    "professional_school_goal",
    "research_interest",
    "job_patient_interaction_level",
    "volunteering_schedule_constraint",
    "horror_game_project",
    "remote_work_preference",
)
COMPUTER_CANDIDATES = (
    "keyboard_preference",
    "computer_build_priority",
    "horror_game_project",
    "research_interest",
    "pottery_hobby",
    "sleep_schedule",
)
SPAIN_FOOD_CANDIDATES = (
    "country_relocation_deliberation",
    "food_preference_spicy",
    "remote_work_preference",
    "volunteering_schedule_constraint",
    "pottery_hobby",
    "sleep_schedule",
)
HOBBY_PROJECT_CANDIDATES = (
    "pottery_hobby",
    "horror_game_project",
    "research_interest",
    "computer_build_priority",
    "keyboard_preference",
    "food_preference_spicy",
)
SLEEP_CANDIDATES = (
    "sleep_schedule",
    "job_patient_interaction_level",
    "professional_school_goal",
    "research_interest",
    "food_preference_spicy",
    "remote_work_preference",
)


def _rotate(topics: tuple[str, ...], amount: int) -> tuple[str, ...]:
    amount %= len(topics)
    return topics[amount:] + topics[:amount]


def reranker_benchmark_cases() -> list[RerankerCase]:
    """Forty fixed-candidate cases designed to expose over-retrieval."""
    return [
        RerankerCase(
            "patient_contact_paraphrase",
            "I keep wishing I spent more of my day face-to-face with the people we're supposed to help.",
            _rotate(WORK_CANDIDATES, 1),
            ("job_patient_interaction_level",),
            "minimal_pair",
        ),
        RerankerCase(
            "patient_contact_direct",
            "I want more patient interaction in my current work.",
            _rotate(WORK_CANDIDATES, 3),
            ("job_patient_interaction_level",),
            "minimal_pair",
        ),
        RerankerCase(
            "volunteer_constraint_paraphrase",
            "I'd help in the community more often if my weekday job didn't swallow nearly all of my time.",
            _rotate(WORK_CANDIDATES, 2),
            ("volunteering_schedule_constraint",),
            "minimal_pair",
        ),
        RerankerCase(
            "work_and_volunteer_both",
            (
                "Part of why I'm thinking about changing my work is that I want more time "
                "with the people we're helping and more room to volunteer."
            ),
            _rotate(WORK_CANDIDATES, 4),
            (
                "job_patient_interaction_level",
                "volunteering_schedule_constraint",
            ),
            "multi_memory",
        ),
        RerankerCase(
            "patient_contact_general_question",
            "Why do some healthcare jobs involve so little direct patient contact?",
            _rotate(WORK_CANDIDATES, 5),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "volunteering_general_question",
            "What commonly makes volunteering difficult for people who work full time?",
            _rotate(WORK_CANDIDATES, 0),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "vague_helping_people",
            "I want work that feels more meaningful and lets me help people.",
            _rotate(WORK_CANDIDATES, 2),
            (),
            "ambiguous_negative",
        ),
        RerankerCase(
            "vague_time_after_work",
            "I never seem to have enough time after work.",
            _rotate(WORK_CANDIDATES, 4),
            (),
            "ambiguous_negative",
        ),
        RerankerCase(
            "patient_contact_with_school_goal",
            (
                "I want more direct patient contact in my work because I still plan to apply "
                "to medical school."
            ),
            _rotate(WORK_CANDIDATES, 1),
            (
                "job_patient_interaction_level",
                "professional_school_goal",
            ),
            "multi_memory",
        ),
        RerankerCase(
            "school_goal_direct",
            "I need to get serious about my medical school application plan.",
            _rotate(WORK_CANDIDATES, 3),
            ("professional_school_goal",),
            "personal_continuity",
        ),
        RerankerCase(
            "remote_work_paraphrase",
            "I really don't want my next job to require being in the office five days a week.",
            _rotate(WORK_CANDIDATES, 5),
            ("remote_work_preference",),
            "minimal_pair",
        ),
        RerankerCase(
            "volunteer_schedule_direct",
            "My weekday work schedule leaves almost no room for regular volunteering.",
            _rotate(WORK_CANDIDATES, 1),
            ("volunteering_schedule_constraint",),
            "minimal_pair",
        ),
        RerankerCase(
            "remote_work_general_question",
            "Which kinds of jobs are most commonly remote or hybrid?",
            _rotate(WORK_CANDIDATES, 3),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "office_schedule_general_question",
            "Are five-day office schedules becoming less common?",
            _rotate(WORK_CANDIDATES, 0),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "research_interest_paraphrase",
            "I keep gravitating toward projects where machine learning solves a practical clinical problem.",
            _rotate(SCHOOL_RESEARCH_CANDIDATES, 3),
            ("research_interest",),
            "minimal_pair",
        ),
        RerankerCase(
            "school_goal_paraphrase",
            "I still picture myself becoming a physician, so I don't want to let the application plan drift.",
            _rotate(SCHOOL_RESEARCH_CANDIDATES, 1),
            ("professional_school_goal",),
            "minimal_pair",
        ),
        RerankerCase(
            "clinical_ai_general_question",
            "How is artificial intelligence being used in clinical research?",
            _rotate(SCHOOL_RESEARCH_CANDIDATES, 4),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "medical_school_general_question",
            "How does medical school accreditation work in the United States?",
            _rotate(SCHOOL_RESEARCH_CANDIDATES, 2),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "research_and_school_both",
            "I want my healthcare AI work to support my longer-term plan to apply to medical school.",
            _rotate(SCHOOL_RESEARCH_CANDIDATES, 5),
            (
                "research_interest",
                "professional_school_goal",
            ),
            "multi_memory",
        ),
        RerankerCase(
            "keyboard_direct",
            "I still want a quiet linear keyboard.",
            _rotate(COMPUTER_CANDIDATES, 2),
            ("keyboard_preference",),
            "minimal_pair",
        ),
        RerankerCase(
            "computer_quality_paraphrase",
            "For my next PC I'd rather pay for parts I can trust than save money on mystery components.",
            _rotate(COMPUTER_CANDIDATES, 4),
            ("computer_build_priority",),
            "minimal_pair",
        ),
        RerankerCase(
            "keyboard_and_computer_quality",
            (
                "For the new computer I still want a quiet linear keyboard, and I don't want "
                "to cheap out on unknown components."
            ),
            _rotate(COMPUTER_CANDIDATES, 1),
            (
                "keyboard_preference",
                "computer_build_priority",
            ),
            "multi_memory",
        ),
        RerankerCase(
            "keyboard_general_question",
            "What switch types are usually best for an office keyboard?",
            _rotate(COMPUTER_CANDIDATES, 5),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "cpu_cache_general_question",
            "Why do CPUs use multiple levels of cache?",
            _rotate(COMPUTER_CANDIDATES, 3),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "vague_quieter",
            "I want something quieter this time.",
            _rotate(COMPUTER_CANDIDATES, 0),
            (),
            "ambiguous_negative",
        ),
        RerankerCase(
            "vague_quality",
            "I don't want to cheap out again.",
            _rotate(COMPUTER_CANDIDATES, 2),
            (),
            "ambiguous_negative",
        ),
        RerankerCase(
            "relocation_paraphrase",
            "I'm still torn about whether I should actually make the move to Spain.",
            _rotate(SPAIN_FOOD_CANDIDATES, 3),
            ("country_relocation_deliberation",),
            "personal_continuity",
        ),
        RerankerCase(
            "relocation_contradiction",
            "I'm starting to think I might not want to move to Spain after all.",
            _rotate(SPAIN_FOOD_CANDIDATES, 5),
            ("country_relocation_deliberation",),
            "same_thread_update",
        ),
        RerankerCase(
            "spanish_dessert_general_question",
            "What's a good Spanish dessert to make for a dinner party?",
            _rotate(SPAIN_FOOD_CANDIDATES, 1),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "spain_population_general_question",
            "What is Spain's population and how has it changed recently?",
            _rotate(SPAIN_FOOD_CANDIDATES, 4),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "spicy_thai_personal",
            "I'm craving spicy Thai food again.",
            _rotate(SPAIN_FOOD_CANDIDATES, 2),
            ("food_preference_spicy",),
            "personal_continuity",
        ),
        RerankerCase(
            "thai_spice_general_question",
            "What ingredients usually make Thai food spicy?",
            _rotate(SPAIN_FOOD_CANDIDATES, 0),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "spain_tourist_food_general_question",
            "What foods should a tourist make sure to try in Spain?",
            _rotate(SPAIN_FOOD_CANDIDATES, 5),
            (),
            "cross_topic_negative",
        ),
        RerankerCase(
            "pottery_paraphrase",
            "I've been missing the feeling of making something with clay at the studio.",
            _rotate(HOBBY_PROJECT_CANDIDATES, 3),
            ("pottery_hobby",),
            "personal_continuity",
        ),
        RerankerCase(
            "pottery_general_question",
            "What type of clay is easiest for a beginner to use?",
            _rotate(HOBBY_PROJECT_CANDIDATES, 1),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "game_project_paraphrase",
            "I want to get back to designing that eerie neighborhood game idea.",
            _rotate(HOBBY_PROJECT_CANDIDATES, 5),
            ("horror_game_project",),
            "personal_continuity",
        ),
        RerankerCase(
            "suburban_horror_general_question",
            "What usually makes suburban horror feel unsettling?",
            _rotate(HOBBY_PROJECT_CANDIDATES, 2),
            (),
            "general_question_negative",
        ),
        RerankerCase(
            "ambiguous_project_reference",
            "I want to get back to that project.",
            _rotate(HOBBY_PROJECT_CANDIDATES, 4),
            (),
            "ambiguous_negative",
        ),
        RerankerCase(
            "sleep_schedule_personal",
            "I keep ending up with about seven hours because I go to bed around midnight.",
            _rotate(SLEEP_CANDIDATES, 3),
            ("sleep_schedule",),
            "personal_continuity",
        ),
        RerankerCase(
            "insomnia_general_question",
            "What are the most common causes of insomnia?",
            _rotate(SLEEP_CANDIDATES, 1),
            (),
            "general_question_negative",
        ),
    ]


async def run_reranker_case(
    case: RerankerCase,
    beliefs: list[dict],
    gate: MemoryRelevanceGate,
) -> dict:
    beliefs_by_topic = {str(row["topic_key"]): row for row in beliefs}
    missing = [topic for topic in case.candidate_topics if topic not in beliefs_by_topic]
    if missing:
        raise ValueError(f"Unknown benchmark candidate topics: {missing}")

    candidates = [
        {
            "belief": beliefs_by_topic[topic],
            "candidate_sources": ["benchmark"],
            "lexical_score": None,
            "semantic_score": None,
        }
        for topic in case.candidate_topics
    ]

    started = perf_counter()
    decisions = await gate.judge(case.text, candidates)
    wall_ms = (perf_counter() - started) * 1000

    if len(decisions) != len(candidates):
        raise ValueError("Relevance gate returned an unexpected number of decisions")

    candidate_decisions: list[dict] = []
    actual_topics: list[str] = []
    for topic, decision in zip(case.candidate_topics, decisions, strict=True):
        accepted = bool(decision.get("relevant", False))
        candidate_decisions.append(
            {
                "topic_key": topic,
                "selected": accepted,
                "decision": "reranker_accept" if accepted else "reranker_reject",
                "reason": str(decision.get("reason", "")),
            }
        )
        if accepted:
            actual_topics.append(topic)

    expected = set(case.expected_topics)
    actual = set(actual_topics)
    status = gate.status()
    metrics = status.get("last_metrics", {}) if isinstance(status, dict) else {}

    return {
        "case_id": case.case_id,
        "category": case.category,
        "text": case.text,
        "candidate_topics": list(case.candidate_topics),
        "expected_topics": list(case.expected_topics),
        "actual_topics": actual_topics,
        "exact": actual == expected,
        "true_positive": len(expected & actual),
        "false_positive": len(actual - expected),
        "false_negative": len(expected - actual),
        "candidate_decisions": candidate_decisions,
        "relevance_gate": status,
        "latency_ms": {
            "total": round(wall_ms, 3),
            "gate": round(float(metrics.get("total_ms", 0) or 0), 3),
            "load": round(float(metrics.get("load_ms", 0) or 0), 3),
            "prompt_eval": round(float(metrics.get("prompt_eval_ms", 0) or 0), 3),
            "eval": round(float(metrics.get("eval_ms", 0) or 0), 3),
        },
    }


def _pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize_reranker_results(results: list[dict]) -> dict:
    tp = sum(int(row["true_positive"]) for row in results)
    fp = sum(int(row["false_positive"]) for row in results)
    fn = sum(int(row["false_negative"]) for row in results)
    exact = sum(1 for row in results if row["exact"])
    expected_memories = sum(len(row["expected_topics"]) for row in results)

    negatives = [row for row in results if not row["expected_topics"]]
    negative_passes = sum(1 for row in negatives if not row["actual_topics"])
    clean_queries = sum(1 for row in results if int(row["false_positive"]) == 0)
    false_positive_cases = len(results) - clean_queries

    total_latencies = [float(row["latency_ms"]["total"]) for row in results]
    gate_latencies = [
        float(row["latency_ms"].get("gate", 0))
        for row in results
        if float(row["latency_ms"].get("gate", 0)) > 0
    ]

    category_rows: dict[str, list[dict]] = {}
    for row in results:
        category_rows.setdefault(str(row["category"]), []).append(row)

    category_summary = {}
    for category, rows in sorted(category_rows.items()):
        category_tp = sum(int(row["true_positive"]) for row in rows)
        category_fp = sum(int(row["false_positive"]) for row in rows)
        category_fn = sum(int(row["false_negative"]) for row in rows)
        category_exact = sum(1 for row in rows if row["exact"])
        category_summary[category] = {
            "cases": len(rows),
            "exact_cases": category_exact,
            "exact_case_accuracy": _pct(category_exact, len(rows)),
            "precision": _pct(category_tp, category_tp + category_fp),
            "recall": _pct(category_tp, category_tp + category_fn),
            "false_positive": category_fp,
            "false_negative": category_fn,
        }

    return {
        "cases": len(results),
        "exact_cases": exact,
        "exact_case_accuracy": _pct(exact, len(results)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": _pct(tp, tp + fp),
        "recall": _pct(tp, tp + fn),
        "negative_controls": len(negatives),
        "negative_controls_passed": negative_passes,
        "negative_control_accuracy": _pct(negative_passes, len(negatives)),
        "expected_memories": expected_memories,
        "over_retrieval_rate": _pct(fp, expected_memories),
        "clean_queries": clean_queries,
        "false_positive_cases": false_positive_cases,
        "clean_query_rate": _pct(clean_queries, len(results)),
        "category_summary": category_summary,
        "latency_ms": {
            "median_total": round(median(total_latencies), 3) if total_latencies else 0.0,
            "median_gate": round(median(gate_latencies), 3) if gate_latencies else 0.0,
            "max_gate": round(max(gate_latencies), 3) if gate_latencies else 0.0,
            "max_total": round(max(total_latencies), 3) if total_latencies else 0.0,
        },
    }
