from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter

from aisha.memory.retrieval import explain_relevant_beliefs
from aisha.memory.semantic import OllamaSemanticMemoryRetriever


@dataclass(frozen=True)
class RecallCase:
    case_id: str
    text: str
    expected_topics: tuple[str, ...]
    category: str


def synthetic_beliefs() -> list[dict]:
    """Confusable, non-user-specific memories for retrieval evaluation."""
    rows = [
        (
            "job_patient_interaction_level",
            "The user's current healthcare role involves very little direct patient interaction.",
            None,
        ),
        (
            "professional_school_goal",
            "The user plans to apply to medical school in the future.",
            None,
        ),
        (
            "volunteering_schedule_constraint",
            "The user's full-time weekday schedule makes regular community volunteering difficult.",
            None,
        ),
        (
            "country_relocation_deliberation",
            "The user is considering moving to Spain but has not made a final decision.",
            "Whether the user will ultimately move to Spain remains unresolved.",
        ),
        (
            "remote_work_preference",
            "The user prefers jobs that allow at least some work from home.",
            None,
        ),
        (
            "pottery_hobby",
            "The user enjoys making pottery and wants to spend more time at the ceramics studio.",
            None,
        ),
        (
            "food_preference_spicy",
            "The user especially likes spicy Thai food.",
            None,
        ),
        (
            "sleep_schedule",
            "The user usually sleeps from around midnight until about seven in the morning.",
            None,
        ),
        (
            "keyboard_preference",
            "The user prefers quiet linear mechanical keyboards.",
            None,
        ),
        (
            "computer_build_priority",
            "The user values durable, well-documented computer components over the cheapest option.",
            None,
        ),
        (
            "horror_game_project",
            "The user wants to build a surreal suburban horror game as a personal project.",
            None,
        ),
        (
            "research_interest",
            "The user enjoys applied artificial-intelligence projects related to healthcare.",
            None,
        ),
    ]
    return [
        {
            "belief_id": f"benchmark_{index}",
            "topic_key": topic,
            "text": text,
            "open_question": question,
            "evidence_status": "verified",
            "revision": 1,
            "updated_at": f"2026-01-{index + 1:02d}T12:00:00",
        }
        for index, (topic, text, question) in enumerate(rows)
    ]


def benchmark_cases() -> list[RecallCase]:
    return [
        RecallCase(
            "patient_contact_paraphrase",
            "I keep wishing I spent more of my day face-to-face with the people we're supposed to help.",
            ("job_patient_interaction_level",),
            "semantic_positive",
        ),
        RecallCase(
            "patient_contact_direct",
            "I want more patient interaction in my current work.",
            ("job_patient_interaction_level",),
            "lexical_positive",
        ),
        RecallCase(
            "school_goal_paraphrase",
            "I still picture myself becoming a physician, so I don't want to let the application plan drift.",
            ("professional_school_goal",),
            "semantic_positive",
        ),
        RecallCase(
            "volunteer_constraint_paraphrase",
            "I'd help in the community more often if my weekday job didn't swallow nearly all of my time.",
            ("volunteering_schedule_constraint",),
            "semantic_positive",
        ),
        RecallCase(
            "relocation_paraphrase",
            "I'm still torn about whether I should actually make the move to Spain.",
            ("country_relocation_deliberation",),
            "semantic_positive",
        ),
        RecallCase(
            "relocation_contradiction",
            "I'm starting to think I might not want to move to Spain after all.",
            ("country_relocation_deliberation",),
            "same_thread_update",
        ),
        RecallCase(
            "remote_work_paraphrase",
            "I really don't want my next job to require being in the office five days a week.",
            ("remote_work_preference",),
            "semantic_positive",
        ),
        RecallCase(
            "pottery_paraphrase",
            "I've been missing the feeling of making something with clay at the studio.",
            ("pottery_hobby",),
            "semantic_positive",
        ),
        RecallCase(
            "keyboard_direct",
            "I still want a quiet linear keyboard.",
            ("keyboard_preference",),
            "lexical_positive",
        ),
        RecallCase(
            "computer_quality_paraphrase",
            "For my next PC I'd rather pay for parts I can trust than save money on mystery components.",
            ("computer_build_priority",),
            "semantic_positive",
        ),
        RecallCase(
            "game_project_paraphrase",
            "I want to get back to designing that eerie neighborhood game idea.",
            ("horror_game_project",),
            "semantic_positive",
        ),
        RecallCase(
            "research_paraphrase",
            "I keep gravitating toward projects where machine learning solves a practical clinical problem.",
            ("research_interest",),
            "semantic_positive",
        ),
        RecallCase(
            "multi_work_and_volunteer",
            "Part of why I'm thinking about changing my work is that I want more time with the people we're helping and more room to volunteer.",
            (
                "job_patient_interaction_level",
                "volunteering_schedule_constraint",
            ),
            "multi_memory",
        ),
        RecallCase(
            "negative_spanish_food",
            "What's a good Spanish dessert to make for a dinner party?",
            (),
            "hard_negative",
        ),
        RecallCase(
            "negative_medical_school_general",
            "How does medical school accreditation work in the United States?",
            (),
            "hard_negative",
        ),
        RecallCase(
            "negative_ai_general",
            "Explain the difference between supervised and self-supervised learning.",
            (),
            "hard_negative",
        ),
        RecallCase(
            "negative_sleep_general",
            "What are the most common causes of insomnia?",
            (),
            "hard_negative",
        ),
        RecallCase(
            "negative_computers_general",
            "Why do CPUs use cache hierarchies?",
            (),
            "hard_negative",
        ),
        RecallCase(
            "negative_food",
            "What is a good dessert to make tonight?",
            (),
            "unrelated_negative",
        ),
        RecallCase(
            "negative_history",
            "Give me a short explanation of the Roman Republic.",
            (),
            "unrelated_negative",
        ),
    ]


async def run_case(
    case: RecallCase,
    beliefs: list[dict],
    semantic_retriever: OllamaSemanticMemoryRetriever,
    *,
    total_limit: int = 4,
) -> dict:
    started = perf_counter()
    lexical_started = perf_counter()
    lexical = explain_relevant_beliefs(case.text, beliefs, limit=total_limit)
    lexical_ms = (perf_counter() - lexical_started) * 1000

    semantic: list[dict] = []
    semantic_ms = 0.0
    if len(lexical) < total_limit:
        lexical_ids = {
            str(detail["belief"].get("belief_id", ""))
            for detail in lexical
        }
        semantic_started = perf_counter()
        semantic = await semantic_retriever.recall(
            case.text,
            beliefs,
            exclude_belief_ids=lexical_ids,
            remaining_limit=total_limit - len(lexical),
        )
        semantic_ms = (perf_counter() - semantic_started) * 1000

    details = lexical + semantic
    actual_topics = tuple(str(detail["belief"]["topic_key"]) for detail in details)
    expected = set(case.expected_topics)
    actual = set(actual_topics)
    true_positive = len(expected & actual)
    false_positive = len(actual - expected)
    false_negative = len(expected - actual)

    status = semantic_retriever.status()
    return {
        "case_id": case.case_id,
        "category": case.category,
        "text": case.text,
        "expected_topics": list(case.expected_topics),
        "actual_topics": list(actual_topics),
        "exact": actual == expected,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "methods": [
            {
                "topic_key": detail["belief"]["topic_key"],
                "method": detail["method"],
                "score": detail.get("score"),
                "reranker_reason": detail.get("reranker_reason"),
            }
            for detail in details
        ],
        "semantic_candidates": status.get("last_candidates", []),
        "relevance_gate": status.get("relevance_gate", {}),
        "latency_ms": {
            "total": round((perf_counter() - started) * 1000, 3),
            "lexical": round(lexical_ms, 3),
            "semantic": round(semantic_ms, 3),
        },
    }


def summarize_results(results: list[dict]) -> dict:
    tp = sum(row["true_positive"] for row in results)
    fp = sum(row["false_positive"] for row in results)
    fn = sum(row["false_negative"] for row in results)
    exact = sum(1 for row in results if row["exact"])
    negatives = [row for row in results if not row["expected_topics"]]
    negative_passes = sum(1 for row in negatives if not row["actual_topics"])
    semantic_latencies = [
        float(row["latency_ms"]["semantic"])
        for row in results
        if float(row["latency_ms"]["semantic"]) > 0
    ]
    total_latencies = [float(row["latency_ms"]["total"]) for row in results]

    method_counts: dict[str, int] = {}
    candidate_decisions: dict[str, int] = {}
    for row in results:
        for method in row["methods"]:
            name = str(method["method"])
            method_counts[name] = method_counts.get(name, 0) + 1
        for candidate in row["semantic_candidates"]:
            name = str(candidate.get("decision", "unknown"))
            candidate_decisions[name] = candidate_decisions.get(name, 0) + 1

    def pct(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "cases": len(results),
        "exact_cases": exact,
        "exact_case_accuracy": pct(exact, len(results)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": pct(tp, tp + fp),
        "recall": pct(tp, tp + fn),
        "negative_controls": len(negatives),
        "negative_controls_passed": negative_passes,
        "negative_control_accuracy": pct(negative_passes, len(negatives)),
        "method_counts": method_counts,
        "candidate_decisions": candidate_decisions,
        "latency_ms": {
            "median_total": round(median(total_latencies), 3) if total_latencies else 0.0,
            "median_semantic": (
                round(median(semantic_latencies), 3) if semantic_latencies else 0.0
            ),
            "max_total": round(max(total_latencies), 3) if total_latencies else 0.0,
        },
    }
