from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from time import perf_counter

from aisha.memory.benchmark import synthetic_beliefs
from aisha.memory.retrieval import explain_relevant_beliefs
from aisha.memory.semantic import OllamaSemanticMemoryRetriever


@dataclass(frozen=True)
class PipelineRecallCase:
    case_id: str
    text: str
    expected_topics: tuple[str, ...]
    category: str


NOISY_MEMORY_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "work": (
        ("work_team_collaboration_preference", "The user prefers jobs with a collaborative and supportive team."),
        ("work_commute_preference", "The user would rather avoid a long daily commute to work."),
        ("work_schedule_flexibility", "The user values having some flexibility in their daily work schedule."),
        ("work_job_stability_priority", "The user values stability and predictable employment."),
        ("work_salary_priority", "The user wants compensation to be competitive when comparing jobs."),
        ("work_learning_opportunities", "The user likes roles that provide frequent opportunities to learn new skills."),
        ("work_mentor_access", "The user values having accessible mentors at work."),
        ("work_independence_preference", "The user enjoys having ownership over projects at work."),
        ("work_team_size_preference", "The user tends to prefer smaller work teams."),
        ("work_documentation_dislike", "The user dislikes spending large amounts of time on administrative documentation."),
    ),
    "clinical": (
        ("clinical_exposure_goal", "The user wants to build more clinical experience over time."),
        ("patient_education_interest", "The user enjoys explaining healthcare information directly to patients."),
        ("clinical_research_interest", "The user is interested in participating in clinical research studies."),
        ("laboratory_work_experience", "The user has experience working in a biomedical laboratory."),
        ("pathology_interest", "The user has a particular academic interest in pathology."),
        ("clinical_trial_interest", "The user is interested in how clinical trials are designed and operated."),
        ("healthcare_team_interest", "The user likes working with physicians and other healthcare professionals."),
        ("clinical_data_interest", "The user enjoys working with healthcare datasets."),
        ("patient_recruitment_interest", "The user is interested in learning how participants are recruited for studies."),
        ("hospital_environment_familiarity", "The user is familiar with working in hospital and academic medical environments."),
    ),
    "school": (
        ("medical_school_timeline_uncertainty", "The user has not fixed an exact date for submitting a medical school application."),
        ("medical_school_cost_concern", "The user is concerned about the financial cost of medical education."),
        ("medical_school_location_preference", "The user would prefer a medical school near an established support network."),
        ("medical_school_research_value", "The user values strong research opportunities when considering medical schools."),
        ("medical_school_clinical_value", "The user values strong clinical training when considering medical schools."),
        ("application_experience_concern", "The user wants a future professional-school application to show meaningful experiences."),
        ("exam_preparation_goal", "The user expects to prepare for a standardized admissions exam."),
        ("physician_shadowing_interest", "The user wants continued opportunities to observe physicians at work."),
        ("healthcare_career_commitment", "The user expects to build a long-term career in healthcare."),
        ("science_education_interest", "The user enjoys learning advanced biomedical science."),
    ),
    "service": (
        ("community_service_interest", "The user wants community service to remain part of their life."),
        ("weekend_service_preference", "The user would prefer volunteer opportunities that can happen on weekends."),
        ("service_healthcare_preference", "The user is especially interested in service opportunities connected to healthcare."),
        ("service_consistency_goal", "The user would like volunteer work to be regular rather than occasional."),
        ("service_local_preference", "The user prefers community activities that are close to home."),
        ("service_team_preference", "The user enjoys volunteer activities done with a team."),
        ("service_direct_help_preference", "The user likes service activities where the benefit to other people is visible."),
        ("service_training_interest", "The user is willing to complete training required for a meaningful volunteer role."),
        ("service_time_concern", "The user worries about balancing service commitments with other responsibilities."),
        ("service_long_term_goal", "The user wants to maintain community involvement over the long term."),
    ),
    "research": (
        ("machine_learning_interest", "The user enjoys learning about machine-learning methods."),
        ("research_publication_goal", "The user would like future research work to lead to publishable results."),
        ("research_independence_goal", "The user wants opportunities to lead an independent research project."),
        ("research_reproducibility_value", "The user values reproducible and well-documented research workflows."),
        ("research_automation_interest", "The user enjoys automating repetitive research tasks."),
        ("research_data_quality_interest", "The user cares about data quality and careful preprocessing."),
        ("research_explainability_interest", "The user values explainable models in scientific applications."),
        ("research_open_source_interest", "The user likes research software that can be inspected and modified."),
        ("research_productization_interest", "The user enjoys turning technical research ideas into usable tools."),
        ("research_benchmarking_interest", "The user likes evaluating models with explicit benchmark suites."),
    ),
    "computer": (
        ("computer_performance_priority", "The user wants a computer with strong performance for demanding workloads."),
        ("computer_upgradeability_priority", "The user values computers that can be upgraded later."),
        ("computer_cooling_priority", "The user cares about good thermal performance and cooling."),
        ("computer_noise_priority", "The user prefers a computer that does not become excessively loud."),
        ("computer_warranty_priority", "The user values a clear warranty and responsive hardware support."),
        ("computer_storage_priority", "The user wants ample fast storage for project files."),
        ("computer_memory_priority", "The user values having plenty of system memory."),
        ("monitor_brightness_preference", "The user prefers computer monitors with good brightness."),
        ("monitor_resolution_preference", "The user prefers higher-resolution monitors for work."),
        ("mouse_quality_preference", "The user wants a reliable and comfortable computer mouse."),
    ),
    "keyboard": (
        ("keyboard_full_size_preference", "The user likes having a full-size keyboard layout with a number pad."),
        ("keyboard_build_quality", "The user values solid construction in a keyboard."),
        ("keyboard_wireless_interest", "The user is open to a wireless keyboard if latency is low."),
        ("keyboard_office_use", "The user wants keyboards that are appropriate for shared work environments."),
        ("keyboard_keycap_preference", "The user likes durable keycaps that do not become shiny quickly."),
        ("keyboard_backlight_preference", "The user finds keyboard backlighting useful in dark rooms."),
        ("keyboard_latency_priority", "The user wants low input latency from computer peripherals."),
        ("keyboard_mac_compatibility", "The user values good keyboard compatibility with macOS."),
        ("keyboard_windows_compatibility", "The user values good keyboard compatibility with Windows."),
        ("keyboard_price_sensitivity", "The user does not want to overpay for a keyboard."),
    ),
    "creative": (
        ("drawing_hobby", "The user enjoys drawing as a creative hobby."),
        ("painting_hobby", "The user enjoys painting and visual art."),
        ("animation_interest", "The user is interested in animation and digital art."),
        ("creative_studio_time", "The user likes spending uninterrupted time on creative projects."),
        ("creative_learning_goal", "The user enjoys learning new artistic techniques."),
        ("creative_tools_interest", "The user likes experimenting with tools for making visual art."),
        ("creative_project_completion", "The user wants to finish more personal creative projects."),
        ("creative_portfolio_interest", "The user likes preserving finished creative work in a portfolio."),
        ("creative_class_interest", "The user is open to taking classes to improve artistic skills."),
        ("creative_social_interest", "The user enjoys creative activities that can also be social."),
    ),
    "games": (
        ("game_development_interest", "The user wants to learn more about game development."),
        ("horror_media_interest", "The user enjoys unsettling horror media."),
        ("game_engine_learning", "The user is interested in learning a modern game engine."),
        ("game_version_control", "The user wants game-development work tracked with version control."),
        ("game_environment_design", "The user enjoys thinking about environmental storytelling in games."),
        ("game_audio_interest", "The user is interested in how sound design affects game atmosphere."),
        ("game_lighting_interest", "The user is interested in lighting and color for game atmosphere."),
        ("game_mechanics_interest", "The user enjoys designing gameplay systems and rules."),
        ("game_story_interest", "The user enjoys combining narrative with interactive mechanics."),
        ("game_release_goal", "The user would like to finish and release a game project."),
    ),
    "travel": (
        ("europe_travel_interest", "The user is interested in traveling in Europe."),
        ("spain_travel_interest", "The user would like to visit Spain as a traveler."),
        ("japan_travel_interest", "The user is interested in spending time in Japan."),
        ("city_walkability_preference", "The user prefers places where daily needs can be reached on foot."),
        ("public_transit_preference", "The user values good public transportation when choosing where to live or travel."),
        ("warm_weather_preference", "The user generally prefers milder weather."),
        ("international_living_interest", "The user is interested in what daily life is like in other countries."),
        ("language_learning_interest", "The user enjoys learning useful phrases in other languages."),
        ("travel_food_interest", "The user likes exploring local food while traveling."),
        ("travel_budget_concern", "The user pays attention to cost when planning travel."),
    ),
    "food": (
        ("thai_food_interest", "The user enjoys Thai cuisine."),
        ("spicy_food_general", "The user generally enjoys foods with noticeable heat."),
        ("japanese_food_interest", "The user enjoys Japanese cuisine."),
        ("dessert_interest", "The user enjoys trying interesting desserts."),
        ("restaurant_quality_priority", "The user values food quality more than trendy presentation."),
        ("food_value_priority", "The user likes restaurants that feel worth the price."),
        ("home_cooking_interest", "The user enjoys learning how to cook new dishes."),
        ("savory_food_preference", "The user usually prefers savory foods to very sweet foods."),
        ("no_caffeine_habit", "The user generally does not rely on caffeine."),
        ("food_variety_interest", "The user likes trying cuisines they have not had before."),
    ),
    "sleep": (
        ("afternoon_sleepiness", "The user sometimes becomes sleepy in the early afternoon."),
        ("nap_history", "The user has used afternoon naps at times in the past."),
        ("morning_wake_time", "The user often wakes in the morning around seven or eight."),
        ("bedtime_consistency_goal", "The user would like to keep a more consistent bedtime."),
        ("sleep_duration_interest", "The user pays attention to how many hours of sleep they get."),
        ("sleep_environment_interest", "The user prefers a quiet environment for sleeping."),
        ("fatigue_tracking_interest", "The user is interested in understanding patterns in daytime fatigue."),
        ("sleep_schedule_weekends", "The user's sleep timing can differ somewhat on weekends."),
        ("morning_routine_interest", "The user would like a more consistent morning routine."),
        ("evening_screen_use", "The user sometimes uses computers late in the evening."),
    ),
}


def noisy_synthetic_beliefs(*, distractor_limit: int | None = None) -> list[dict]:
    """Core benchmark memories plus a deterministic bank of adjacent distractors."""
    core = synthetic_beliefs()
    rows = [
        (topic_key, text)
        for group in NOISY_MEMORY_GROUPS.values()
        for topic_key, text in group
    ]
    if distractor_limit is not None:
        if distractor_limit < 0:
            raise ValueError("distractor_limit must be non-negative")
        rows = rows[:distractor_limit]

    distractors = [
        {
            "belief_id": f"noise_{index:03d}",
            "topic_key": topic_key,
            "text": text,
            "open_question": None,
            "evidence_status": "verified",
            "revision": 1,
            "updated_at": f"2026-02-{(index % 28) + 1:02d}T12:00:00",
        }
        for index, (topic_key, text) in enumerate(rows)
    ]
    return core + distractors


def pipeline_benchmark_cases() -> list[PipelineRecallCase]:
    """Unseen end-to-end cases for retrieval plus final relevance gating."""
    return [
        PipelineRecallCase(
            "pipeline_patient_contact",
            "I miss having actual conversations with patients during my workday.",
            ("job_patient_interaction_level",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_school_goal",
            "I need to keep moving toward eventually submitting my med-school application.",
            ("professional_school_goal",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_volunteer_constraint",
            "My full-time weekday schedule keeps crowding out regular community service.",
            ("volunteering_schedule_constraint",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_spain_relocation",
            "I'm still undecided about whether living in Spain is actually right for me.",
            ("country_relocation_deliberation",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_remote_work",
            "I still care about having at least some days where I can work from home.",
            ("remote_work_preference",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_pottery",
            "I want to spend more time at the ceramics studio making things from clay.",
            ("pottery_hobby",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_spicy_thai",
            "I could really go for something hot and Thai tonight.",
            ("food_preference_spicy",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_sleep_schedule",
            "My usual sleep window is still roughly midnight until seven in the morning.",
            ("sleep_schedule",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_keyboard",
            "For typing I still prefer switches that are linear and very quiet.",
            ("keyboard_preference",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_pc_quality",
            "I care more about trustworthy PC components than squeezing out the lowest purchase price.",
            ("computer_build_priority",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_horror_game",
            "I want to keep developing my strange suburban horror game.",
            ("horror_game_project",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_healthcare_ai",
            "Applied AI is most interesting to me when it solves a real healthcare problem.",
            ("research_interest",),
            "positive",
        ),
        PipelineRecallCase(
            "pipeline_patient_and_school",
            "I want more direct patient experience while I keep preparing for medical school.",
            ("job_patient_interaction_level", "professional_school_goal"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "pipeline_patient_and_volunteer",
            "I want both more patient-facing work and enough room in my schedule to volunteer regularly.",
            ("job_patient_interaction_level", "volunteering_schedule_constraint"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "pipeline_keyboard_and_pc",
            "For my next PC I want a nearly silent linear keyboard and components from manufacturers I trust.",
            ("keyboard_preference", "computer_build_priority"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "pipeline_research_and_school",
            "I want to keep building useful healthcare AI projects while working toward medical school.",
            ("research_interest", "professional_school_goal"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "pipeline_spain_update",
            "I'm becoming less convinced that moving to Spain would suit me.",
            ("country_relocation_deliberation",),
            "update",
        ),
        PipelineRecallCase(
            "pipeline_remote_update",
            "I'm starting to wonder whether remote work matters as much to me as it used to.",
            ("remote_work_preference",),
            "update",
        ),
        PipelineRecallCase(
            "pipeline_general_patient",
            "What jobs in healthcare involve the most face-to-face patient contact?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_general_volunteer",
            "How do full-time workers usually find time for community volunteering?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_general_school",
            "What are the usual prerequisite courses for medical school?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_general_remote",
            "Which industries have the highest proportion of remote jobs?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_general_keyboard",
            "Why are linear switches often quieter than clicky keyboard switches?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_general_pc",
            "What component brands are considered reliable for desktop computers?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_general_spain",
            "What are the major differences between living in Madrid and Barcelona?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_general_pottery",
            "How long does pottery normally need to dry before firing?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_general_horror",
            "What camera perspectives work well in horror games?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_general_sleep",
            "Why do many people naturally wake after about seven hours of sleep?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "pipeline_ambiguous_more_time",
            "I wish I had more time for the things I care about.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "pipeline_ambiguous_back_to_it",
            "I've been thinking about getting back to it seriously.",
            (),
            "ambiguous_negative",
        ),
    ]


async def run_pipeline_case(
    case: PipelineRecallCase,
    beliefs: list[dict],
    semantic_retriever: OllamaSemanticMemoryRetriever,
    *,
    lexical_limit: int = 4,
    candidate_limit: int = 8,
    total_limit: int = 4,
) -> dict:
    started = perf_counter()

    lexical_started = perf_counter()
    lexical = explain_relevant_beliefs(case.text, beliefs, limit=lexical_limit)
    lexical_ms = (perf_counter() - lexical_started) * 1000

    semantic_started = perf_counter()
    details = await semantic_retriever.recall_hybrid(
        case.text,
        beliefs,
        lexical_candidates=lexical,
        total_limit=total_limit,
        candidate_limit=candidate_limit,
    )
    semantic_ms = (perf_counter() - semantic_started) * 1000

    status = semantic_retriever.status()
    diagnostics = status.get("last_candidates", [])
    gate_status = status.get("relevance_gate", {})
    evaluated = [
        row
        for row in diagnostics
        if row.get("decision") in {"reranker_accept", "reranker_reject"}
    ]

    expected = set(case.expected_topics)
    actual = {str(detail["belief"]["topic_key"]) for detail in details}
    pool_topics = {str(row.get("topic_key", "")) for row in evaluated}
    gate_accepted_topics = {
        str(row.get("topic_key", ""))
        for row in evaluated
        if bool(row.get("selected"))
    }

    retrieval_misses = sorted(expected - pool_topics)
    gate_false_negatives = sorted(
        (expected & pool_topics) - gate_accepted_topics
    )
    final_limit_displacements = sorted(
        (expected & gate_accepted_topics) - actual
    )
    gate_false_positives = sorted(gate_accepted_topics - expected)
    final_false_positives = sorted(actual - expected)
    correct_rejections = sorted(
        (pool_topics - expected) - gate_accepted_topics
    )

    expected_in_pool = len(expected & pool_topics)
    expected_count = len(expected)

    return {
        "case_id": case.case_id,
        "category": case.category,
        "text": case.text,
        "memory_store_size": len(beliefs),
        "expected_topics": sorted(expected),
        "candidate_pool_topics": [str(row.get("topic_key", "")) for row in evaluated],
        "gate_accepted_topics": sorted(gate_accepted_topics),
        "actual_topics": sorted(actual),
        "exact": actual == expected,
        "candidate_recall_numerator": expected_in_pool,
        "candidate_recall_denominator": expected_count,
        "true_positive": len(expected & actual),
        "false_positive": len(actual - expected),
        "false_negative": len(expected - actual),
        "retrieval_misses": retrieval_misses,
        "gate_false_negatives": gate_false_negatives,
        "final_limit_displacements": final_limit_displacements,
        "gate_false_positives": gate_false_positives,
        "final_false_positives": final_false_positives,
        "correct_rejections": correct_rejections,
        "lexical_candidates": [
            {
                "topic_key": str(row["belief"]["topic_key"]),
                "score": row.get("score"),
                "matched_tokens": row.get("matched_tokens", []),
            }
            for row in lexical
        ],
        "candidate_diagnostics": diagnostics,
        "relevance_gate": gate_status,
        "gate_error": gate_status.get("last_error"),
        "latency_ms": {
            "total": round((perf_counter() - started) * 1000, 3),
            "lexical": round(lexical_ms, 3),
            "retrieval_and_gate": round(semantic_ms, 3),
            "gate": round(
                float(gate_status.get("last_metrics", {}).get("total_ms", 0) or 0),
                3,
            ),
        },
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize_pipeline_results(results: list[dict]) -> dict:
    tp = sum(int(row["true_positive"]) for row in results)
    fp = sum(int(row["false_positive"]) for row in results)
    fn = sum(int(row["false_negative"]) for row in results)
    exact = sum(1 for row in results if row["exact"])

    expected_memories = sum(len(row["expected_topics"]) for row in results)
    expected_in_pool = sum(int(row["candidate_recall_numerator"]) for row in results)
    retrieval_misses = sum(len(row["retrieval_misses"]) for row in results)
    gate_false_negatives = sum(len(row["gate_false_negatives"]) for row in results)
    final_limit_displacements = sum(
        len(row.get("final_limit_displacements", [])) for row in results
    )
    gate_false_positives = sum(len(row["gate_false_positives"]) for row in results)
    final_false_positives = sum(
        len(row.get("final_false_positives", row["gate_false_positives"]))
        for row in results
    )
    correct_rejections = sum(len(row["correct_rejections"]) for row in results)

    protocol_failures = sum(1 for row in results if row.get("gate_error"))
    negatives = [row for row in results if not row["expected_topics"]]
    negative_passes = sum(1 for row in negatives if not row["actual_topics"])

    candidate_pool_sizes = [len(row["candidate_pool_topics"]) for row in results]
    total_latencies = [float(row["latency_ms"]["total"]) for row in results]
    gate_latencies = [
        float(row["latency_ms"]["gate"])
        for row in results
        if float(row["latency_ms"]["gate"]) > 0
    ]

    category_summary: dict[str, dict] = {}
    for category in sorted({str(row["category"]) for row in results}):
        rows = [row for row in results if row["category"] == category]
        category_summary[category] = {
            "cases": len(rows),
            "exact_cases": sum(1 for row in rows if row["exact"]),
            "retrieval_misses": sum(len(row["retrieval_misses"]) for row in rows),
            "gate_false_negatives": sum(
                len(row["gate_false_negatives"]) for row in rows
            ),
            "final_limit_displacements": sum(
                len(row.get("final_limit_displacements", [])) for row in rows
            ),
            "gate_false_positives": sum(
                len(row["gate_false_positives"]) for row in rows
            ),
        }

    return {
        "cases": len(results),
        "exact_cases": exact,
        "exact_case_accuracy": _ratio(exact, len(results)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "negative_control_accuracy": _ratio(negative_passes, len(negatives)),
        "candidate_pool_recall": _ratio(expected_in_pool, expected_memories),
        "expected_memories": expected_memories,
        "expected_memories_in_pool": expected_in_pool,
        "retrieval_misses": retrieval_misses,
        "gate_false_negatives": gate_false_negatives,
        "final_limit_displacements": final_limit_displacements,
        "gate_false_positives": gate_false_positives,
        "final_false_positives": final_false_positives,
        "correct_rejections": correct_rejections,
        "protocol_failures": protocol_failures,
        "median_candidate_pool_size": (
            round(median(candidate_pool_sizes), 3) if candidate_pool_sizes else 0.0
        ),
        "category_summary": category_summary,
        "latency_ms": {
            "median_total": round(median(total_latencies), 3) if total_latencies else 0.0,
            "median_gate": round(median(gate_latencies), 3) if gate_latencies else 0.0,
            "max_gate": round(max(gate_latencies), 3) if gate_latencies else 0.0,
        },
    }
