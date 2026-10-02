from __future__ import annotations

from aisha.memory.benchmark import synthetic_beliefs
from aisha.memory.pipeline_benchmark import PipelineRecallCase

VALIDATION_SUITE_VERSION = "unseen-v1-2026-09-30"

VALIDATION_DISTRACTOR_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "work": (
        ("validation_onsite_work_preference", "The user likes having access to a physical workplace when collaboration is easier in person."),
        ("validation_schedule_predictability", "The user values knowing their work schedule well in advance."),
        ("validation_leadership_interest", "The user is interested in eventually supervising projects or small teams."),
        ("validation_coworker_social_preference", "The user enjoys friendly social interaction with coworkers."),
        ("validation_commute_tolerance", "The user can tolerate a moderate commute for the right role."),
        ("validation_job_prestige_concern", "The user sometimes considers institutional reputation when comparing jobs."),
    ),
    "clinical": (
        ("validation_bedside_procedure_interest", "The user is interested in learning hands-on bedside procedures."),
        ("validation_patient_counseling_confidence", "The user wants to become more confident explaining care plans to patients."),
        ("validation_clinic_pace_preference", "The user prefers clinical environments that are busy but organized."),
        ("validation_inpatient_care_interest", "The user is interested in how inpatient medical teams manage hospitalized patients."),
        ("validation_clinical_documentation_efficiency", "The user values efficient clinical documentation workflows."),
        ("validation_rounds_interest", "The user likes the idea of participating in interdisciplinary clinical rounds."),
    ),
    "school": (
        ("validation_application_essay_concern", "The user expects personal statements to be an important part of professional-school applications."),
        ("validation_school_class_size_preference", "The user would prefer a medical school where students can know faculty personally."),
        ("validation_admissions_competitiveness_concern", "The user thinks about how competitive professional-school admissions can be."),
        ("validation_interview_preparation_goal", "The user expects to practice for future admissions interviews."),
        ("validation_curriculum_style_preference", "The user is interested in medical programs with integrated clinical and classroom learning."),
        ("validation_gap_year_openness", "The user is open to taking additional time before starting professional school if useful."),
    ),
    "service": (
        ("validation_event_service_preference", "The user is open to occasional event-based volunteering."),
        ("validation_solo_service_preference", "The user is comfortable doing some volunteer work independently."),
        ("validation_youth_mentorship_interest", "The user is interested in mentoring younger students."),
        ("validation_food_bank_interest", "The user is open to volunteering with food-distribution programs."),
        ("validation_service_transport_constraint", "Transportation can affect which volunteer opportunities are practical for the user."),
        ("validation_service_leadership_interest", "The user would consider coordinating a volunteer project in the future."),
    ),
    "research": (
        ("validation_translational_research_interest", "The user likes research that can eventually affect real-world practice."),
        ("validation_large_dataset_interest", "The user enjoys projects that involve large datasets."),
        ("validation_statistical_rigor_value", "The user values careful statistical analysis and well-defined evaluation plans."),
        ("validation_model_deployment_interest", "The user is interested in deploying machine-learning models into usable software."),
        ("validation_conference_presentation_goal", "The user would like technical work to be strong enough to present at conferences."),
        ("validation_research_collaboration_preference", "The user enjoys research projects that involve collaborators with complementary expertise."),
    ),
    "computer": (
        ("validation_pc_brand_transparency", "The user values computer vendors that clearly identify the parts they install."),
        ("validation_pc_power_efficiency", "The user cares about reasonable power efficiency in a desktop computer."),
        ("validation_pc_port_selection", "The user values having enough high-speed ports for peripherals and external devices."),
        ("validation_motherboard_quality", "The user cares about motherboard quality when evaluating a desktop build."),
        ("validation_psu_quality", "The user considers power-supply quality important in a reliable PC."),
        ("validation_pc_repairability", "The user prefers computers that can be repaired without proprietary obstacles."),
    ),
    "keyboard": (
        ("validation_tactile_switch_curiosity", "The user is curious about trying tactile keyboard switches."),
        ("validation_compact_layout_interest", "The user sometimes considers compact keyboard layouts for saving desk space."),
        ("validation_wired_keyboard_preference", "The user is comfortable using a wired keyboard for reliability."),
        ("validation_hotswap_keyboard_interest", "The user likes keyboards that allow switches to be changed without soldering."),
        ("validation_wrist_rest_interest", "The user values a comfortable wrist position during long typing sessions."),
        ("validation_keyboard_knob_interest", "The user finds a physical volume knob useful on a keyboard."),
    ),
    "creative": (
        ("validation_sculpture_interest", "The user is interested in three-dimensional sculptural art."),
        ("validation_photography_interest", "The user enjoys photography as a creative activity."),
        ("validation_digital_illustration_interest", "The user is interested in making digital illustrations."),
        ("validation_gallery_interest", "The user enjoys visiting galleries and seeing finished artwork in person."),
        ("validation_woodworking_interest", "The user is curious about learning woodworking."),
        ("validation_music_hobby", "The user enjoys music as a recreational hobby."),
    ),
    "games": (
        ("validation_puzzle_design_interest", "The user enjoys thinking about puzzle design in games."),
        ("validation_survival_horror_interest", "The user likes survival-horror games as a genre."),
        ("validation_retro_graphics_interest", "The user is interested in retro-inspired game visuals."),
        ("validation_first_person_game_interest", "The user likes the immediacy of first-person game perspectives."),
        ("validation_game_ui_interest", "The user is interested in how game interfaces communicate information."),
        ("validation_level_design_interest", "The user enjoys thinking about how game levels guide player movement."),
    ),
    "travel": (
        ("validation_spanish_language_interest", "The user is interested in learning more Spanish for travel or daily life."),
        ("validation_relocation_paperwork_concern", "The user thinks immigration paperwork is an important part of international relocation."),
        ("validation_japan_relocation_interest", "The user has considered what it would be like to live in Japan."),
        ("validation_climate_adaptation_concern", "The user thinks climate can affect how comfortable a place is to live."),
        ("validation_abroad_housing_cost_concern", "The user pays attention to housing costs when considering life abroad."),
        ("validation_cultural_integration_interest", "The user is interested in how newcomers build community in another country."),
    ),
    "food": (
        ("validation_curry_interest", "The user enjoys flavorful curry dishes."),
        ("validation_chili_sauce_interest", "The user likes adding chili sauces to some meals."),
        ("validation_thai_dessert_interest", "The user enjoys trying Thai desserts."),
        ("validation_noodle_interest", "The user likes noodle dishes from different cuisines."),
        ("validation_seafood_dislike", "The user is not especially enthusiastic about seafood."),
        ("validation_quiet_restaurant_preference", "The user prefers restaurants where conversation is easy."),
    ),
    "sleep": (
        ("validation_sleep_onset_latency", "The user pays attention to how long it takes to fall asleep."),
        ("validation_alarm_dependence", "The user sometimes relies on an alarm to wake on time."),
        ("validation_earlier_bedtime_interest", "The user has considered going to bed earlier."),
        ("validation_sleep_tracker_interest", "The user is interested in what wearable devices can reveal about sleep."),
        ("validation_weekend_nap_interest", "The user sometimes considers taking naps on weekends."),
        ("validation_bedroom_temperature_preference", "The user prefers a comfortable cool bedroom for sleep."),
    ),
}


def validation_beliefs() -> list[dict]:
    """Frozen target memories plus a distractor bank unused by the dev suite."""
    core = synthetic_beliefs()
    rows = [
        (topic_key, text)
        for group in VALIDATION_DISTRACTOR_GROUPS.values()
        for topic_key, text in group
    ]
    distractors = [
        {
            "belief_id": f"validation_noise_{index:03d}",
            "topic_key": topic_key,
            "text": text,
            "open_question": None,
            "evidence_status": "verified",
            "revision": 1,
            "updated_at": f"2026-03-{(index % 28) + 1:02d}T12:00:00",
        }
        for index, (topic_key, text) in enumerate(rows)
    ]
    return core + distractors


def validation_cases() -> list[PipelineRecallCase]:
    """Frozen unseen cases. Do not tune prompts or thresholds against these."""
    return [
        # One fresh positive for every core target.
        PipelineRecallCase(
            "validation_patient_state",
            "Most days in my current role I barely get to speak directly with the people receiving care.",
            ("job_patient_interaction_level",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_school_intent",
            "I still intend to apply to medical school once my application is ready.",
            ("professional_school_goal",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_service_schedule",
            "My weekday job makes it hard to commit to community service on a regular basis.",
            ("volunteering_schedule_constraint",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_spain_undecided",
            "Spain is still only a possible place for me to live; I have not made that decision.",
            ("country_relocation_deliberation",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_remote_value",
            "Having some ability to do my job from home remains important to me.",
            ("remote_work_preference",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_ceramics_hobby",
            "Working with clay at a ceramics studio is one of the hobbies I want more time for.",
            ("pottery_hobby",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_spicy_thai_preference",
            "Thai food tastes best to me when it has a serious amount of heat.",
            ("food_preference_spicy",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_sleep_window",
            "On a normal night I am asleep around midnight and awake again around seven.",
            ("sleep_schedule",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_keyboard_feel",
            "My ideal keyboard switch is still smooth, linear, and quiet.",
            ("keyboard_preference",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_pc_parts",
            "I would rather pay for dependable, clearly identified PC parts than save money on mystery components.",
            ("computer_build_priority",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_horror_project",
            "The liminal suburban horror game is still a personal project I want to finish building.",
            ("horror_game_project",),
            "positive",
        ),
        PipelineRecallCase(
            "validation_healthcare_ai_interest",
            "The AI projects I enjoy most are the ones applied to real healthcare problems.",
            ("research_interest",),
            "positive",
        ),

        # Fresh cross-domain combinations.
        PipelineRecallCase(
            "validation_patient_service_combo",
            "I want a role with more direct patient contact, but my weekdays also leave little room for regular volunteering.",
            ("job_patient_interaction_level", "volunteering_schedule_constraint"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "validation_school_research_combo",
            "I want to keep doing healthcare AI work as I prepare to apply to medical school.",
            ("research_interest", "professional_school_goal"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "validation_spain_remote_combo",
            "I am still undecided about living in Spain, and having some work-from-home flexibility would matter to me wherever I live.",
            ("country_relocation_deliberation", "remote_work_preference"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "validation_keyboard_pc_combo",
            "For a future desktop I care about trustworthy named components and a keyboard that stays quiet with linear switches.",
            ("computer_build_priority", "keyboard_preference"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "validation_food_ceramics_combo",
            "I still love spicy Thai food, and I want to get back to spending more time making pottery.",
            ("food_preference_spicy", "pottery_hobby"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "validation_game_research_combo",
            "Two projects I want to keep pursuing are my suburban horror game and applied AI work in healthcare.",
            ("horror_game_project", "research_interest"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "validation_school_service_combo",
            "I am still planning for medical school even though my weekday schedule makes regular volunteering difficult.",
            ("professional_school_goal", "volunteering_schedule_constraint"),
            "multi_memory",
        ),
        PipelineRecallCase(
            "validation_sleep_remote_combo",
            "My normal sleep is roughly midnight to seven, and I still value having some work I can do from home.",
            ("sleep_schedule", "remote_work_preference"),
            "multi_memory",
        ),

        # Updates on explicitly named threads.
        PipelineRecallCase(
            "validation_spain_update",
            "Lately I am leaning away from the idea of actually relocating to Spain.",
            ("country_relocation_deliberation",),
            "update",
        ),
        PipelineRecallCase(
            "validation_remote_update",
            "I am less sure than I used to be that working from home should matter much in my next job.",
            ("remote_work_preference",),
            "update",
        ),
        PipelineRecallCase(
            "validation_pottery_update",
            "I have been less motivated to spend time at the ceramics studio lately.",
            ("pottery_hobby",),
            "update",
        ),
        PipelineRecallCase(
            "validation_keyboard_update",
            "I still like linear switches, but I am starting to care less about absolute keyboard silence.",
            ("keyboard_preference",),
            "update",
        ),

        # General informational negatives.
        PipelineRecallCase(
            "validation_general_patient",
            "Why do some healthcare jobs involve much more direct patient contact than others?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_school",
            "How early do applicants usually begin preparing a medical school application?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_service",
            "What scheduling strategies help full-time employees volunteer consistently?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_spain",
            "What paperwork is normally required for an American to move to Spain?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_remote",
            "How do hybrid workplaces usually decide which days employees work from home?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_ceramics",
            "What is the difference between stoneware clay and porcelain clay?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_food",
            "Which Thai dishes are traditionally the hottest?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_sleep",
            "How much does bedtime vary across adults with conventional work schedules?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_keyboard",
            "What physical properties make one mechanical switch quieter than another?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_pc",
            "How can a buyer verify which components a prebuilt PC actually contains?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_game",
            "What level-design techniques make suburban environments feel uncanny in horror games?",
            (),
            "general_negative",
        ),
        PipelineRecallCase(
            "validation_general_ai",
            "What are common ways artificial intelligence is evaluated in healthcare research?",
            (),
            "general_negative",
        ),

        # Personal-sounding but deliberately unanchored negatives.
        PipelineRecallCase(
            "validation_ambiguous_more",
            "I keep feeling like I need more of it in my life.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_priority",
            "I am starting to wonder if it should still be such a big priority for me.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_return",
            "I really want to get back into that when I have the chance.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_schedule",
            "My schedule keeps getting in the way of what I want to do.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_change",
            "I think my preferences around that are changing.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_project",
            "I have been thinking a lot about whether to keep working on it.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_place",
            "I still cannot decide whether that place is really right for me.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_hobby",
            "I miss doing that kind of thing with my hands.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_work",
            "I want something different from work, but I cannot put my finger on what.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_goal",
            "I know I still want to pursue it eventually.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_food",
            "I have really been craving that kind of thing lately.",
            (),
            "ambiguous_negative",
        ),
        PipelineRecallCase(
            "validation_ambiguous_routine",
            "My usual routine around it has been shifting a bit.",
            (),
            "ambiguous_negative",
        ),
    ]
