from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from aisha.memory.pipeline_benchmark import run_pipeline_case, summarize_pipeline_results
from aisha.memory.relevance import (
    RELEVANCE_GATE_PROMPT_VERSION,
    OllamaMemoryRelevanceGate,
)
from aisha.memory.semantic import (
    SEMANTIC_PIPELINE_VERSION,
    OllamaSemanticMemoryRetriever,
)
from aisha.memory.validation_benchmark import (
    VALIDATION_SUITE_VERSION,
    validation_beliefs,
    validation_cases,
)
from aisha.settings import Settings

ALLOWED_PROFILES = {"mac-m2max-96gb", "nvidia-5080"}
FROZEN_GATE_MODELS = {
    "mac-m2max-96gb": "qwen3.5:35b-mlx",
    "nvidia-5080": "qwen3.5:35b",
}
FROZEN_EMBEDDING_MODEL = "qwen3-embedding:4b"
FROZEN_GATE_VERSION = "personal-continuity-v10.2-source-grounded-anchors"
FROZEN_PIPELINE_VERSION = "hybrid-final-gate-v10.2-source-grounded-anchors"
FROZEN_LEXICAL_LIMIT = 4
FROZEN_CANDIDATE_LIMIT = 8
FROZEN_TOTAL_LIMIT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AISHA's frozen unseen memory validation suite. "
            "Do not tune memory behavior against this suite."
        )
    )
    parser.add_argument(
        "--profile",
        choices=sorted(ALLOWED_PROFILES),
        default="nvidia-5080",
        help=(
            "Hardware profile. Only profiles with the frozen model/memory "
            "configuration are allowed."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/aisha-memory-validation-unseen-v1.json"),
        help="JSON output path.",
    )
    return parser.parse_args()


def assert_frozen_configuration(profile, *, profile_name: str) -> None:
    problems: list[str] = []
    if profile_name not in ALLOWED_PROFILES:
        problems.append(
            f"profile {profile_name!r} is not one of {sorted(ALLOWED_PROFILES)!r}"
        )
    if RELEVANCE_GATE_PROMPT_VERSION != FROZEN_GATE_VERSION:
        problems.append(
            f"gate version {RELEVANCE_GATE_PROMPT_VERSION!r} != {FROZEN_GATE_VERSION!r}"
        )
    if SEMANTIC_PIPELINE_VERSION != FROZEN_PIPELINE_VERSION:
        problems.append(
            f"pipeline version {SEMANTIC_PIPELINE_VERSION!r} "
            f"!= {FROZEN_PIPELINE_VERSION!r}"
        )
    expected_gate_model = FROZEN_GATE_MODELS[profile_name]
    if profile.llm.model != expected_gate_model:
        problems.append(
            f"gate model {profile.llm.model!r} != {expected_gate_model!r}"
        )
    if profile.memory.embedding_model != FROZEN_EMBEDDING_MODEL:
        problems.append(
            f"embedding model {profile.memory.embedding_model!r} "
            f"!= {FROZEN_EMBEDDING_MODEL!r}"
        )
    if problems:
        joined = "\n  - ".join(problems)
        raise SystemExit(
            "Frozen validation configuration does not match:\n"
            f"  - {joined}\n"
            "Do not modify the validation suite to accommodate a changed configuration."
        )


async def assert_ollama_ready(profile) -> None:
    base_url = (profile.llm.base_url or "").rstrip("/")
    if not base_url:
        raise SystemExit("Selected profile has no Ollama base URL.")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        raise SystemExit(
            "Ollama is not reachable at "
            f"{base_url}. Start Ollama before running the holdout. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    installed = {
        str(row.get("name") or row.get("model") or "")
        for row in payload.get("models", [])
        if isinstance(row, dict)
    }
    required = {str(profile.llm.model), FROZEN_EMBEDDING_MODEL}
    missing = sorted(model for model in required if model not in installed)
    if missing:
        commands = "\n".join(f"  ollama pull {model}" for model in missing)
        raise SystemExit(
            "Ollama is running, but required validation models are missing:\n"
            f"{commands}"
        )


def build_retriever(profile) -> OllamaSemanticMemoryRetriever:
    gate = OllamaMemoryRelevanceGate(
        model=profile.llm.model,
        base_url=profile.memory.semantic_relevance_base_url or profile.llm.base_url,
        keep_alive=(
            profile.memory.semantic_relevance_keep_alive
            if profile.memory.semantic_relevance_keep_alive is not None
            else profile.llm.keep_alive
        ),
    )
    return OllamaSemanticMemoryRetriever(
        model=FROZEN_EMBEDDING_MODEL,
        base_url=profile.memory.embedding_base_url or profile.llm.base_url,
        threshold=profile.memory.semantic_threshold,
        candidate_floor=profile.memory.semantic_candidate_floor,
        limit=profile.memory.semantic_limit,
        keep_alive=profile.memory.embedding_keep_alive,
        relevance_gate=gate,
        query_instruction=profile.memory.semantic_query_instruction,
    )


def print_summary(summary: dict) -> None:
    print(
        f"  exact: {summary['exact_cases']}/{summary['cases']} "
        f"({summary['exact_case_accuracy']:.1%})"
    )
    print(
        f"  final precision: {summary['precision']:.1%} · "
        f"final recall: {summary['recall']:.1%} · "
        f"negatives: {summary['negative_control_accuracy']:.1%}"
    )
    print(
        f"  candidate-pool recall: {summary['candidate_pool_recall']:.1%} "
        f"({summary['expected_memories_in_pool']}/{summary['expected_memories']})"
    )
    print(
        f"  retrieval misses: {summary['retrieval_misses']} · "
        f"gate false negatives: {summary['gate_false_negatives']} · "
        f"final-limit displacements: {summary['final_limit_displacements']} · "
        f"gate false positives: {summary['gate_false_positives']} · "
        f"final false positives: {summary['final_false_positives']} · "
        f"correct rejections: {summary['correct_rejections']}"
    )
    print(
        f"  median pool: {summary['median_candidate_pool_size']:.1f} · "
        f"median total: {summary['latency_ms']['median_total']:.0f} ms · "
        f"median gate: {summary['latency_ms']['median_gate']:.0f} ms · "
        f"max gate: {summary['latency_ms']['max_gate']:.0f} ms · "
        f"protocol failures: {summary['protocol_failures']}"
    )
    print("  categories:")
    for category, row in summary["category_summary"].items():
        print(
            f"    {category}: {row['exact_cases']}/{row['cases']} exact · "
            f"retrieval miss {row['retrieval_misses']} · "
            f"gate FN {row['gate_false_negatives']} · "
            f"limit {row['final_limit_displacements']} · "
            f"gate FP {row['gate_false_positives']}"
        )


def print_failures(results: list[dict]) -> None:
    failures = [row for row in results if not row["exact"]]
    if not failures:
        print("  failures: none")
        return

    print(f"  failures: {len(failures)}")
    for row in failures:
        expected = ", ".join(row["expected_topics"]) or "none"
        actual = ", ".join(row["actual_topics"]) or "none"
        print(f"    {row['case_id']}: expected [{expected}] actual [{actual}]")
        if row["retrieval_misses"]:
            print(f"      RETRIEVAL MISS: {', '.join(row['retrieval_misses'])}")
        if row["gate_false_negatives"]:
            print(f"      GATE FALSE NEGATIVE: {', '.join(row['gate_false_negatives'])}")
        if row["gate_false_positives"]:
            print(f"      GATE FALSE POSITIVE: {', '.join(row['gate_false_positives'])}")
        if row.get("final_limit_displacements"):
            print(
                "      FINAL-LIMIT DISPLACEMENT: "
                f"{', '.join(row['final_limit_displacements'])}"
            )

        gate_status = row.get("relevance_gate", {})
        if gate_status.get("message_scope"):
            print(
                f"      scope: {gate_status.get('message_scope')} · "
                f"{gate_status.get('scope_reason') or ''}"
            )
        for proposition in gate_status.get("propositions") or []:
            print(
                f"      proposition {proposition.get('index')}: "
                f"{proposition.get('text')}"
            )
            if proposition.get("thread_core"):
                print(
                    "        thread: "
                    f"{proposition.get('thread_core')} · "
                    f"mode={proposition.get('continuity_mode')}"
                )
            if proposition.get("required_anchors"):
                print(
                    "        anchors: "
                    + ", ".join(str(item) for item in proposition["required_anchors"])
                )
            if proposition.get("anchor_evidence"):
                print(
                    "        evidence: "
                    + ", ".join(str(item) for item in proposition["anchor_evidence"])
                )
            if proposition.get("turn_modifiers"):
                print(
                    "        modifiers: "
                    + ", ".join(str(item) for item in proposition["turn_modifiers"])
                )

        if row.get("gate_error"):
            print(f"      GATE ERROR: {row['gate_error']}")
            if gate_status.get("protocol_phase"):
                print(f"      protocol phase: {gate_status['protocol_phase']}")
            if gate_status.get("protocol_response_preview"):
                print(
                    "      raw preview: "
                    f"{gate_status['protocol_response_preview']}"
                )

        expected_set = set(row["expected_topics"])
        for candidate in row["candidate_diagnostics"]:
            topic = str(candidate.get("topic_key", ""))
            if (
                topic in expected_set
                or candidate.get("selected")
                or candidate.get("decision") == "reranker_reject"
            ):
                print(
                    f"      {topic}: {candidate.get('decision')} · "
                    f"lex={candidate.get('lexical_score')} · "
                    f"sem={candidate.get('semantic_score')} · "
                    f"{candidate.get('reason', '')}"
                )


async def main() -> None:
    args = parse_args()
    settings = Settings(aisha_profile=args.profile)
    profile = settings.load_profile()
    assert_frozen_configuration(profile, profile_name=args.profile)

    beliefs = validation_beliefs()
    cases = validation_cases()
    await assert_ollama_ready(profile)
    retriever = build_retriever(profile)

    print(
        "AISHA FROZEN unseen memory validation\n"
        f"Suite: {VALIDATION_SUITE_VERSION}\n"
        f"Cases: {len(cases)} · Memories: {len(beliefs)} "
        f"(12 targets + {len(beliefs) - 12} unseen distractors)\n"
        f"Embedding: {FROZEN_EMBEDDING_MODEL} · "
        f"Lexical limit: {FROZEN_LEXICAL_LIMIT} · "
        f"Semantic candidate limit: {FROZEN_CANDIDATE_LIMIT} · "
        f"Final limit: {FROZEN_TOTAL_LIMIT}\n"
        f"Gate: {profile.llm.model}\n"
        f"Gate version: {FROZEN_GATE_VERSION}\n"
        f"Pipeline version: {FROZEN_PIPELINE_VERSION}\n"
        "IMPORTANT: treat this as holdout evaluation, not tuning data."
    )

    results = []
    for case in cases:
        results.append(
            await run_pipeline_case(
                case,
                beliefs,
                retriever,
                lexical_limit=FROZEN_LEXICAL_LIMIT,
                candidate_limit=FROZEN_CANDIDATE_LIMIT,
                total_limit=FROZEN_TOTAL_LIMIT,
            )
        )

    summary = summarize_pipeline_results(results)
    print("\nRESULT")
    print_summary(summary)
    print_failures(results)

    payload = {
        "suite_version": VALIDATION_SUITE_VERSION,
        "frozen_configuration": {
            "profile": args.profile,
            "gate_model": profile.llm.model,
            "embedding_model": FROZEN_EMBEDDING_MODEL,
            "gate_version": FROZEN_GATE_VERSION,
            "pipeline_version": FROZEN_PIPELINE_VERSION,
            "lexical_limit": FROZEN_LEXICAL_LIMIT,
            "candidate_limit": FROZEN_CANDIDATE_LIMIT,
            "total_limit": FROZEN_TOTAL_LIMIT,
        },
        "memory_store_size": len(beliefs),
        "target_memories": 12,
        "distractor_memories": len(beliefs) - 12,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
