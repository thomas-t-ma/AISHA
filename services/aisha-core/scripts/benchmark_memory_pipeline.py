from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aisha.memory.pipeline_benchmark import (
    NOISY_MEMORY_GROUPS,
    noisy_synthetic_beliefs,
    pipeline_benchmark_cases,
    run_pipeline_case,
    summarize_pipeline_results,
)
from aisha.memory.relevance import OllamaMemoryRelevanceGate
from aisha.memory.semantic import OllamaSemanticMemoryRetriever
from aisha.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AISHA's full memory retrieval -> relevance-gate pipeline."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Relevance-gate model. Repeat to compare models.",
    )
    parser.add_argument(
        "--profile",
        default="mac-m2max-96gb",
        help="AISHA runtime profile supplying embedding and Ollama settings.",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=None,
        help=(
            "Number of adjacent distractor memories to include. "
            "Default: all available distractors."
        ),
    )
    parser.add_argument(
        "--lexical-limit",
        type=int,
        default=4,
        help="Maximum lexical candidates before semantic expansion.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=8,
        help="Maximum semantic candidates sent into the final candidate union.",
    )
    parser.add_argument(
        "--total-limit",
        type=int,
        default=4,
        help="Maximum memories allowed through the final gate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def build_retriever(model: str, profile) -> OllamaSemanticMemoryRetriever:
    gate = OllamaMemoryRelevanceGate(
        model=model,
        base_url=profile.memory.semantic_relevance_base_url or profile.llm.base_url,
        keep_alive=(
            profile.memory.semantic_relevance_keep_alive
            if profile.memory.semantic_relevance_keep_alive is not None
            else profile.llm.keep_alive
        ),
    )
    return OllamaSemanticMemoryRetriever(
        model=profile.memory.embedding_model,
        base_url=profile.memory.embedding_base_url or profile.llm.base_url,
        threshold=profile.memory.semantic_threshold,
        candidate_floor=profile.memory.semantic_candidate_floor,
        limit=profile.memory.semantic_limit,
        keep_alive=profile.memory.embedding_keep_alive,
        relevance_gate=gate,
        query_instruction=profile.memory.semantic_query_instruction,
    )


async def evaluate_model(
    model: str,
    *,
    profile,
    beliefs: list[dict],
    cases,
    lexical_limit: int,
    candidate_limit: int,
    total_limit: int,
) -> dict:
    retriever = build_retriever(model, profile)
    results = []
    for case in cases:
        results.append(
            await run_pipeline_case(
                case,
                beliefs,
                retriever,
                lexical_limit=lexical_limit,
                candidate_limit=candidate_limit,
                total_limit=total_limit,
            )
        )
    return {
        "model": model,
        "summary": summarize_pipeline_results(results),
        "results": results,
    }


def print_result(result: dict) -> None:
    summary = result["summary"]
    print(f"\n{result['model']}")
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

    failures = [row for row in result["results"] if not row["exact"]]
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
        if row.get("final_limit_displacements"):
            print(
                "      FINAL-LIMIT DISPLACEMENT: "
                f"{', '.join(row['final_limit_displacements'])}"
            )
        if row["gate_false_positives"]:
            print(f"      GATE FALSE POSITIVE: {', '.join(row['gate_false_positives'])}")
        gate_status = row.get("relevance_gate", {})
        if gate_status.get("message_scope"):
            print(
                f"      scope: {gate_status.get('message_scope')} · "
                f"{gate_status.get('scope_reason') or ''}"
            )
        propositions = gate_status.get("propositions") or []
        for proposition in propositions:
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

    if not profile.llm.base_url:
        raise SystemExit(f"Profile {args.profile!r} does not define an Ollama URL.")
    if not profile.memory.embedding_model:
        raise SystemExit(f"Profile {args.profile!r} does not define an embedding model.")

    available_distractors = sum(len(rows) for rows in NOISY_MEMORY_GROUPS.values())
    distractors = available_distractors if args.distractors is None else args.distractors
    if distractors < 0 or distractors > available_distractors:
        raise SystemExit(
            f"--distractors must be between 0 and {available_distractors}; "
            f"received {distractors}."
        )

    beliefs = noisy_synthetic_beliefs(distractor_limit=distractors)
    cases = pipeline_benchmark_cases()

    print(
        "AISHA end-to-end memory pipeline benchmark\n"
        f"Cases: {len(cases)} · Memories: {len(beliefs)} "
        f"(12 targets + {distractors} distractors)\n"
        f"Embedding: {profile.memory.embedding_model} · "
        f"Lexical limit: {args.lexical_limit} · "
        f"Semantic candidate limit: {args.candidate_limit} · "
        f"Final limit: {args.total_limit}\n"
        f"Gate models: {', '.join(args.models)}"
    )

    comparisons = []
    for model in args.models:
        print(f"\nRunning {model}...")
        comparisons.append(
            await evaluate_model(
                model,
                profile=profile,
                beliefs=beliefs,
                cases=cases,
                lexical_limit=args.lexical_limit,
                candidate_limit=args.candidate_limit,
                total_limit=args.total_limit,
            )
        )

    print("\nCOMPARISON")
    for result in comparisons:
        print_result(result)

    payload = {
        "profile": args.profile,
        "embedding_model": profile.memory.embedding_model,
        "memory_store_size": len(beliefs),
        "target_memories": 12,
        "distractor_memories": distractors,
        "lexical_limit": args.lexical_limit,
        "candidate_limit": args.candidate_limit,
        "total_limit": args.total_limit,
        "models": comparisons,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
