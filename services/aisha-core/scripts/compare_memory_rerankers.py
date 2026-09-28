from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aisha.memory.benchmark import (
    benchmark_cases,
    run_case,
    summarize_results,
    synthetic_beliefs,
)
from aisha.memory.relevance import OllamaMemoryRelevanceGate
from aisha.memory.semantic import OllamaSemanticMemoryRetriever
from aisha.settings import Settings

SMOKE_CASES = {
    "patient_contact_paraphrase",
    "multi_work_and_volunteer",
    "negative_medical_school_general",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AISHA memory relevance-gate models on one fixed suite."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Reranker model to test. Repeat for multiple models.",
    )
    parser.add_argument(
        "--suite",
        choices=("smoke", "full"),
        default="smoke",
        help="Three hard cases or the full 20-case synthetic suite.",
    )
    parser.add_argument(
        "--profile",
        default="mac-m2max-96gb",
        help="AISHA runtime profile supplying embedding/retrieval settings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file for comparison results.",
    )
    return parser.parse_args()


async def evaluate_model(
    model: str,
    *,
    profile,
    cases,
    beliefs: list[dict],
) -> dict:
    base_url = profile.memory.semantic_relevance_base_url or profile.llm.base_url
    keep_alive = (
        profile.memory.semantic_relevance_keep_alive
        if profile.memory.semantic_relevance_keep_alive is not None
        else profile.llm.keep_alive
    )
    gate = OllamaMemoryRelevanceGate(
        model=model,
        base_url=base_url,
        keep_alive=keep_alive,
    )
    retriever = OllamaSemanticMemoryRetriever(
        model=profile.memory.embedding_model,
        base_url=profile.memory.embedding_base_url or profile.llm.base_url,
        threshold=profile.memory.semantic_threshold,
        candidate_floor=profile.memory.semantic_candidate_floor,
        limit=profile.memory.semantic_limit,
        keep_alive=profile.memory.embedding_keep_alive,
        relevance_gate=gate,
        query_instruction=profile.memory.semantic_query_instruction,
    )

    results: list[dict] = []
    for case in cases:
        result = await run_case(case, beliefs, retriever)
        results.append(result)

    return {
        "model": model,
        "summary": summarize_results(results),
        "results": results,
    }


def print_model_result(result: dict) -> None:
    summary = result["summary"]
    print(f"\n{result['model']}")
    print(
        f"  exact: {summary['exact_cases']}/{summary['cases']} "
        f"({summary['exact_case_accuracy']:.1%})"
    )
    print(
        f"  precision: {summary['precision']:.1%} · "
        f"recall: {summary['recall']:.1%} · "
        f"negatives: {summary['negative_control_accuracy']:.1%}"
    )
    print(
        f"  median retrieval: {summary['latency_ms']['median_total']:.0f} ms · "
        f"median gate: {summary['latency_ms']['median_gate']:.0f} ms · "
        f"max gate: {summary['latency_ms']['max_gate']:.0f} ms"
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
        for candidate in row["semantic_candidates"]:
            if candidate.get("decision") not in {"reranker_accept", "reranker_reject"}:
                continue
            print(
                "      "
                f"{candidate.get('topic_key')}: {candidate.get('decision')} · "
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

    all_cases = benchmark_cases()
    cases = (
        [case for case in all_cases if case.case_id in SMOKE_CASES]
        if args.suite == "smoke"
        else all_cases
    )
    beliefs = synthetic_beliefs()

    print(
        "AISHA memory reranker comparison\n"
        f"Suite: {args.suite} · Cases: {len(cases)} · "
        f"Embedding: {profile.memory.embedding_model}\n"
        f"Models: {', '.join(args.models)}"
    )

    comparisons = []
    for model in args.models:
        print(f"\nRunning {model}...")
        comparisons.append(
            await evaluate_model(
                model,
                profile=profile,
                cases=cases,
                beliefs=beliefs,
            )
        )

    print("\nCOMPARISON")
    for result in comparisons:
        print_model_result(result)

    payload = {
        "profile": args.profile,
        "suite": args.suite,
        "embedding_model": profile.memory.embedding_model,
        "models": comparisons,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
