from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aisha.memory.benchmark import benchmark_cases, run_case, summarize_results, synthetic_beliefs
from aisha.memory.relevance import OllamaMemoryRelevanceGate
from aisha.memory.reranker_benchmark import (
    SMOKE_RERANKER_CASE_IDS,
    reranker_benchmark_cases,
    run_reranker_case,
    summarize_reranker_results,
)
from aisha.memory.semantic import OllamaSemanticMemoryRetriever
from aisha.settings import Settings

SMOKE_RETRIEVAL_CASES = {
    "patient_contact_paraphrase",
    "multi_work_and_volunteer",
    "negative_medical_school_general",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AISHA memory relevance-gate models on fixed benchmark suites."
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
        choices=("smoke", "full", "reranker-smoke", "reranker-hard"),
        default="reranker-smoke",
        help=(
            "Retrieval-pipeline smoke/full suites, or fixed-candidate reranker suites. "
            "Use reranker-hard for the 40-case model-selection benchmark."
        ),
    )
    parser.add_argument(
        "--profile",
        default="mac-m2max-96gb",
        help="AISHA runtime profile supplying Ollama settings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file for comparison results.",
    )
    return parser.parse_args()


def build_gate(model: str, profile) -> OllamaMemoryRelevanceGate:
    base_url = profile.memory.semantic_relevance_base_url or profile.llm.base_url
    keep_alive = (
        profile.memory.semantic_relevance_keep_alive
        if profile.memory.semantic_relevance_keep_alive is not None
        else profile.llm.keep_alive
    )
    return OllamaMemoryRelevanceGate(
        model=model,
        base_url=base_url,
        keep_alive=keep_alive,
    )


async def evaluate_retrieval_model(
    model: str,
    *,
    profile,
    cases,
    beliefs: list[dict],
) -> dict:
    gate = build_gate(model, profile)
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

    results = [await run_case(case, beliefs, retriever) for case in cases]
    return {
        "model": model,
        "benchmark": "retrieval_pipeline",
        "summary": summarize_results(results),
        "results": results,
    }


async def evaluate_reranker_model(
    model: str,
    *,
    profile,
    cases,
    beliefs: list[dict],
) -> dict:
    gate = build_gate(model, profile)
    results = [await run_reranker_case(case, beliefs, gate) for case in cases]
    return {
        "model": model,
        "benchmark": "fixed_candidate_reranker",
        "summary": summarize_reranker_results(results),
        "results": results,
    }


def print_retrieval_result(result: dict) -> None:
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


def print_reranker_result(result: dict) -> None:
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
        f"  over-retrieval: {summary['over_retrieval_rate']:.1%} · "
        f"clean queries: {summary['clean_queries']}/{summary['cases']} "
        f"({summary['clean_query_rate']:.1%})"
    )
    print(
        f"  median gate: {summary['latency_ms']['median_gate']:.0f} ms · "
        f"max gate: {summary['latency_ms']['max_gate']:.0f} ms"
    )

    print("  categories:")
    for category, row in summary["category_summary"].items():
        print(
            f"    {category}: {row['exact_cases']}/{row['cases']} exact · "
            f"P {row['precision']:.1%} · R {row['recall']:.1%} · "
            f"FP {row['false_positive']} · FN {row['false_negative']}"
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
        for candidate in row["candidate_decisions"]:
            marker = "*" if candidate["selected"] else "-"
            print(
                f"      {marker} {candidate['topic_key']}: "
                f"{candidate['decision']} · {candidate['reason']}"
            )


async def main() -> None:
    args = parse_args()
    settings = Settings(aisha_profile=args.profile)
    profile = settings.load_profile()
    if not profile.llm.base_url:
        raise SystemExit(f"Profile {args.profile!r} does not define an Ollama URL.")

    beliefs = synthetic_beliefs()
    reranker_mode = args.suite.startswith("reranker-")

    if reranker_mode:
        all_cases = reranker_benchmark_cases()
        cases = (
            [case for case in all_cases if case.case_id in SMOKE_RERANKER_CASE_IDS]
            if args.suite == "reranker-smoke"
            else all_cases
        )
    else:
        if not profile.memory.embedding_model:
            raise SystemExit(f"Profile {args.profile!r} does not define an embedding model.")
        all_cases = benchmark_cases()
        cases = (
            [case for case in all_cases if case.case_id in SMOKE_RETRIEVAL_CASES]
            if args.suite == "smoke"
            else all_cases
        )

    print(
        "AISHA memory reranker comparison\n"
        f"Suite: {args.suite} · Cases: {len(cases)}\n"
        f"Models: {', '.join(args.models)}"
    )

    comparisons = []
    for model in args.models:
        print(f"\nRunning {model}...")
        if reranker_mode:
            result = await evaluate_reranker_model(
                model,
                profile=profile,
                cases=cases,
                beliefs=beliefs,
            )
        else:
            result = await evaluate_retrieval_model(
                model,
                profile=profile,
                cases=cases,
                beliefs=beliefs,
            )
        comparisons.append(result)

    print("\nCOMPARISON")
    for result in comparisons:
        if result["benchmark"] == "fixed_candidate_reranker":
            print_reranker_result(result)
        else:
            print_retrieval_result(result)

    payload = {
        "profile": args.profile,
        "suite": args.suite,
        "models": comparisons,
    }
    if not reranker_mode:
        payload["embedding_model"] = profile.memory.embedding_model

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
