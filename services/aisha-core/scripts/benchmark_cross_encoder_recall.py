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
from aisha.memory.cross_encoder_relevance import CrossEncoderMemoryRelevanceGate
from aisha.memory.semantic import OllamaSemanticMemoryRetriever
from aisha.settings import Settings

SMOKE_CASES = {
    "patient_contact_paraphrase",
    "multi_work_and_volunteer",
    "negative_medical_school_general",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a dedicated cross-encoder memory relevance gate."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Sentence Transformers compatible cross-encoder model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for accepting a candidate memory.",
    )
    parser.add_argument(
        "--suite",
        choices=("smoke", "full"),
        default="smoke",
        help="Three hard cases or the complete 20-case synthetic suite.",
    )
    parser.add_argument(
        "--profile",
        default="mac-m2max-96gb",
        help="AISHA profile supplying embedding/retrieval configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    profile = Settings(aisha_profile=args.profile).load_profile()
    if not profile.llm.base_url or not profile.memory.embedding_model:
        raise SystemExit("Selected profile does not support the recall benchmark.")

    gate = CrossEncoderMemoryRelevanceGate(
        model=args.model,
        threshold=args.threshold,
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

    all_cases = benchmark_cases()
    cases = (
        [case for case in all_cases if case.case_id in SMOKE_CASES]
        if args.suite == "smoke"
        else all_cases
    )
    beliefs = synthetic_beliefs()

    print(
        "AISHA dedicated reranker benchmark\n"
        f"Reranker: {args.model}\n"
        f"Threshold: {args.threshold:.3f}\n"
        f"Suite: {args.suite} · Cases: {len(cases)} · "
        f"Embedding: {profile.memory.embedding_model}\n"
    )

    results: list[dict] = []
    for index, case in enumerate(cases, start=1):
        result = await run_case(case, beliefs, retriever)
        results.append(result)
        marker = "PASS" if result["exact"] else "FAIL"
        print(
            f"[{index:02d}/{len(cases):02d}] {marker} {case.case_id} "
            f"· {result['latency_ms']['total']:.0f} ms"
        )
        if not result["exact"]:
            expected = ", ".join(result["expected_topics"]) or "none"
            actual = ", ".join(result["actual_topics"]) or "none"
            print(f"  expected: {expected}")
            print(f"  actual:   {actual}")
        for candidate in result["semantic_candidates"]:
            decision = candidate.get("decision")
            if decision not in {"reranker_accept", "reranker_reject"}:
                continue
            print(
                f"  {candidate.get('topic_key')}: {decision} "
                f"· {candidate.get('reason', '')}"
            )
        metrics = result.get("relevance_gate", {}).get("last_metrics", {})
        if metrics:
            print(
                f"  gate: {metrics.get('total_ms', 0):.0f} ms · "
                f"{metrics.get('pairs', 0)} pairs · "
                f"device={metrics.get('device')}"
            )

    summary = summarize_results(results)
    print("\nSUMMARY")
    print(
        f"Exact cases: {summary['exact_cases']} / {summary['cases']} "
        f"({summary['exact_case_accuracy']:.1%})"
    )
    print(
        f"Precision: {summary['precision']:.1%} · "
        f"Recall: {summary['recall']:.1%}"
    )
    print(
        f"Negative controls: {summary['negative_controls_passed']} / "
        f"{summary['negative_controls']} "
        f"({summary['negative_control_accuracy']:.1%})"
    )
    print(
        f"Median retrieval: {summary['latency_ms']['median_total']:.0f} ms · "
        f"Median gate: {summary['latency_ms']['median_gate']:.0f} ms · "
        f"Max retrieval: {summary['latency_ms']['max_total']:.0f} ms"
    )

    payload = {
        "profile": args.profile,
        "suite": args.suite,
        "model": args.model,
        "threshold": args.threshold,
        "summary": summary,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
