from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aisha.memory.benchmark import benchmark_cases, run_case, summarize_results, synthetic_beliefs
from aisha.memory.mlx_reranker import MLXQwen3MemoryReranker
from aisha.memory.semantic import OllamaSemanticMemoryRetriever
from aisha.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AISHA memory recall with the dedicated MLX Qwen3 reranker."
    )
    parser.add_argument(
        "--profile",
        default="mac-m2max-96gb",
        help="AISHA runtime profile supplying embedding settings.",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3-Reranker-0.6B-4bit",
        help="MLX Qwen3 reranker model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for relevance.",
    )
    parser.add_argument(
        "--suite",
        choices=("smoke", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = Settings(aisha_profile=args.profile)
    profile = settings.load_profile()
    if not profile.llm.base_url:
        raise SystemExit(f"Profile {args.profile!r} does not define an Ollama URL.")
    if not profile.memory.embedding_model:
        raise SystemExit(f"Profile {args.profile!r} does not define an embedding model.")

    smoke_ids = {
        "patient_contact_paraphrase",
        "multi_work_and_volunteer",
        "negative_medical_school_general",
    }
    cases = benchmark_cases()
    if args.suite == "smoke":
        cases = [case for case in cases if case.case_id in smoke_ids]

    gate = MLXQwen3MemoryReranker(
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

    beliefs = synthetic_beliefs()
    results: list[dict] = []

    print(
        "AISHA dedicated MLX memory reranker benchmark\n"
        f"Suite: {args.suite} · Cases: {len(cases)} · "
        f"Embedding: {profile.memory.embedding_model}\n"
        f"Reranker: {args.model} · threshold={args.threshold:.2f}\n"
    )

    for index, case in enumerate(cases, start=1):
        result = await run_case(case, beliefs, retriever)
        results.append(result)
        marker = "PASS" if result["exact"] else "FAIL"
        expected = ", ".join(result["expected_topics"]) or "none"
        actual = ", ".join(result["actual_topics"]) or "none"
        print(
            f"[{index:02d}/{len(cases):02d}] {marker} {case.case_id} "
            f"· {result['latency_ms']['total']:.0f} ms"
        )
        if not result["exact"]:
            print(f"  expected: {expected}")
            print(f"  actual:   {actual}")
        for candidate in result["semantic_candidates"]:
            decision = candidate.get("decision")
            if decision not in {"reranker_accept", "reranker_reject"}:
                continue
            print(
                f"  {candidate.get('topic_key')}: {decision} · "
                f"{candidate.get('reason', '')}"
            )
        metrics = result.get("relevance_gate", {}).get("last_metrics", {})
        if metrics:
            print(
                "  gate: "
                f"{metrics.get('total_ms', 0):.0f} ms · "
                f"load={metrics.get('load_ms', 0):.0f} ms · "
                f"score={metrics.get('score_ms', 0):.0f} ms · "
                f"pairs={metrics.get('pairs', 0)}"
            )

    summary = summarize_results(results)
    print("\nSUMMARY")
    print(
        f"Exact: {summary['exact_cases']}/{summary['cases']} "
        f"({summary['exact_case_accuracy']:.1%})"
    )
    print(
        f"Precision: {summary['precision']:.1%} · "
        f"Recall: {summary['recall']:.1%} · "
        f"Negatives: {summary['negative_control_accuracy']:.1%}"
    )
    print(
        f"Median retrieval: {summary['latency_ms']['median_total']:.0f} ms · "
        f"Median gate: {summary['latency_ms']['median_gate']:.0f} ms · "
        f"Max gate: {summary['latency_ms']['max_gate']:.0f} ms"
    )

    payload = {
        "profile": args.profile,
        "suite": args.suite,
        "embedding_model": profile.memory.embedding_model,
        "reranker_model": args.model,
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
