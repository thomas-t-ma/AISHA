from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from aisha.memory.benchmark import benchmark_cases, run_case, summarize_results, synthetic_beliefs
from aisha.memory.relevance import OllamaMemoryRelevanceGate
from aisha.memory.semantic import OllamaSemanticMemoryRetriever
from aisha.settings import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AISHA's synthetic multi-memory retrieval benchmark."
    )
    parser.add_argument(
        "--profile",
        default="mac-m2max-96gb",
        help="AISHA runtime profile to use for the benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file for full per-case results.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named case ID. May be repeated.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = Settings(aisha_profile=args.profile)
    profile = settings.load_profile()

    if not profile.memory.semantic_recall or not profile.memory.embedding_model:
        raise SystemExit(f"Profile {args.profile!r} does not enable semantic recall.")
    if not profile.llm.base_url:
        raise SystemExit(f"Profile {args.profile!r} does not define an Ollama base URL.")

    gate = None
    if profile.memory.semantic_relevance_gate:
        gate = OllamaMemoryRelevanceGate(
            model=profile.llm.model,
            base_url=profile.llm.base_url,
            keep_alive=profile.llm.keep_alive,
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

    selected_cases = benchmark_cases()
    if args.case:
        wanted = set(args.case)
        selected_cases = [case for case in selected_cases if case.case_id in wanted]
        missing = wanted - {case.case_id for case in selected_cases}
        if missing:
            raise SystemExit("Unknown benchmark case(s): " + ", ".join(sorted(missing)))

    beliefs = synthetic_beliefs()
    results: list[dict] = []

    print(
        f"AISHA semantic recall benchmark\n"
        f"Embedding: {profile.memory.embedding_model}\n"
        f"Reranker: {profile.llm.model if gate is not None else 'disabled'}\n"
        f"Cases: {len(selected_cases)} · Synthetic memories: {len(beliefs)}\n"
    )

    for index, case in enumerate(selected_cases, start=1):
        result = await run_case(case, beliefs, retriever)
        results.append(result)
        marker = "PASS" if result["exact"] else "FAIL"
        expected = ", ".join(result["expected_topics"]) or "none"
        actual = ", ".join(result["actual_topics"]) or "none"
        total_ms = result["latency_ms"]["total"]
        print(
            f"[{index:02d}/{len(selected_cases):02d}] {marker} {case.case_id} "
            f"({case.category}) · {total_ms:.0f} ms"
        )
        if not result["exact"]:
            print(f"  expected: {expected}")
            print(f"  actual:   {actual}")
        for candidate in result["semantic_candidates"]:
            sources = "+".join(candidate.get("candidate_sources") or [])
            lexical_score = candidate.get("lexical_score")
            semantic_score = candidate.get("semantic_score")
            score_parts = []
            if lexical_score is not None:
                score_parts.append(f"lex={lexical_score}")
            if semantic_score is not None:
                score_parts.append(f"sem={semantic_score}")
            print(
                "  candidate: "
                f"{candidate.get('topic_key')} "
                f"[{sources or 'semantic'}] "
                f"{' '.join(score_parts)} "
                f"{candidate.get('decision')}"
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
        f"Median semantic: {summary['latency_ms']['median_semantic']:.0f} ms · "
        f"Max retrieval: {summary['latency_ms']['max_total']:.0f} ms"
    )
    print("Methods:", json.dumps(summary["method_counts"], sort_keys=True))
    print("Candidate decisions:", json.dumps(summary["candidate_decisions"], sort_keys=True))

    payload = {
        "profile": args.profile,
        "embedding_model": profile.memory.embedding_model,
        "reranker_model": profile.llm.model if gate is not None else None,
        "beliefs": beliefs,
        "summary": summary,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
