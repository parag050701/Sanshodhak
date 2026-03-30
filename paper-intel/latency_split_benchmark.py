#!/usr/bin/env python3
"""
Module 4: Retrieval vs End-to-End Latency Benchmark
Produces explicit latency split artifacts for journal-safe reporting.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

from eval_compare import RAGWrapper
from ollama_rag import OllamaRAG
from enhanced_rag import EnhancedRAG
from graph_rag import GraphRAG


def percentile_95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = 0.95 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * frac)


def aggregate(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean_s": 0.0, "median_s": 0.0, "p95_s": 0.0}
    return {
        "mean_s": float(mean(values)),
        "median_s": float(median(values)),
        "p95_s": percentile_95(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency split benchmark")
    parser.add_argument("--index", default="rag_index", help="Index directory")
    parser.add_argument("--questions", default="rag_test_questions.json", help="Questions JSON")
    parser.add_argument("--sample", type=int, default=5, help="Number of questions to benchmark")
    parser.add_argument("--threshold", type=float, default=0.10, help="Graph similarity threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).parent
    index_dir = str(base_dir / args.index)
    questions_path = base_dir / args.questions
    out_dir = base_dir / "journal_claim_sync"
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if args.sample:
        questions = questions[: args.sample]

    print(f"Loaded {len(questions)} questions for latency benchmark")

    systems: List[RAGWrapper] = []
    embed_model = "bge-m3"
    llm_model = "deepseek-r1:7b"

    print("[1/4] Loading VR-D (Dense)")
    vrd = OllamaRAG(embed_model=embed_model, llm_model=llm_model, use_openrouter=False)
    vrd.load(index_dir)
    systems.append(RAGWrapper("VR-D (Dense)", vrd))

    print("[2/4] Loading VR-H (Hybrid)")
    vrh = EnhancedRAG(use_ollama_fallback=True, ollama_embed=embed_model, ollama_llm=llm_model)
    vrh.load(index_dir)
    systems.append(RAGWrapper("VR-H (Hybrid)", vrh))

    print("[3/4] Loading GR (Graph)")
    gr = GraphRAG(use_bm25=False, ollama_embed=embed_model, ollama_llm=llm_model, similarity_threshold=args.threshold)
    gr.load(index_dir)
    systems.append(RAGWrapper("GR (Graph)", gr))

    print("[4/4] Loading HGR (Hybrid+Graph)")
    hgr = GraphRAG(use_bm25=True, ollama_embed=embed_model, ollama_llm=llm_model, similarity_threshold=args.threshold)
    hgr.load(index_dir)
    systems.append(RAGWrapper("HGR (Hybrid+Graph)", hgr))

    results: Dict[str, Dict] = {}

    for wrapper in systems:
        retrieval_latencies: List[float] = []
        end_to_end_latencies: List[float] = []
        generation_estimates: List[float] = []

        print(f"\nBenchmarking: {wrapper.name}")
        for idx, item in enumerate(questions, start=1):
            query = item["question"]

            t0 = time.time()
            _ = wrapper.retrieve(query, top_k=10)
            retrieval_latency = time.time() - t0

            _, end_to_end_latency = wrapper.generate(query, top_k=5)
            generation_only_estimate = max(0.0, end_to_end_latency - retrieval_latency)

            retrieval_latencies.append(retrieval_latency)
            end_to_end_latencies.append(end_to_end_latency)
            generation_estimates.append(generation_only_estimate)

            print(
                f"  [{idx}/{len(questions)}] "
                f"retr={retrieval_latency:.3f}s e2e={end_to_end_latency:.3f}s"
            )

        results[wrapper.name] = {
            "retrieval_only": aggregate(retrieval_latencies),
            "end_to_end": aggregate(end_to_end_latencies),
            "generation_estimate": aggregate(generation_estimates),
            "n_questions": len(questions),
        }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "notes": [
            "end_to_end is measured from full query() runtime",
            "generation_estimate = max(0, end_to_end - retrieval_only)",
            "generation_estimate is approximate because retrieval and query runs are separate invocations",
        ],
        "meta": {
            "index_dir": str((base_dir / args.index).relative_to(base_dir.parent)),
            "questions_file": str((base_dir / args.questions).relative_to(base_dir.parent)),
            "sample": len(questions),
            "graph_similarity_threshold": args.threshold,
        },
        "systems": results,
    }

    json_path = out_dir / "latency_split_report.json"
    md_path = out_dir / "latency_split_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Latency Split Benchmark",
        "",
        f"- Generated UTC: {report['generated_utc']}",
        f"- Questions benchmarked: {len(questions)}",
        f"- Index: `{report['meta']['index_dir']}`",
        f"- Questions file: `{report['meta']['questions_file']}`",
        "",
        "| System | Retrieval Mean (s) | E2E Mean (s) | Gen Est Mean (s) | Retrieval P95 (s) | E2E P95 (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for system_name, payload in results.items():
        retr = payload["retrieval_only"]
        e2e = payload["end_to_end"]
        gen = payload["generation_estimate"]
        lines.append(
            "| "
            f"{system_name} | {retr['mean_s']:.3f} | {e2e['mean_s']:.3f} | {gen['mean_s']:.3f} | "
            f"{retr['p95_s']:.3f} | {e2e['p95_s']:.3f} |"
        )

    lines += [
        "",
        "## Notes",
        "- End-to-end latency is measured directly from full `query()` execution.",
        "- Generation-only latency is estimated from separate runs and should be treated as approximate.",
    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n[OK] Latency split report written:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
