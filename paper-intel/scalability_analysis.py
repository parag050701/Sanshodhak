#!/usr/bin/env python3
"""
Scalability Analysis — Part 5.

Measures how index build time, graph build time, retrieval latency, and
memory footprint scale with corpus size.

Synthetic corpus sizes tested:   1 000  |  5 000  |  10 000  |  50 000

Measurements
------------
    index_build_time  : FAISS index construction (seconds)
    bm25_build_time   : BM25 inverted index construction (seconds)
    graph_build_time  : GraphIndex construction (seconds)
    retrieval_latency : Mean query latency over 20 queries (milliseconds)
    memory_mb         : Peak RSS after index build (MB, via psutil)

Output
------
    scalability_results/scalability_results.json   — raw numbers
    scalability_results/plots/*.png                — one plot per metric

Usage
-----
    python scalability_analysis.py
    python scalability_analysis.py --sizes 1000 5000 10000
    python scalability_analysis.py --no-graph      # skip graph (slow for 50k)
    python scalability_analysis.py --output-dir my_results
"""

import argparse
import json
import logging
import os
import random
import string
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.retrieval import EmbeddingBackend, BM25Retriever, DenseRetriever, HybridRetriever
from core.graph import GraphIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scalability")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic corpus generator
# ─────────────────────────────────────────────────────────────────────────────

_VOCAB = (
    "transformer attention retrieval dense sparse hybrid bm25 faiss embedding "
    "neural network language model document query passage relevance ranking "
    "information retrieval knowledge graph node edge pagerank jaccard similarity "
    "cosine vector index inverted term frequency inverse document citation "
    "paper abstract title author journal conference dataset benchmark evaluation "
    "precision recall ndcg map mrr score baseline experiment result method "
    "approach technique system performance latency throughput memory scalable"
).split()


def _random_sentence(rng: random.Random, length: int = 12) -> str:
    return " ".join(rng.choices(_VOCAB, k=length))


def generate_corpus(n: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Generate a synthetic corpus of n documents.

    Returns
    -------
    doc_ids : list of unique string IDs
    texts   : list of ~200-word synthetic documents
    """
    rng = random.Random(seed)
    doc_ids: List[str] = [f"doc_{i:06d}" for i in range(n)]
    texts: List[str] = []
    for _ in range(n):
        # Mix topic-specific sentences with shared vocabulary (mimics academic papers)
        topic = rng.choices(_VOCAB[:20], k=3)
        sentences = [" ".join(topic)]
        sentences += [_random_sentence(rng, rng.randint(8, 18)) for _ in range(15)]
        texts.append(". ".join(sentences) + ".")
    return doc_ids, texts


def generate_queries(n: int = 20, seed: int = 99) -> List[str]:
    """Generate n synthetic queries."""
    rng = random.Random(seed)
    return [
        " ".join(rng.choices(_VOCAB[:30], k=rng.randint(3, 7)))
        for _ in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    """Current process RSS memory in MB (requires psutil)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        return float("nan")


def measure_index_build(
    doc_ids: List[str],
    texts: List[str],
    embed_backend: EmbeddingBackend,
    include_graph: bool = True,
) -> Dict[str, float]:
    """
    Build BM25, FAISS, and optionally Graph indices; return timing + memory.
    """
    results: Dict[str, float] = {}

    # ── BM25 ──────────────────────────────────────────────────────────────
    mem_before = _rss_mb()
    t0 = time.perf_counter()
    bm25 = BM25Retriever().build(doc_ids, texts)
    results["bm25_build_s"] = time.perf_counter() - t0
    results["bm25_memory_mb"] = _rss_mb() - mem_before

    # ── Dense (FAISS) ─────────────────────────────────────────────────────
    mem_before = _rss_mb()
    t0 = time.perf_counter()
    dense = DenseRetriever(embed_backend).build(doc_ids, texts, batch_size=64)
    results["faiss_build_s"] = time.perf_counter() - t0
    results["faiss_memory_mb"] = _rss_mb() - mem_before

    # ── Graph ─────────────────────────────────────────────────────────────
    if include_graph:
        truncated = [t[:2048] for t in texts]
        embeddings = embed_backend.encode(truncated, batch_size=64)
        mem_before = _rss_mb()
        t0 = time.perf_counter()
        metadata = [{"title": t[:60], "doi": ""} for t in texts]
        graph = GraphIndex(
            jaccard_threshold=0.10,
            embed_threshold=0.80,  # stricter to keep graph sparse for large corpora
        ).build(doc_ids, texts, embeddings, metadata)
        results["graph_build_s"] = time.perf_counter() - t0
        results["graph_memory_mb"] = _rss_mb() - mem_before
        results["graph_nodes"] = graph.graph.number_of_nodes()
        results["graph_edges"] = graph.graph.number_of_edges()
    else:
        for k in ["graph_build_s", "graph_memory_mb", "graph_nodes", "graph_edges"]:
            results[k] = float("nan")
        dense_obj = dense  # reference for retrieval latency measurement
        bm25_obj = bm25

    return results, dense, bm25


def measure_retrieval_latency(
    queries: List[str],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    top_k: int = 20,
) -> Dict[str, float]:
    """
    Measure mean + P95 retrieval latency for Dense and Hybrid.
    """
    dense_times: List[float] = []
    hybrid_times: List[float] = []

    from core.fusion import reciprocal_rank_fusion

    for q in queries:
        # Dense
        t0 = time.perf_counter()
        dense.retrieve(q, top_k=top_k)
        dense_times.append((time.perf_counter() - t0) * 1000)

        # Hybrid
        t0 = time.perf_counter()
        d_res = dense.retrieve(q, top_k=top_k * 2)
        b_res = bm25.retrieve(q, top_k=top_k * 2)
        reciprocal_rank_fusion([d_res, b_res])
        hybrid_times.append((time.perf_counter() - t0) * 1000)

    def _stats(times: List[float]) -> Dict[str, float]:
        arr = sorted(times)
        n = len(arr)
        return {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(arr[int(0.95 * n)] if n > 0 else 0),
        }

    return {
        "dense": _stats(dense_times),
        "hybrid": _stats(hybrid_times),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: Dict[int, Dict], output_dir: Path) -> None:
    """Generate one plot per metric, saved as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sizes = sorted(results.keys())

    # Helper
    def _get(size: int, *keys: str) -> Optional[float]:
        d = results[size]
        for k in keys:
            if k in d:
                v = d[k]
                return v if (v is not None and not (isinstance(v, float) and np.isnan(v))) else None
        return None

    # ── Plot 1: Index build time vs corpus size ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, key in [
        ("BM25 build", "bm25_build_s"),
        ("FAISS build", "faiss_build_s"),
        ("Graph build", "graph_build_s"),
    ]:
        vals = [_get(s, key) for s in sizes]
        valid = [(s, v) for s, v in zip(sizes, vals) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Corpus size (documents)")
    ax.set_ylabel("Build time (seconds)")
    ax.set_title("Index Build Time vs Corpus Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "build_time.png", dpi=150)
    plt.close(fig)
    logger.info("Saved build_time.png")

    # ── Plot 2: Memory footprint vs corpus size ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, key in [
        ("BM25 memory", "bm25_memory_mb"),
        ("FAISS memory", "faiss_memory_mb"),
        ("Graph memory", "graph_memory_mb"),
    ]:
        vals = [_get(s, key) for s in sizes]
        valid = [(s, v) for s, v in zip(sizes, vals) if v is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker="s", label=label)
    ax.set_xlabel("Corpus size (documents)")
    ax.set_ylabel("Memory increase (MB)")
    ax.set_title("Memory Footprint vs Corpus Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "memory.png", dpi=150)
    plt.close(fig)
    logger.info("Saved memory.png")

    # ── Plot 3: Retrieval latency vs corpus size ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for sys_label, key_prefix in [("Dense", "dense"), ("Hybrid", "hybrid")]:
        mean_vals = [
            _get(s, f"latency_{key_prefix}_mean_ms") for s in sizes
        ]
        p95_vals = [
            _get(s, f"latency_{key_prefix}_p95_ms") for s in sizes
        ]
        valid_mean = [(s, v) for s, v in zip(sizes, mean_vals) if v is not None]
        valid_p95 = [(s, v) for s, v in zip(sizes, p95_vals) if v is not None]
        if valid_mean:
            xs, ys = zip(*valid_mean)
            ax.plot(xs, ys, marker="o", label=f"{sys_label} mean")
        if valid_p95:
            xs, ys = zip(*valid_p95)
            ax.plot(xs, ys, marker="^", linestyle="--", label=f"{sys_label} P95", alpha=0.7)
    ax.set_xlabel("Corpus size (documents)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Retrieval Latency vs Corpus Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "latency.png", dpi=150)
    plt.close(fig)
    logger.info("Saved latency.png")

    # ── Plot 4: Graph edges vs corpus size ────────────────────────────────
    vals = [(s, _get(s, "graph_edges")) for s in sizes]
    valid = [(s, v) for s, v in vals if v is not None]
    if valid:
        fig, ax = plt.subplots(figsize=(8, 5))
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker="D", color="purple", label="Graph edges")
        ax.set_xlabel("Corpus size (documents)")
        ax.set_ylabel("Number of edges")
        ax.set_title("Graph Edge Count vs Corpus Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "graph_edges.png", dpi=150)
        plt.close(fig)
        logger.info("Saved graph_edges.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Scalability analysis for Sanshodhak IR system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[1000, 5000, 10000, 50000],
        help="Corpus sizes to benchmark",
    )
    parser.add_argument(
        "--no-graph", dest="include_graph", action="store_false",
        help="Skip graph construction",
    )
    parser.add_argument(
        "--graph-max", type=int, default=15000,
        help="Maximum corpus size for graph construction",
    )
    parser.add_argument(
        "--n-queries", type=int, default=20,
        help="Number of synthetic queries for latency measurement",
    )
    parser.add_argument(
        "--embed-model", default="BAAI/bge-m3",
        help="Embedding model",
    )
    parser.add_argument(
        "--output-dir", default="scalability_results",
        help="Output directory for results and plots",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising embedding backend...")
    embed_backend = EmbeddingBackend(st_model=args.embed_model)

    queries = generate_queries(args.n_queries)
    all_results: Dict[int, Dict] = {}

    for size in sorted(args.sizes):
        logger.info("\n%s", "─" * 55)
        logger.info("Corpus size: %d documents", size)
        logger.info("%s", "─" * 55)

        doc_ids, texts = generate_corpus(size)
        include_graph = args.include_graph and size <= args.graph_max

        build_metrics, dense, bm25 = measure_index_build(
            doc_ids, texts, embed_backend, include_graph=include_graph
        )

        logger.info(
            "  BM25 build:  %.2fs | FAISS build: %.2fs | Graph build: %s",
            build_metrics.get("bm25_build_s", 0),
            build_metrics.get("faiss_build_s", 0),
            f"{build_metrics.get('graph_build_s', 0):.2f}s"
            if include_graph else "skipped",
        )

        lat = measure_retrieval_latency(queries, dense, bm25)
        logger.info(
            "  Dense latency:  mean=%.1fms  P95=%.1fms",
            lat["dense"]["mean_ms"], lat["dense"]["p95_ms"],
        )
        logger.info(
            "  Hybrid latency: mean=%.1fms  P95=%.1fms",
            lat["hybrid"]["mean_ms"], lat["hybrid"]["p95_ms"],
        )

        row = {
            **build_metrics,
            "latency_dense_mean_ms": lat["dense"]["mean_ms"],
            "latency_dense_p95_ms": lat["dense"]["p95_ms"],
            "latency_hybrid_mean_ms": lat["hybrid"]["mean_ms"],
            "latency_hybrid_p95_ms": lat["hybrid"]["p95_ms"],
        }
        all_results[size] = row

    # ── Save results ───────────────────────────────────────────────────────
    json_out = output_dir / "scalability_results.json"
    with open(json_out, "w") as fh:
        json.dump({str(k): v for k, v in all_results.items()}, fh, indent=2)
    logger.info("\nSaved → %s", json_out)

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_results(all_results, output_dir)

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Size':>8}  {'BM25(s)':>8}  {'FAISS(s)':>9}  {'Graph(s)':>9}  {'Dense(ms)':>10}  {'Hybrid(ms)':>11}")
    print("-" * 70)
    for size in sorted(all_results):
        r = all_results[size]
        graph_s = r.get("graph_build_s", float("nan"))
        print(
            f"{size:>8}  "
            f"{r.get('bm25_build_s', 0):>8.2f}  "
            f"{r.get('faiss_build_s', 0):>9.2f}  "
            f"{graph_s:>9.2f}  "
            f"{r.get('latency_dense_mean_ms', 0):>10.1f}  "
            f"{r.get('latency_hybrid_mean_ms', 0):>11.1f}"
        )
    print("=" * 70)
    print(f"\nPlots saved to: {output_dir}/plots/")


if __name__ == "__main__":
    main()
