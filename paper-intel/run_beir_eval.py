#!/usr/bin/env python3
"""
BEIR Benchmark Evaluation Pipeline (Part 1).

Downloads, indexes, and evaluates five BEIR datasets:
    scifact   — scientific fact-checking  (~5k docs, 300 queries)
    nfcorpus  — medical retrieval          (~3.6k docs, 323 queries)
    fiqa      — financial QA               (~57k docs, 648 queries)
    arguana   — argument retrieval         (~8.7k docs, 1406 queries)
    trec-covid — biomedical COVID-19       (~171k docs, 50 queries)

Systems evaluated
-----------------
    Dense           : FAISS cosine similarity only
    Hybrid          : Dense + BM25 fused via RRF
    AHR             : Hybrid + Graph expansion (Adaptive slot budget)
    AHR+Reranker    : AHR then cross-encoder reranking of top-50

Metrics reported
----------------
    NDCG@10, MAP, MRR, Recall@100 (per dataset and per system)

Statistical tests
-----------------
    Wilcoxon signed-rank test on NDCG@10 per-query scores:
        Dense vs Hybrid, Dense vs AHR, Hybrid vs AHR, AHR vs AHR+Reranker

Output
------
    beir_results/summary.json   — structured metrics + significance tests
    beir_results/summary.csv    — table for LaTeX / spreadsheet import
    beir_results/<ds>_run_<sys>.json — raw ranked lists (TREC format)

Usage
-----
    python run_beir_eval.py                          # all 5 datasets
    python run_beir_eval.py --datasets scifact nfcorpus
    python run_beir_eval.py --datasets scifact --skip-graph
    python run_beir_eval.py --datasets scifact --reranker
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.retrieval import EmbeddingBackend, BM25Retriever, DenseRetriever, HybridRetriever
from core.graph import GraphIndex
from core.fusion import reciprocal_rank_fusion
from core.adaptive_budget import compute_graph_budget
from core.reranker import CrossEncoderReranker
from core.evaluation import evaluate_run, wilcoxon_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("beir_eval")

# ─────────────────────────────────────────────────────────────────────────────
# BEIR dataset URLs and names
# ─────────────────────────────────────────────────────────────────────────────

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
BEIR_DATASETS = {
    "scifact":   "scifact.zip",
    "nfcorpus":  "nfcorpus.zip",
    "fiqa":      "fiqa.zip",
    "arguana":   "arguana.zip",
    "trec-covid": "trec-covid.zip",
}

# Skip graph construction for large corpora (O(n*k) edges — still fast,
# but graph traversal results may be diluted for very large corpora)
GRAPH_MAX_CORPUS = 30_000


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(name: str, data_dir: str) -> Path:
    """
    Download and unzip a BEIR dataset if not already present.

    Returns the path to the dataset directory.
    """
    try:
        from beir import util as beir_util

        out = Path(data_dir) / name
        if out.exists():
            logger.info("Dataset %r already present at %s", name, out)
            return out

        url = f"{BEIR_BASE_URL}/{BEIR_DATASETS[name]}"
        logger.info("Downloading %r from %s ...", name, url)
        beir_util.download_and_unzip(url, str(data_dir))
        return out
    except ImportError:
        logger.error(
            "The 'beir' package is required: pip install beir\n"
            "Alternatively, manually place the dataset at %s/%s/",
            data_dir, name,
        )
        raise


def load_dataset(
    name: str, data_dir: str
) -> Tuple[Dict[str, Dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Load corpus, queries, and qrels for a BEIR dataset.

    Returns
    -------
    corpus  : {doc_id: {"title": str, "text": str}}
    queries : {query_id: query_text}
    qrels   : {query_id: {doc_id: relevance_score}}
    """
    try:
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        logger.error("The 'beir' package is required: pip install beir")
        raise

    dataset_path = download_dataset(name, data_dir)
    # Determine split (trec-covid uses "test"; most others have test split)
    split = "test"
    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split=split)
    logger.info(
        "Loaded %r — corpus=%d, queries=%d, qrels queries=%d",
        name, len(corpus), len(queries), len(qrels),
    )
    return corpus, queries, qrels


# ─────────────────────────────────────────────────────────────────────────────
# Index building
# ─────────────────────────────────────────────────────────────────────────────

def build_indices(
    corpus: Dict[str, Dict],
    embed_backend: EmbeddingBackend,
    skip_graph: bool = False,
    graph_max_corpus: int = GRAPH_MAX_CORPUS,
    graph_alpha: float = 0.5,
    graph_beta: float = 0.5,
    batch_size: int = 32,
) -> Tuple[DenseRetriever, BM25Retriever, Optional[GraphIndex]]:
    """
    Build FAISS, BM25, and (optionally) graph indices from a BEIR corpus.

    Parameters
    ----------
    corpus          : BEIR corpus dict {doc_id: {"title", "text"}}.
    embed_backend   : shared embedding backend.
    skip_graph      : if True, graph is not built.
    graph_max_corpus: skip graph if corpus exceeds this size.
    graph_alpha     : Jaccard weight in GraphIndex edge formula.
    graph_beta      : embedding weight in GraphIndex edge formula.
    batch_size      : embedding batch size.
    """
    doc_ids = list(corpus.keys())
    texts = [
        (corpus[did].get("title", "") + " " + corpus[did].get("text", "")).strip()
        for did in doc_ids
    ]

    # ── BM25 ──────────────────────────────────────────────────────────────
    logger.info("Building BM25 index (%d docs)...", len(doc_ids))
    t0 = time.perf_counter()
    bm25 = BM25Retriever().build(doc_ids, texts)
    logger.info("BM25 built in %.1fs", time.perf_counter() - t0)

    # ── Dense (FAISS) ─────────────────────────────────────────────────────
    logger.info("Building FAISS index (%d docs)...", len(doc_ids))
    t0 = time.perf_counter()
    dense = DenseRetriever(embed_backend).build(doc_ids, texts, batch_size=batch_size)
    logger.info("FAISS built in %.1fs", time.perf_counter() - t0)

    # ── Graph ─────────────────────────────────────────────────────────────
    graph: Optional[GraphIndex] = None
    if not skip_graph and len(doc_ids) <= graph_max_corpus:
        logger.info("Building GraphIndex (%d docs)...", len(doc_ids))
        t0 = time.perf_counter()

        # Embed all docs for the graph (reuse dense index vectors if possible)
        embeddings = embed_backend.encode(
            [t[:2048] for t in texts], batch_size=batch_size
        )
        metadata = [
            {"title": corpus[did].get("title", ""), "doi": ""}
            for did in doc_ids
        ]
        graph = GraphIndex(alpha=graph_alpha, beta=graph_beta).build(
            doc_ids, texts, embeddings, metadata
        )
        logger.info("Graph built in %.1fs — %s", time.perf_counter() - t0, graph.get_stats())
    elif not skip_graph:
        logger.info(
            "Skipping graph for large corpus (%d docs > %d limit)",
            len(doc_ids), graph_max_corpus,
        )

    return dense, bm25, graph


# ─────────────────────────────────────────────────────────────────────────────
# System runners
# ─────────────────────────────────────────────────────────────────────────────

def run_dense(
    queries: Dict[str, str],
    dense: DenseRetriever,
    top_k: int = 100,
) -> Dict[str, List[str]]:
    """Dense-only retrieval run."""
    run: Dict[str, List[str]] = {}
    for qid, qtext in queries.items():
        results = dense.retrieve(qtext, top_k=top_k)
        run[qid] = [did for did, _ in results]
    return run


def run_hybrid(
    queries: Dict[str, str],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    top_k: int = 100,
    rrf_k: int = 60,
) -> Dict[str, List[str]]:
    """Hybrid Dense+BM25 RRF run."""
    run: Dict[str, List[str]] = {}
    fetch = max(top_k * 2, 100)
    for qid, qtext in queries.items():
        d_results = dense.retrieve(qtext, top_k=fetch)
        b_results = bm25.retrieve(qtext, top_k=fetch)
        fused = reciprocal_rank_fusion([d_results, b_results], k=rrf_k)
        run[qid] = [did for did, _ in fused[:top_k]]
    return run


def run_ahr(
    queries: Dict[str, str],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    graph: Optional[GraphIndex],
    top_k: int = 100,
    expansion_mode: str = "1hop",
    rrf_k: int = 60,
) -> Dict[str, List[str]]:
    """
    Adaptive Hybrid Retrieval (AHR): Dense + BM25 + Graph with adaptive budget.

    Falls back to pure Hybrid if graph is None.
    """
    run: Dict[str, List[str]] = {}
    fetch = max(top_k * 2, 100)
    vocab = bm25.vocabulary

    for qid, qtext in queries.items():
        d_results = dense.retrieve(qtext, top_k=fetch)
        b_results = bm25.retrieve(qtext, top_k=fetch)

        # Compute adaptive graph budget
        dense_scores = [s for _, s in d_results[:20]]
        bm25_scores = [s for _, s in b_results[:20]]
        graph_budget = compute_graph_budget(
            qtext, dense_scores, bm25_scores, top_k, vocabulary=vocab
        )

        # Base hybrid results
        fused = reciprocal_rank_fusion([d_results, b_results], k=rrf_k)
        base_ids = [did for did, _ in fused]

        if graph is None or graph_budget == 0:
            run[qid] = base_ids[:top_k]
            continue

        # Graph expansion from hybrid top results
        seed_ids = set(base_ids[: top_k - graph_budget])

        if expansion_mode == "qbpr":
            q_emb = dense.encode_query(qtext)
            graph_results = graph.expand(
                seed_ids, mode="qbpr", top_k=graph_budget, query_embedding=q_emb
            )
        else:
            graph_results = graph.expand(seed_ids, mode=expansion_mode, top_k=graph_budget)

        graph_ids = [did for did, _ in graph_results]

        # Merge: hybrid fill the first (top_k - graph_budget) slots, graph fills the rest
        seen: Set[str] = set()
        final: List[str] = []
        for did in base_ids:
            if did not in seen:
                seen.add(did)
                final.append(did)
            if len(final) == top_k - graph_budget:
                break
        for did in graph_ids:
            if did not in seen and len(final) < top_k:
                seen.add(did)
                final.append(did)
        # Fill any remaining slots from base
        for did in base_ids:
            if did not in seen and len(final) < top_k:
                seen.add(did)
                final.append(did)

        run[qid] = final[:top_k]

    return run


def run_ahr_reranked(
    queries: Dict[str, str],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    graph: Optional[GraphIndex],
    reranker: CrossEncoderReranker,
    corpus: Dict[str, Dict],
    top_k: int = 100,
    expansion_mode: str = "1hop",
) -> Dict[str, List[str]]:
    """AHR run followed by cross-encoder reranking of top-50."""
    if not reranker.available:
        logger.warning("Reranker not available; returning AHR results unchanged")
        return run_ahr(queries, dense, bm25, graph, top_k, expansion_mode)

    ahr_run = run_ahr(
        queries, dense, bm25, graph, top_k=50, expansion_mode=expansion_mode
    )
    id_to_text: Dict[str, str] = {
        did: (corpus[did].get("title", "") + " " + corpus[did].get("text", "")).strip()
        for did in corpus
    }

    run: Dict[str, List[str]] = {}
    for qid, ranked_ids in ahr_run.items():
        qtext = queries[qid]
        reranked = reranker.rerank_ids(qtext, ranked_ids, id_to_text, top_k=top_k)
        run[qid] = reranked

    return run


# ─────────────────────────────────────────────────────────────────────────────
# Statistical testing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _align_queries(
    per_query_a: List[float],
    per_query_b: List[float],
    qids_a: List[str],
    qids_b: List[str],
) -> Tuple[List[float], List[float]]:
    """Align per-query scores to the same query order (intersection)."""
    id_to_a = dict(zip(qids_a, per_query_a))
    id_to_b = dict(zip(qids_b, per_query_b))
    common = sorted(set(id_to_a) & set(id_to_b))
    return [id_to_a[q] for q in common], [id_to_b[q] for q in common]


def run_significance_tests(
    all_per_query: Dict[str, Dict[str, List[float]]],
    metric: str = "ndcg@10",
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """
    Run pairwise Wilcoxon signed-rank tests between all system pairs.

    Parameters
    ----------
    all_per_query : {system_name: {metric_name: [per_query_scores]}}
    metric        : metric to test (default ndcg@10)
    alpha         : significance threshold

    Returns
    -------
    {"{A}_vs_{B}": {"statistic", "p_value", "significant"}}
    """
    systems = list(all_per_query.keys())
    results = {}
    for i, sys_a in enumerate(systems):
        for sys_b in systems[i + 1 :]:
            scores_a = all_per_query[sys_a].get(metric, [])
            scores_b = all_per_query[sys_b].get(metric, [])
            if len(scores_a) < 2 or len(scores_b) < 2:
                continue
            n = min(len(scores_a), len(scores_b))
            try:
                stat, pval = wilcoxon_test(scores_a[:n], scores_b[:n])
                results[f"{sys_a}_vs_{sys_b}"] = {
                    "statistic": round(stat, 4),
                    "p_value": round(pval, 6),
                    "significant": pval < alpha,
                    "n_queries": n,
                }
            except Exception as exc:
                logger.warning("Wilcoxon failed for %s vs %s: %s", sys_a, sys_b, exc)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Results serialisation
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    logger.info("Saved → %s", path)


def _save_csv(rows: List[Dict], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    name: str,
    data_dir: str,
    embed_backend: EmbeddingBackend,
    reranker: Optional[CrossEncoderReranker],
    output_dir: Path,
    skip_graph: bool = False,
    expansion_mode: str = "1hop",
    top_k: int = 100,
    k_values: Optional[List[int]] = None,
) -> Tuple[Dict, Dict]:
    """
    Run full evaluation on a single BEIR dataset.

    Returns
    -------
    (mean_metrics, per_query_metrics)

    Both are dicts keyed by system name.
    """
    k_values = k_values or [10, 100]
    logger.info("=" * 60)
    logger.info("Evaluating dataset: %s", name)
    logger.info("=" * 60)

    corpus, queries, qrels = load_dataset(name, data_dir)

    dense, bm25, graph = build_indices(
        corpus, embed_backend, skip_graph=skip_graph
    )
    hybrid = HybridRetriever(dense, bm25)  # for convenience

    systems: Dict[str, Dict[str, List[str]]] = {}

    # 1. Dense
    logger.info("[1/4] Running Dense...")
    systems["Dense"] = run_dense(queries, dense, top_k=top_k)

    # 2. Hybrid
    logger.info("[2/4] Running Hybrid...")
    systems["Hybrid"] = run_hybrid(queries, dense, bm25, top_k=top_k)

    # 3. AHR
    logger.info("[3/4] Running AHR (graph=%s)...", expansion_mode)
    systems["AHR"] = run_ahr(
        queries, dense, bm25, graph, top_k=top_k, expansion_mode=expansion_mode
    )

    # 4. AHR + Reranker
    if reranker is not None and reranker.available:
        logger.info("[4/4] Running AHR+Reranker...")
        systems["AHR+Reranker"] = run_ahr_reranked(
            queries, dense, bm25, graph, reranker, corpus,
            top_k=top_k, expansion_mode=expansion_mode,
        )
    else:
        logger.info("[4/4] Reranker disabled — skipping AHR+Reranker")

    # ── Compute metrics ────────────────────────────────────────────────────
    mean_metrics: Dict[str, Dict] = {}
    per_query_metrics: Dict[str, Dict] = {}

    for sys_name, run in systems.items():
        mean, per_query = evaluate_run(run, qrels, k_values=k_values)
        mean_metrics[sys_name] = {k: round(v, 4) for k, v in mean.items()}
        per_query_metrics[sys_name] = per_query
        logger.info(
            "  %-16s  NDCG@10=%.4f  MAP=%.4f  MRR=%.4f  R@100=%.4f",
            sys_name,
            mean.get("ndcg@10", 0),
            mean.get("map", 0),
            mean.get("mrr", 0),
            mean.get("recall@100", 0),
        )

    # ── Save per-dataset raw runs ──────────────────────────────────────────
    for sys_name, run in systems.items():
        safe_name = sys_name.replace("+", "_plus_").replace(" ", "_")
        _save_json(run, output_dir / f"{name}_run_{safe_name}.json")

    return mean_metrics, per_query_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="BEIR Benchmark Evaluation for Sanshodhak",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(BEIR_DATASETS.keys()),
        help="BEIR dataset names to evaluate",
    )
    parser.add_argument(
        "--dataset",
        help="Single dataset alias (equivalent to --datasets DATASET)",
    )
    parser.add_argument(
        "--data-dir", default="beir_datasets",
        help="Directory to download/cache BEIR datasets",
    )
    parser.add_argument(
        "--output-dir", default="beir_results",
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--embed-model", default="BAAI/bge-m3",
        help="sentence-transformers model for dense retrieval",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server URL (fallback when sentence-transformers unavailable)",
    )
    parser.add_argument(
        "--skip-graph", action="store_true",
        help="Skip graph construction (faster, Dense+Hybrid only)",
    )
    parser.add_argument(
        "--expansion-mode", default="1hop",
        choices=["1hop", "2hop", "ppr", "qbpr"],
        help="Graph expansion mode for AHR",
    )
    parser.add_argument(
        "--reranker", action="store_true",
        help="Enable cross-encoder reranking (AHR+Reranker system)",
    )
    parser.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model for reranking",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Number of documents to retrieve per query",
    )

    args = parser.parse_args(argv)

    # --dataset (singular) overrides --datasets
    if args.dataset:
        args.datasets = [args.dataset]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Shared resources ───────────────────────────────────────────────────
    logger.info("Initialising embedding backend [%s]...", args.embed_model)
    embed_backend = EmbeddingBackend(
        st_model=args.embed_model,
        ollama_url=args.ollama_url,
    )

    reranker: Optional[CrossEncoderReranker] = None
    if args.reranker:
        logger.info("Loading cross-encoder reranker [%s]...", args.reranker_model)
        reranker = CrossEncoderReranker(model_name=args.reranker_model)

    # ── Per-dataset evaluation ─────────────────────────────────────────────
    all_mean: Dict[str, Dict] = {}           # {dataset: {system: {metric: value}}}
    all_per_query: Dict[str, Dict] = {}      # {dataset: {system: {metric: [values]}}}

    for ds in args.datasets:
        if ds not in BEIR_DATASETS:
            logger.warning("Unknown dataset %r — skipping (known: %s)", ds, list(BEIR_DATASETS))
            continue
        try:
            mean, per_q = evaluate_dataset(
                name=ds,
                data_dir=args.data_dir,
                embed_backend=embed_backend,
                reranker=reranker,
                output_dir=output_dir,
                skip_graph=args.skip_graph,
                expansion_mode=args.expansion_mode,
                top_k=args.top_k,
                k_values=[10, 100],
            )
            all_mean[ds] = mean
            all_per_query[ds] = per_q
        except Exception as exc:
            logger.error("Dataset %r failed: %s", ds, exc, exc_info=True)

    if not all_mean:
        logger.error("No datasets evaluated successfully — exiting")
        return

    # ── Significance tests ─────────────────────────────────────────────────
    sig_tests: Dict[str, Dict] = {}
    for ds, pq in all_per_query.items():
        sig_tests[ds] = run_significance_tests(pq, metric="ndcg@10")

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "metadata": {
            "datasets": args.datasets,
            "expansion_mode": args.expansion_mode,
            "reranker_enabled": args.reranker and (reranker is not None and reranker.available),
            "embed_model": args.embed_model,
            "top_k": args.top_k,
        },
        "results": all_mean,
        "statistical_tests": {
            ds: {
                pair: {
                    "statistic": info["statistic"],
                    "p_value": info["p_value"],
                    "significant_p05": info["significant"],
                    "n_queries": info["n_queries"],
                }
                for pair, info in tests.items()
            }
            for ds, tests in sig_tests.items()
        },
    }
    _save_json(summary, output_dir / "summary.json")

    # ── Summary CSV ───────────────────────────────────────────────────────
    csv_rows = []
    systems_seen: Set[str] = set()
    for ds in all_mean:
        systems_seen.update(all_mean[ds].keys())
    systems_list = sorted(systems_seen)
    metrics_list = ["ndcg@10", "map", "mrr", "recall@100"]

    for ds in all_mean:
        for sys_name in systems_list:
            if sys_name not in all_mean[ds]:
                continue
            row = {"dataset": ds, "system": sys_name}
            for m in metrics_list:
                row[m] = all_mean[ds][sys_name].get(m, "—")
            csv_rows.append(row)

    fieldnames = ["dataset", "system"] + metrics_list
    _save_csv(csv_rows, output_dir / "summary.csv", fieldnames)

    # ── Console table ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Dataset':<14} {'System':<18} {'NDCG@10':>8} {'MAP':>8} {'MRR':>8} {'R@100':>8}")
    print("-" * 72)
    for ds in all_mean:
        for sys_name, metrics in all_mean[ds].items():
            print(
                f"{ds:<14} {sys_name:<18} "
                f"{metrics.get('ndcg@10', 0):>8.4f} "
                f"{metrics.get('map', 0):>8.4f} "
                f"{metrics.get('mrr', 0):>8.4f} "
                f"{metrics.get('recall@100', 0):>8.4f}"
            )
        print()
    print("=" * 72)
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
