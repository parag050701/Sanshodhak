#!/usr/bin/env python3
"""
SciFact Deep Research Evaluation — Parts 1–7.

Runs a complete ablation study on the SciFact BEIR dataset (~5,183 docs, 300
test queries, binary relevance).  All experiments share the same FAISS and
BM25 indices; graph variants are rebuilt only when edge configuration changes.

Parts
-----
1. Main 4-system evaluation  (Dense / Hybrid / AHR / AHR+Reranker)
2. Per-query NDCG@10 delta analysis
3. Wilcoxon signed-rank tests + Cohen's d effect sizes
4. Graph traversal mode ablation  (1hop / ppr / qbpr)
5. Adaptive vs fixed graph slot budget
6. Edge type ablation  (jaccard / embed / citation / all)
7. Structured research interpretation

Usage
-----
    # Full study (~20-40 min on CPU with bge-m3)
    python run_scifact_eval.py

    # Skip cross-encoder reranker (saves time)
    python run_scifact_eval.py --no-reranker

    # Use a lighter embedding model for quick testing
    python run_scifact_eval.py --embed-model all-MiniLM-L6-v2

    # Custom data / output directories
    python run_scifact_eval.py --data-dir beir_datasets --output-dir beir_results
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.retrieval import EmbeddingBackend, BM25Retriever, DenseRetriever
from core.graph import GraphIndex
from core.fusion import reciprocal_rank_fusion
from core.adaptive_budget import compute_graph_budget
from core.reranker import CrossEncoderReranker
from core.evaluation import (
    ndcg_at_k_graded,
    precision_at_k,
    recall_at_k,
    average_precision,
    reciprocal_rank,
    wilcoxon_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scifact_eval")

# ─────────────────────────────────────────────────────────────────────────────
# Edge configuration catalogue
# ─────────────────────────────────────────────────────────────────────────────

# Each entry defines exactly which edge components are active.
# Thresholds are set to impossibly high values (2.0) to effectively disable a
# component, since Jaccard and cosine are both bounded in [0, 1].
EDGE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "jaccard_only": {
        # Vocabulary-Jaccard edges only.
        "alpha": 0.5,           "beta": 0.0,
        "jaccard_threshold": 0.10,
        "embed_threshold": 2.0,   # disabled
        "citation_boost": 0.0,
        "fuzzy_title_threshold": 2.0,  # disabled
    },
    "embed_only": {
        # Embedding cosine-similarity edges only.
        "alpha": 0.0,           "beta": 0.5,
        "jaccard_threshold": 2.0,  # disabled
        "embed_threshold": 0.75,
        "citation_boost": 0.0,
        "fuzzy_title_threshold": 2.0,  # disabled
    },
    "citation_only": {
        # DOI co-citation + fuzzy-title edges only.
        # NOTE: BEIR SciFact docs contain no DOI strings in their body text, so
        # this variant reduces to fuzzy-title matching (title token Jaccard ≥ 0.30
        # confirmed by embedding cosine ≥ 0.50).
        "alpha": 0.0,           "beta": 0.0,
        "jaccard_threshold": 2.0,  # disabled
        "embed_threshold": 2.0,    # disabled
        "citation_boost": 1.0,
        "fuzzy_title_threshold": 0.30,
    },
    "all_combined": {
        # Default: all four edge types active.
        "alpha": 0.5,           "beta": 0.5,
        "jaccard_threshold": 0.10,
        "embed_threshold": 0.75,
        "citation_boost": 0.20,
        "fuzzy_title_threshold": 0.30,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_scifact(data_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Download (if needed) and load the SciFact BEIR dataset.

    Returns
    -------
    corpus  : {doc_id: {"title": str, "text": str}}
    queries : {query_id: query_text}
    qrels   : {query_id: {doc_id: relevance_score}}
    """
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        raise ImportError(
            "The 'beir' package is required for dataset loading.\n"
            "Install with:  pip install beir"
        )

    data_path = Path(data_dir) / "scifact"
    if not data_path.exists():
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
        logger.info("Downloading SciFact from %s ...", url)
        beir_util.download_and_unzip(url, data_dir)

    corpus, queries, qrels = GenericDataLoader(str(data_path)).load(split="test")
    logger.info(
        "SciFact loaded — corpus=%d docs, %d queries, %d annotated queries",
        len(corpus), len(queries), len(qrels),
    )
    return corpus, queries, qrels


# ─────────────────────────────────────────────────────────────────────────────
# Index building
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus_texts(corpus: Dict) -> Tuple[List[str], List[str]]:
    """Extract ordered doc_ids and concatenated title+text strings."""
    doc_ids = list(corpus.keys())
    texts = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
        for d in doc_ids
    ]
    return doc_ids, texts


def build_base_indices(
    doc_ids: List[str],
    texts: List[str],
    embed_backend: EmbeddingBackend,
    batch_size: int = 32,
) -> Tuple[DenseRetriever, BM25Retriever, np.ndarray]:
    """
    Build FAISS and BM25 indices.  Also returns the document embedding matrix
    so graph variants can reuse it without re-encoding.

    Returns
    -------
    dense      : DenseRetriever with FAISS index built.
    bm25       : BM25Retriever with inverted index built.
    embeddings : np.ndarray of shape (N, dim), L2-normalised.
    """
    logger.info("Building BM25 index (%d docs)...", len(doc_ids))
    t0 = time.perf_counter()
    bm25 = BM25Retriever().build(doc_ids, texts)
    logger.info("  BM25 done in %.1fs", time.perf_counter() - t0)

    logger.info("Building FAISS index (%d docs)...", len(doc_ids))
    t0 = time.perf_counter()
    dense = DenseRetriever(embed_backend)

    # Encode all docs once; reuse embeddings for graph construction
    truncated = [t[:2048] for t in texts]
    embeddings = embed_backend.encode(truncated, batch_size=batch_size)

    import faiss
    dense._doc_ids = list(doc_ids)
    dense._index = faiss.IndexFlatIP(embeddings.shape[1])
    dense._index.add(embeddings)
    logger.info(
        "  FAISS done in %.1fs — dim=%d, n=%d",
        time.perf_counter() - t0,
        embeddings.shape[1],
        dense._index.ntotal,
    )
    return dense, bm25, embeddings


def build_graph(
    doc_ids: List[str],
    texts: List[str],
    embeddings: np.ndarray,
    corpus: Dict,
    edge_cfg: Dict[str, Any],
    label: str = "default",
) -> GraphIndex:
    """Build a GraphIndex with the specified edge configuration."""
    metadata = [
        {"title": corpus[d].get("title", ""), "doi": corpus[d].get("doi", "")}
        for d in doc_ids
    ]
    logger.info("Building GraphIndex [%s]...", label)
    t0 = time.perf_counter()
    graph = GraphIndex(**edge_cfg).build(doc_ids, texts, embeddings, metadata)
    stats = graph.get_stats()
    logger.info(
        "  Graph [%s] done in %.1fs — nodes=%d, edges=%d, avg_deg=%.2f",
        label,
        time.perf_counter() - t0,
        stats["nodes"],
        stats["edges"],
        stats.get("avg_degree", 0),
    )
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# System runners — all return {query_id: [doc_id, ...]}
# ─────────────────────────────────────────────────────────────────────────────

def _run_dense(
    queries: Dict[str, str],
    dense: DenseRetriever,
    top_k: int = 100,
) -> Dict[str, List[str]]:
    return {
        qid: [d for d, _ in dense.retrieve(qtxt, top_k=top_k)]
        for qid, qtxt in queries.items()
    }


def _run_hybrid(
    queries: Dict[str, str],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    top_k: int = 100,
    rrf_k: int = 60,
) -> Dict[str, List[str]]:
    fetch = max(top_k * 2, 100)
    result = {}
    for qid, qtxt in queries.items():
        d_res = dense.retrieve(qtxt, top_k=fetch)
        b_res = bm25.retrieve(qtxt, top_k=fetch)
        fused = reciprocal_rank_fusion([d_res, b_res], k=rrf_k)
        result[qid] = [d for d, _ in fused[:top_k]]
    return result


def _run_ahr(
    queries: Dict[str, str],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    graph: Optional[GraphIndex],
    top_k: int = 100,
    expansion_mode: str = "1hop",
    budget_mode: str = "adaptive",  # "adaptive" | "fixed"
    rrf_k: int = 60,
) -> Dict[str, List[str]]:
    """
    Adaptive Hybrid Retrieval.

    budget_mode
    -----------
    adaptive : compute_graph_budget() — data-driven per-query slot count.
    fixed    : always k // 4 slots.
    """
    fetch = max(top_k * 2, 100)
    vocab = bm25.vocabulary
    result = {}

    for qid, qtxt in queries.items():
        d_res = dense.retrieve(qtxt, top_k=fetch)
        b_res = bm25.retrieve(qtxt, top_k=fetch)

        if budget_mode == "adaptive":
            graph_budget = compute_graph_budget(
                qtxt,
                [s for _, s in d_res[:20]],
                [s for _, s in b_res[:20]],
                top_k,
                vocabulary=vocab,
            )
        else:
            graph_budget = max(1, top_k // 4)

        fused = reciprocal_rank_fusion([d_res, b_res], k=rrf_k)
        base_ids = [d for d, _ in fused]

        if graph is None or graph_budget == 0:
            result[qid] = base_ids[:top_k]
            continue

        # Seed = top (top_k − graph_budget) results from hybrid
        seed_ids: Set[str] = set(base_ids[: top_k - graph_budget])

        if expansion_mode == "qbpr":
            q_emb = dense.encode_query(qtxt)
            graph_hits = graph.expand(
                seed_ids, mode="qbpr", top_k=graph_budget, query_embedding=q_emb
            )
        else:
            graph_hits = graph.expand(seed_ids, mode=expansion_mode, top_k=graph_budget)

        graph_ids = [d for d, _ in graph_hits]

        # Merge: hybrid fills first (top_k − budget) slots; graph fills remainder
        seen: Set[str] = set()
        final: List[str] = []
        for d in base_ids:
            if d not in seen and len(final) < top_k - graph_budget:
                seen.add(d); final.append(d)
        for d in graph_ids:
            if d not in seen and len(final) < top_k:
                seen.add(d); final.append(d)
        for d in base_ids:
            if d not in seen and len(final) < top_k:
                seen.add(d); final.append(d)

        result[qid] = final
    return result


def _run_ahr_reranked(
    queries: Dict[str, str],
    dense: DenseRetriever,
    bm25: BM25Retriever,
    graph: Optional[GraphIndex],
    reranker: CrossEncoderReranker,
    id_to_text: Dict[str, str],
    top_k: int = 100,
    expansion_mode: str = "1hop",
) -> Dict[str, List[str]]:
    """AHR with cross-encoder reranking of top-50 candidates."""
    ahr_run = _run_ahr(
        queries, dense, bm25, graph, top_k=50,
        expansion_mode=expansion_mode, budget_mode="adaptive",
    )
    result = {}
    for qid, ranked_ids in ahr_run.items():
        reranked = reranker.rerank_ids(
            queries[qid], ranked_ids, id_to_text, top_k=top_k
        )
        result[qid] = reranked
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_run_metrics(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Tuple[int, ...] = (10, 100),
    relevance_threshold: int = 1,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute mean and per-query metrics.

    Returns
    -------
    mean_metrics    : {metric_name: mean_value}
    per_query       : {qid: {metric_name: value}}
    """
    metric_accum: Dict[str, List[float]] = defaultdict(list)
    per_query: Dict[str, Dict[str, float]] = {}

    for qid, ranked in run.items():
        raw = qrels.get(qid, {})
        relevant = {d for d, s in raw.items() if s >= relevance_threshold}
        if not relevant:
            continue

        row: Dict[str, float] = {}
        for k in k_values:
            row[f"ndcg@{k}"] = ndcg_at_k_graded(ranked, raw, k)
            row[f"p@{k}"] = precision_at_k(ranked, relevant, k)
            row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        row["map"] = average_precision(ranked, relevant)
        row["mrr"] = reciprocal_rank(ranked, relevant)

        per_query[qid] = row
        for m, v in row.items():
            metric_accum[m].append(v)

    mean_metrics = {
        m: round(sum(vs) / len(vs), 4) for m, vs in metric_accum.items() if vs
    }
    return mean_metrics, per_query


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d_paired(a: List[float], b: List[float]) -> float:
    """Cohen's d for paired samples: d = mean(a−b) / std(a−b)."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return float("nan")
    mean_d = sum(diffs) / n
    std_d = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (n - 1))
    return mean_d / std_d if std_d > 0 else 0.0


def _effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:   return "negligible"
    if a < 0.5:   return "small"
    if a < 0.8:   return "medium"
    return "large"


def significance_test(
    scores_a: List[float],
    scores_b: List[float],
    name_a: str,
    name_b: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test with Cohen's d effect size.

    Returns a dict with p_value, statistic, cohens_d, effect_label,
    significant, and a human-readable interpretation.
    """
    n = min(len(scores_a), len(scores_b))
    a, b = scores_a[:n], scores_b[:n]

    try:
        stat, pval = wilcoxon_test(a, b)
    except Exception as exc:
        return {"error": str(exc)}

    d = cohens_d_paired(a, b)
    sig = pval < alpha
    mean_delta = sum(a[i] - b[i] for i in range(n)) / n

    if pval < 0.001:
        sig_desc = "highly significant (p < 0.001)"
    elif pval < 0.01:
        sig_desc = "significant — strong evidence (p < 0.01)"
    elif pval < alpha:
        sig_desc = f"significant — moderate evidence (p < {alpha})"
    else:
        sig_desc = f"not significant (p = {pval:.4f} ≥ {alpha})"

    direction = f"{name_a} > {name_b}" if mean_delta > 0 else f"{name_b} > {name_a}"

    return {
        "comparison": f"{name_a} vs {name_b}",
        "n_queries": n,
        "mean_delta": round(mean_delta, 5),      # positive = A better than B
        "statistic": round(stat, 4),
        "p_value": round(pval, 6),
        "cohens_d": round(d, 4),
        "effect_label": _effect_label(d),
        "significant": sig,
        "direction": direction,
        "interpretation": f"{sig_desc}; {_effect_label(d)} effect ({direction}); Δ={mean_delta:+.4f}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-query delta analysis
# ─────────────────────────────────────────────────────────────────────────────

def per_query_delta(
    per_query_a: Dict[str, Dict[str, float]],
    per_query_b: Dict[str, Dict[str, float]],
    queries: Dict[str, str],
    bm25_vocab: set,
    name_a: str,
    name_b: str,
    metric: str = "ndcg@10",
) -> List[Dict[str, Any]]:
    """
    Compute per-query metric delta (B − A) enriched with query-level features.

    Features computed per query:
        query_length   : number of whitespace-separated tokens
        oov_ratio      : fraction of tokens not in BM25 vocabulary
        delta          : metric(B) − metric(A)  [positive = B improved]
    """
    rows = []
    common_qids = sorted(set(per_query_a) & set(per_query_b))
    for qid in common_qids:
        qtxt = queries.get(qid, "")
        tokens = qtxt.lower().split()
        oov = sum(1 for t in tokens if t not in bm25_vocab) / len(tokens) if tokens else 0.0
        score_a = per_query_a[qid].get(metric, 0.0)
        score_b = per_query_b[qid].get(metric, 0.0)
        delta = score_b - score_a
        rows.append({
            "qid": qid,
            "query": qtxt,
            "query_length": len(tokens),
            "oov_ratio": round(oov, 4),
            f"{name_a}_{metric}": round(score_a, 4),
            f"{name_b}_{metric}": round(score_b, 4),
            "delta": round(delta, 4),
            "improved": int(delta > 1e-6),
            "worsened": int(delta < -1e-6),
        })
    return rows


def summarise_delta(rows: List[Dict], name_a: str, name_b: str) -> Dict[str, Any]:
    """Aggregate statistics for a per-query delta list."""
    deltas = [r["delta"] for r in rows]
    improved = sum(1 for d in deltas if d > 1e-6)
    worsened = sum(1 for d in deltas if d < -1e-6)
    unchanged = len(deltas) - improved - worsened
    sorted_d = sorted(deltas)
    n = len(sorted_d)
    median = sorted_d[n // 2] if n % 2 else (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2
    return {
        "comparison": f"{name_b} vs {name_a}",
        "n_queries": n,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "mean_delta": round(sum(deltas) / n, 5) if n else 0,
        "median_delta": round(median, 5),
        "max_gain": round(max(deltas), 4) if deltas else 0,
        "max_loss": round(min(deltas), 4) if deltas else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    logger.info("Saved → %s", path)


def _save_csv(rows: List[Dict], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved → %s", path)


def _header(title: str) -> None:
    border = "═" * 64
    logger.info("")
    logger.info(border)
    logger.info("  %s", title)
    logger.info(border)


def _print_table(rows: List[Dict], col_keys: List[str], col_widths: List[int]) -> None:
    """Print a simple ASCII table to stdout."""
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print("\n" + fmt.format(*col_keys))
    print("  ".join("─" * w for w in col_widths))
    for row in rows:
        vals = [str(row.get(k, "—")) for k in col_keys]
        print(fmt.format(*vals))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# PART 7 — Research interpretation (generated dynamically from results)
# ─────────────────────────────────────────────────────────────────────────────

def generate_interpretation(
    exp1: Dict,           # {system: mean_metrics}
    exp3: Dict,           # significance test results
    exp4: Dict,           # {mode: mean_metrics}
    exp5: Dict,           # {budget_mode: mean_metrics}
    exp6: Dict,           # {edge_config: mean_metrics}
    delta_rows: Dict,     # {"Dense→AHR": [...], "Hybrid→AHR": [...]}
    queries: Dict[str, str],
    bm25_vocab: set,
) -> str:
    lines = [
        "",
        "=" * 70,
        "PART 7 — RESEARCH INTERPRETATION",
        "SciFact Evaluation — Structured Findings",
        "=" * 70,
        "",
    ]

    def _m(system: str, metric: str) -> float:
        return exp1.get(system, {}).get(metric, 0.0)

    def _sig(key: str) -> Dict:
        return exp3.get(key, {})

    # ── Q1: Does AHR significantly outperform Dense? ──────────────────────
    lines.append("Q1. Does AHR significantly outperform Dense?")
    lines.append("-" * 50)
    dense_ndcg = _m("Dense", "ndcg@10")
    ahr_ndcg   = _m("AHR",   "ndcg@10")
    delta_q1   = ahr_ndcg - dense_ndcg
    sig_d_a    = _sig("Dense_vs_AHR")
    lines.append(
        f"   Dense NDCG@10 = {dense_ndcg:.4f}  |  AHR NDCG@10 = {ahr_ndcg:.4f}  "
        f"|  Δ = {delta_q1:+.4f}"
    )
    if sig_d_a:
        lines.append(f"   Statistical test: {sig_d_a.get('interpretation', '—')}")
        verdict = (
            "✓  AHR significantly outperforms Dense."
            if sig_d_a.get("significant") and delta_q1 > 0
            else "✗  No significant improvement of AHR over Dense."
        )
        lines.append(f"   Verdict: {verdict}")
    lines.append("")

    # ── Q2: Dense vs Hybrid ───────────────────────────────────────────────
    lines.append("Q2. Does Hybrid (BM25+Dense+RRF) significantly outperform Dense?")
    lines.append("-" * 50)
    hyb_ndcg = _m("Hybrid", "ndcg@10")
    delta_q2 = hyb_ndcg - dense_ndcg
    sig_d_h  = _sig("Dense_vs_Hybrid")
    lines.append(
        f"   Dense NDCG@10 = {dense_ndcg:.4f}  |  Hybrid NDCG@10 = {hyb_ndcg:.4f}  "
        f"|  Δ = {delta_q2:+.4f}"
    )
    if sig_d_h:
        lines.append(f"   Statistical test: {sig_d_h.get('interpretation', '—')}")
    lines.append("")

    # ── Q3: Adaptive vs fixed budget ──────────────────────────────────────
    lines.append("Q3. Does adaptive graph slot allocation outperform fixed k/4?")
    lines.append("-" * 50)
    if exp5:
        adap = exp5.get("adaptive", {}).get("ndcg@10", float("nan"))
        fixed = exp5.get("fixed", {}).get("ndcg@10", float("nan"))
        d_adap_fix = adap - fixed
        lines.append(f"   Adaptive NDCG@10 = {adap:.4f}  |  Fixed NDCG@10 = {fixed:.4f}  |  Δ = {d_adap_fix:+.4f}")
        sig_b = _sig("AHR_adaptive_vs_AHR_fixed")
        if sig_b:
            lines.append(f"   Statistical test: {sig_b.get('interpretation', '—')}")
        verdict = (
            "✓  Adaptive allocation yields higher NDCG@10." if d_adap_fix > 0
            else "✗  Fixed k/4 performs comparably or better."
        )
        lines.append(f"   Verdict: {verdict}")
    else:
        lines.append("   [Part 5 results unavailable]")
    lines.append("")

    # ── Q4: Graph expansion modes ─────────────────────────────────────────
    lines.append("Q4. Which graph traversal mode (1hop / ppr / qbpr) is best?")
    lines.append("-" * 50)
    if exp4:
        mode_rows = sorted(
            [(m, v.get("ndcg@10", 0)) for m, v in exp4.items()],
            key=lambda x: x[1], reverse=True,
        )
        for mode, ndcg in mode_rows:
            lines.append(f"   {mode:<8}  NDCG@10 = {ndcg:.4f}")
        best_mode = mode_rows[0][0]
        sig_ppr = _sig(f"AHR_1hop_vs_AHR_ppr")
        sig_qbpr = _sig("AHR_1hop_vs_AHR_qbpr")
        verdict = f"   Best mode: {best_mode}."
        if sig_ppr and sig_qbpr:
            verdict += f"  PPR vs 1hop: {sig_ppr.get('interpretation','—')}"
        lines.append(verdict)
    else:
        lines.append("   [Part 4 results unavailable]")
    lines.append("")

    # ── Q5: Edge type contribution ────────────────────────────────────────
    lines.append("Q5. Which graph edge type contributes most?")
    lines.append("-" * 50)
    if exp6:
        edge_rows = sorted(
            [(cfg, v.get("ndcg@10", 0)) for cfg, v in exp6.items()],
            key=lambda x: x[1], reverse=True,
        )
        for cfg, ndcg in edge_rows:
            lines.append(f"   {cfg:<18}  NDCG@10 = {ndcg:.4f}")
        best_edge = edge_rows[0][0]
        all_comb = exp6.get("all_combined", {}).get("ndcg@10", 0)
        lines.append(
            f"\n   Strongest single-type: {best_edge}. "
            f"All-combined NDCG@10 = {all_comb:.4f}."
        )
        if all_comb >= edge_rows[0][1]:
            lines.append("   → Edge types are complementary (all-combined best or equal).")
        else:
            lines.append(f"   → {best_edge} alone matches or exceeds the combined model.")
        lines.append(
            "\n   NOTE: SciFact docs contain no embedded DOI strings; the "
            "'citation_only' variant reduces to fuzzy-title matching only."
        )
    else:
        lines.append("   [Part 6 results unavailable]")
    lines.append("")

    # ── Q6: Where does graph help most? ──────────────────────────────────
    lines.append("Q6. Where does graph expansion help most?")
    lines.append("-" * 50)
    rows_da = delta_rows.get("Dense→AHR", [])
    if rows_da:
        # By query length (short ≤ 4, long > 4)
        short = [r["delta"] for r in rows_da if r["query_length"] <= 4]
        long_ = [r["delta"] for r in rows_da if r["query_length"] > 4]
        mean_short = sum(short) / len(short) if short else float("nan")
        mean_long  = sum(long_)  / len(long_)  if long_  else float("nan")
        lines.append(f"   By query length:")
        lines.append(f"     Short (≤4 tokens, n={len(short)}): mean Δ NDCG@10 = {mean_short:+.4f}")
        lines.append(f"     Long  (>4 tokens, n={len(long_)}):  mean Δ NDCG@10 = {mean_long:+.4f}")
        if not math.isnan(mean_long) and not math.isnan(mean_short):
            direction = "long" if mean_long > mean_short else "short"
            lines.append(f"   → Graph benefits {direction} queries more.")

        # By OOV ratio
        lo_oov = [r["delta"] for r in rows_da if r["oov_ratio"] <= 0.30]
        hi_oov = [r["delta"] for r in rows_da if r["oov_ratio"] > 0.30]
        mean_lo = sum(lo_oov) / len(lo_oov) if lo_oov else float("nan")
        mean_hi = sum(hi_oov) / len(hi_oov) if hi_oov else float("nan")
        lines.append(f"\n   By OOV ratio:")
        lines.append(f"     Low OOV  (≤30%, n={len(lo_oov)}): mean Δ NDCG@10 = {mean_lo:+.4f}")
        lines.append(f"     High OOV (>30%, n={len(hi_oov)}):  mean Δ NDCG@10 = {mean_hi:+.4f}")
        if not math.isnan(mean_hi) and not math.isnan(mean_lo):
            direction = "OOV-heavy" if mean_hi > mean_lo else "vocabulary-rich"
            lines.append(f"   → Graph benefits {direction} queries more.")

        # Queries most improved and most harmed
        improved = sorted(rows_da, key=lambda r: r["delta"], reverse=True)[:3]
        harmed   = sorted(rows_da, key=lambda r: r["delta"])[:3]
        lines.append("\n   Top 3 queries most improved by AHR over Dense:")
        for r in improved:
            lines.append(f"     [{r['qid']}] {r['query'][:60]}  Δ={r['delta']:+.4f}")
        lines.append("   Top 3 queries most harmed:")
        for r in harmed:
            lines.append(f"     [{r['qid']}] {r['query'][:60]}  Δ={r['delta']:+.4f}")
    else:
        lines.append("   [Per-query delta data unavailable]")
    lines.append("")

    # ── Summary bullets for Results section ────────────────────────────────
    lines.append("─" * 70)
    lines.append("SUMMARY — Suitable for journal Results section")
    lines.append("─" * 70)
    bullet_lines = []

    # AHR vs Dense
    sig_da = _sig("Dense_vs_AHR")
    if sig_da:
        p = sig_da.get("p_value", 1.0)
        d = sig_da.get("cohens_d", 0.0)
        b = (
            f"AHR {'significantly' if p < 0.05 else 'does not significantly'} "
            f"outperforms the Dense baseline on SciFact NDCG@10 "
            f"(Δ={delta_q1:+.4f}, Wilcoxon p={p:.4f}, d={d:.3f} [{_effect_label(d)} effect])."
        )
        bullet_lines.append(b)

    # Hybrid vs Dense
    sig_dh = _sig("Dense_vs_Hybrid")
    if sig_dh:
        dh_delta = hyb_ndcg - dense_ndcg
        p = sig_dh.get("p_value", 1.0)
        d = sig_dh.get("cohens_d", 0.0)
        bullet_lines.append(
            f"Hybrid (BM25+Dense+RRF) {'significantly' if p < 0.05 else 'does not significantly'} "
            f"outperforms Dense (Δ={dh_delta:+.4f}, p={p:.4f}, d={d:.3f})."
        )

    # Adaptive budget
    if exp5:
        adap_n = exp5.get("adaptive", {}).get("ndcg@10", float("nan"))
        fix_n  = exp5.get("fixed", {}).get("ndcg@10", float("nan"))
        if not math.isnan(adap_n) and not math.isnan(fix_n):
            diff_b = adap_n - fix_n
            bullet_lines.append(
                f"Adaptive slot allocation {'outperforms' if diff_b > 0 else 'underperforms'} "
                f"fixed k/4 by {abs(diff_b):.4f} NDCG@10, suggesting that "
                f"{'query-aware' if diff_b > 0 else 'uniform'} budgeting is beneficial."
            )

    # Best expansion mode
    if exp4:
        best_m, best_v = max(exp4.items(), key=lambda x: x[1].get("ndcg@10", 0))
        bullet_lines.append(
            f"Among graph traversal modes, {best_m} achieves the highest NDCG@10 "
            f"({best_v.get('ndcg@10', 0):.4f}) on SciFact."
        )

    # Best edge type
    if exp6:
        best_e, best_ev = max(exp6.items(), key=lambda x: x[1].get("ndcg@10", 0))
        bullet_lines.append(
            f"The most effective single edge type is {best_e} (NDCG@10 = "
            f"{best_ev.get('ndcg@10', 0):.4f}), indicating that "
            + (
                "vocabulary overlap is the primary signal for graph connectivity in SciFact."
                if best_e == "jaccard_only"
                else "semantic embedding similarity is the primary signal for graph connectivity."
                if best_e == "embed_only"
                else "structural title similarity drives graph connectivity."
            )
        )

    # Where graph helps
    if rows_da:
        bullet_lines.append(
            f"Graph expansion yields greater benefits for "
            f"{'longer' if mean_long > mean_short else 'shorter'} queries and "
            f"{'OOV-heavy' if mean_hi > mean_lo else 'vocabulary-rich'} queries, "
            f"consistent with the hypothesis that structural connectivity compensates "
            f"for lexical sparsity."
        )

    for i, b in enumerate(bullet_lines, 1):
        lines.append(f"• {b}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="SciFact deep research evaluation — Parts 1–7",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="beir_datasets")
    parser.add_argument("--output-dir", default="beir_results")
    parser.add_argument("--embed-model", default="BAAI/bge-m3")
    parser.add_argument("--ollama-model", default="nomic-embed-text",
                        help="Ollama embedding model (fallback when sentence-transformers unavailable)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--no-reranker", dest="reranker", action="store_false",
        help="Skip AHR+Reranker system (saves time)",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Shared setup ───────────────────────────────────────────────────────
    logger.info("Initialising embedding backend [%s]...", args.embed_model)
    embed = EmbeddingBackend(
        st_model=args.embed_model,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )

    reranker: Optional[CrossEncoderReranker] = None
    if args.reranker:
        reranker = CrossEncoderReranker()
        if not reranker.available:
            logger.warning("Reranker unavailable — AHR+Reranker will be skipped")
            reranker = None

    corpus, queries, qrels = load_scifact(args.data_dir)
    doc_ids, texts = build_corpus_texts(corpus)
    id_to_text = dict(zip(doc_ids, texts))

    dense, bm25, embeddings = build_base_indices(
        doc_ids, texts, embed, batch_size=args.batch_size
    )
    bm25_vocab = bm25.vocabulary

    # Default graph (all edges combined)
    default_graph = build_graph(
        doc_ids, texts, embeddings, corpus, EDGE_CONFIGS["all_combined"], "all_combined"
    )

    top_k = args.top_k

    # Collect all results for Part 7
    all_exp1: Dict[str, Dict] = {}
    all_exp3: Dict[str, Dict] = {}
    all_exp4: Dict[str, Dict] = {}
    all_exp5: Dict[str, Dict] = {}
    all_exp6: Dict[str, Dict] = {}
    all_delta_rows: Dict[str, List[Dict]] = {}
    per_query_scores: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ══════════════════════════════════════════════════════════════════════
    # PART 1 — Main 4-system evaluation
    # ══════════════════════════════════════════════════════════════════════
    _header("PART 1 — Main 4-system evaluation")

    systems_to_run = {
        "Dense":   lambda: _run_dense(queries, dense, top_k),
        "Hybrid":  lambda: _run_hybrid(queries, dense, bm25, top_k),
        "AHR":     lambda: _run_ahr(queries, dense, bm25, default_graph, top_k,
                                    expansion_mode="1hop", budget_mode="adaptive"),
    }
    if reranker is not None:
        systems_to_run["AHR+Reranker"] = lambda: _run_ahr_reranked(
            queries, dense, bm25, default_graph, reranker, id_to_text, top_k
        )

    part1_csv_rows: List[Dict] = []
    for sys_name, runner in systems_to_run.items():
        logger.info("Running %s...", sys_name)
        t0 = time.perf_counter()
        run = runner()
        elapsed = time.perf_counter() - t0
        mean_m, pq = compute_run_metrics(run, qrels, k_values=(10, 100))
        all_exp1[sys_name] = mean_m
        per_query_scores[sys_name] = pq
        logger.info(
            "  %-18s  NDCG@10=%.4f  MAP=%.4f  MRR=%.4f  R@100=%.4f  (%.0fs)",
            sys_name,
            mean_m.get("ndcg@10", 0), mean_m.get("map", 0),
            mean_m.get("mrr", 0), mean_m.get("recall@100", 0),
            elapsed,
        )
        part1_csv_rows.append({"system": sys_name, **mean_m})

    _save_json({"systems": all_exp1}, out / "scifact_results.json")
    _save_csv(part1_csv_rows, out / "scifact_results.csv",
              ["system", "ndcg@10", "p@10", "recall@10", "map", "mrr", "recall@100"])

    # ══════════════════════════════════════════════════════════════════════
    # PART 2 — Per-query NDCG@10 delta analysis
    # ══════════════════════════════════════════════════════════════════════
    _header("PART 2 — Per-query NDCG@10 delta analysis")

    comparisons = [
        ("Dense", "AHR"),
        ("Hybrid", "AHR"),
    ]
    if "AHR+Reranker" in per_query_scores:
        comparisons.append(("AHR", "AHR+Reranker"))

    delta_csv_rows: List[Dict] = []
    for name_a, name_b in comparisons:
        if name_a not in per_query_scores or name_b not in per_query_scores:
            continue
        rows = per_query_delta(
            per_query_scores[name_a], per_query_scores[name_b],
            queries, bm25_vocab, name_a, name_b,
        )
        key = f"{name_a}→{name_b}"
        all_delta_rows[key] = rows
        summ = summarise_delta(rows, name_a, name_b)
        logger.info(
            "  %s → %s:  improved=%d  worsened=%d  mean_Δ=%+.4f  median_Δ=%+.4f",
            name_a, name_b,
            summ["improved"], summ["worsened"],
            summ["mean_delta"], summ["median_delta"],
        )
        for r in rows:
            delta_csv_rows.append({
                "comparison": key,
                "qid": r["qid"],
                "query": r["query"],
                "query_length": r["query_length"],
                "oov_ratio": r["oov_ratio"],
                f"{name_a}_ndcg@10": r[f"{name_a}_ndcg@10"],
                f"{name_b}_ndcg@10": r[f"{name_b}_ndcg@10"],
                "delta": r["delta"],
                "improved": r["improved"],
                "worsened": r["worsened"],
            })

    _save_csv(delta_csv_rows, out / "scifact_query_deltas.csv")

    # ══════════════════════════════════════════════════════════════════════
    # PART 3 — Statistical significance tests
    # ══════════════════════════════════════════════════════════════════════
    _header("PART 3 — Statistical significance (Wilcoxon + Cohen's d)")

    test_pairs = [
        ("Dense", "Hybrid"),
        ("Dense", "AHR"),
        ("Hybrid", "AHR"),
    ]
    if "AHR+Reranker" in per_query_scores:
        test_pairs.append(("AHR", "AHR+Reranker"))

    for name_a, name_b in test_pairs:
        if name_a not in per_query_scores or name_b not in per_query_scores:
            continue
        qids = sorted(set(per_query_scores[name_a]) & set(per_query_scores[name_b]))
        a_scores = [per_query_scores[name_a][q].get("ndcg@10", 0) for q in qids]
        b_scores = [per_query_scores[name_b][q].get("ndcg@10", 0) for q in qids]
        result = significance_test(a_scores, b_scores, name_a, name_b)
        key = f"{name_a}_vs_{name_b}"
        all_exp3[key] = result
        logger.info("  %s", result.get("interpretation", "—"))

    _save_json(all_exp3, out / "scifact_significance.json")

    # ══════════════════════════════════════════════════════════════════════
    # PART 4 — Graph traversal mode ablation
    # ══════════════════════════════════════════════════════════════════════
    _header("PART 4 — Graph traversal mode ablation (1hop / ppr / qbpr)")

    modes = ["1hop", "ppr", "qbpr"]
    mode_csv_rows: List[Dict] = []
    mode_pq: Dict[str, Dict[str, Dict[str, float]]] = {}

    for mode in modes:
        logger.info("  Running AHR [%s]...", mode)
        run = _run_ahr(
            queries, dense, bm25, default_graph, top_k,
            expansion_mode=mode, budget_mode="adaptive",
        )
        mean_m, pq = compute_run_metrics(run, qrels, k_values=(10, 100))
        all_exp4[mode] = mean_m
        mode_pq[mode] = pq
        logger.info(
            "    %-8s  NDCG@10=%.4f  MAP=%.4f",
            mode, mean_m.get("ndcg@10", 0), mean_m.get("map", 0),
        )
        mode_csv_rows.append({"expansion_mode": mode, **mean_m})

    # Significance between modes
    for ma, mb in [("1hop", "ppr"), ("1hop", "qbpr"), ("ppr", "qbpr")]:
        if ma not in mode_pq or mb not in mode_pq:
            continue
        qids = sorted(set(mode_pq[ma]) & set(mode_pq[mb]))
        sa = [mode_pq[ma][q].get("ndcg@10", 0) for q in qids]
        sb = [mode_pq[mb][q].get("ndcg@10", 0) for q in qids]
        res = significance_test(sa, sb, f"AHR_{ma}", f"AHR_{mb}")
        key = f"AHR_{ma}_vs_AHR_{mb}"
        all_exp3[key] = res
        logger.info("  %s", res.get("interpretation", "—"))

    _save_csv(mode_csv_rows, out / "scifact_graph_modes.csv",
              ["expansion_mode", "ndcg@10", "map", "mrr", "recall@100"])

    # ══════════════════════════════════════════════════════════════════════
    # PART 5 — Adaptive vs fixed budget ablation
    # ══════════════════════════════════════════════════════════════════════
    _header("PART 5 — Adaptive vs fixed graph slot budget")

    budget_pq: Dict[str, Dict] = {}
    for bmode in ("adaptive", "fixed"):
        logger.info("  Running AHR [budget=%s]...", bmode)
        run = _run_ahr(
            queries, dense, bm25, default_graph, top_k,
            expansion_mode="1hop", budget_mode=bmode,
        )
        mean_m, pq = compute_run_metrics(run, qrels, k_values=(10, 100))
        all_exp5[bmode] = mean_m
        budget_pq[bmode] = pq
        logger.info(
            "    %-10s  NDCG@10=%.4f  R@100=%.4f",
            bmode, mean_m.get("ndcg@10", 0), mean_m.get("recall@100", 0),
        )

    if "adaptive" in budget_pq and "fixed" in budget_pq:
        qids = sorted(set(budget_pq["adaptive"]) & set(budget_pq["fixed"]))
        sa = [budget_pq["adaptive"][q].get("ndcg@10", 0) for q in qids]
        sb = [budget_pq["fixed"][q].get("ndcg@10", 0) for q in qids]
        res = significance_test(sa, sb, "AHR_adaptive", "AHR_fixed")
        all_exp3["AHR_adaptive_vs_AHR_fixed"] = res
        logger.info("  %s", res.get("interpretation", "—"))

    budget_csv = [{"budget_mode": k, **v} for k, v in all_exp5.items()]
    _save_csv(budget_csv, out / "scifact_budget_ablation.csv",
              ["budget_mode", "ndcg@10", "map", "recall@100"])

    # ══════════════════════════════════════════════════════════════════════
    # PART 6 — Edge type ablation
    # ══════════════════════════════════════════════════════════════════════
    _header("PART 6 — Edge type ablation")

    edge_csv_rows: List[Dict] = []
    for cfg_name, edge_cfg in EDGE_CONFIGS.items():
        if cfg_name == "all_combined":
            # Already built — reuse default_graph
            g = default_graph
        else:
            g = build_graph(doc_ids, texts, embeddings, corpus, edge_cfg, cfg_name)

        logger.info("  Running AHR [edges=%s]...", cfg_name)
        run = _run_ahr(
            queries, dense, bm25, g, top_k,
            expansion_mode="1hop", budget_mode="adaptive",
        )
        mean_m, _ = compute_run_metrics(run, qrels, k_values=(10, 100))
        all_exp6[cfg_name] = mean_m
        gs = g.get_stats()
        logger.info(
            "    %-20s  NDCG@10=%.4f  MAP=%.4f  edges=%d",
            cfg_name, mean_m.get("ndcg@10", 0), mean_m.get("map", 0), gs.get("edges", 0),
        )
        edge_csv_rows.append({
            "edge_config": cfg_name,
            "graph_edges": gs.get("edges", 0),
            **mean_m,
        })

    _save_csv(edge_csv_rows, out / "scifact_edge_ablation.csv",
              ["edge_config", "graph_edges", "ndcg@10", "map", "mrr", "recall@100"])

    # ══════════════════════════════════════════════════════════════════════
    # Save consolidated significance results
    # ══════════════════════════════════════════════════════════════════════
    _save_json(all_exp3, out / "scifact_significance.json")

    # ══════════════════════════════════════════════════════════════════════
    # PART 7 — Research interpretation
    # ══════════════════════════════════════════════════════════════════════
    _header("PART 7 — Research interpretation")

    interp = generate_interpretation(
        exp1=all_exp1,
        exp3=all_exp3,
        exp4=all_exp4,
        exp5=all_exp5,
        exp6=all_exp6,
        delta_rows=all_delta_rows,
        queries=queries,
        bm25_vocab=bm25_vocab,
    )
    print(interp)

    interp_path = out / "scifact_interpretation.txt"
    interp_path.write_text(interp)
    logger.info("Interpretation saved → %s", interp_path)

    # ══════════════════════════════════════════════════════════════════════
    # Final summary table (stdout)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 68)
    print(f"{'PART 1 — Main Results':^68}")
    print(f"{'System':<22}{'NDCG@10':>10}{'MAP':>10}{'MRR':>10}{'R@100':>10}")
    print("─" * 62)
    for row in part1_csv_rows:
        print(
            f"{row['system']:<22}"
            f"{row.get('ndcg@10', 0):>10.4f}"
            f"{row.get('map', 0):>10.4f}"
            f"{row.get('mrr', 0):>10.4f}"
            f"{row.get('recall@100', 0):>10.4f}"
        )
    print("═" * 68)
    print(f"\nAll results saved to:  {out}/")
    print("  scifact_results.json / .csv")
    print("  scifact_query_deltas.csv")
    print("  scifact_significance.json")
    print("  scifact_graph_modes.csv")
    print("  scifact_budget_ablation.csv")
    print("  scifact_edge_ablation.csv")
    print("  scifact_interpretation.txt")


if __name__ == "__main__":
    main()
