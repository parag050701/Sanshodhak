"""
Retrieval evaluation metrics (Part 1 / Part 4).

Implements
----------
Per-query metrics (binary and graded relevance):
    ndcg_at_k           — binary NDCG@k
    ndcg_at_k_graded    — graded NDCG@k  (uses 2^rel − 1 gain)
    precision_at_k      — P@k
    recall_at_k         — R@k
    average_precision   — AP over the full ranked list
    reciprocal_rank     — MRR component per query

Corpus-level aggregation:
    evaluate_run(run, qrels, k_values)

Statistical significance:
    wilcoxon_test(a_scores, b_scores)   — Wilcoxon signed-rank test
    paired_ttest(a_scores, b_scores)    — Paired t-test

All per-query functions take a ranked list of doc_id strings and a relevance
judgment.  Corpus-level functions take BEIR-style dicts.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Per-query metrics — binary relevance
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """
    Binary NDCG@k.

    Ideal DCG assumes all |relevant| ground-truth documents placed at top ranks.
    """
    def _dcg(lst: List[str], rel: Set[str], cutoff: int) -> float:
        return sum(
            1.0 / math.log2(rank + 2)
            for rank, doc_id in enumerate(lst[:cutoff])
            if doc_id in rel
        )

    idcg = _dcg(list(relevant), relevant, k)  # ideal: relevant docs first
    if idcg == 0:
        return 0.0
    return _dcg(ranked, relevant, k) / idcg


def ndcg_at_k_graded(
    ranked: List[str],
    qrels: Dict[str, int],
    k: int,
) -> float:
    """
    Graded NDCG@k using gain = 2^rel − 1.

    Parameters
    ----------
    ranked : ranked list of doc_ids.
    qrels  : {doc_id: relevance_score} (int ≥ 0).
    k      : cutoff.
    """
    def _gain(rel: int) -> float:
        return 2 ** rel - 1

    dcg = sum(
        _gain(qrels.get(doc_id, 0)) / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranked[:k])
    )
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = sum(
        _gain(rel) / math.log2(rank + 2)
        for rank, rel in enumerate(ideal_rels)
    )
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """P@k: fraction of top-k results that are relevant."""
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant) / k


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    """R@k: fraction of relevant documents found in top-k results."""
    if not relevant:
        return 0.0
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant) / len(relevant)


def average_precision(ranked: List[str], relevant: Set[str]) -> float:
    """
    AP (Average Precision) over the full ranked list.

    AP = (1/|R|) × Σ P@k × rel(k)
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def reciprocal_rank(ranked: List[str], relevant: Set[str]) -> float:
    """RR: 1/rank of the first relevant document (0 if none found)."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Corpus-level evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Optional[List[int]] = None,
    relevance_threshold: int = 1,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Evaluate a retrieval run against ground-truth relevance judgements.

    Parameters
    ----------
    run                 : {query_id: [doc_id, ...]} — ranked lists.
    qrels               : {query_id: {doc_id: relevance_score}}.
    k_values            : cutoffs to evaluate (default [10]).
    relevance_threshold : minimum relevance score to treat a doc as relevant
                         for binary metrics (default 1).

    Returns
    -------
    (mean_metrics, per_query_metrics)

    mean_metrics     : {metric_name: mean_value} over all queries.
    per_query_metrics: {metric_name: [value_per_query]} for significance tests.
    """
    k_values = k_values or [10]
    metric_lists: Dict[str, List[float]] = defaultdict(list)

    for qid, ranked in run.items():
        raw_rels = qrels.get(qid, {})
        # Binary relevant set
        relevant = {did for did, score in raw_rels.items() if score >= relevance_threshold}
        if not relevant:
            continue  # skip unannotated queries

        for k in k_values:
            metric_lists[f"ndcg@{k}"].append(ndcg_at_k_graded(ranked, raw_rels, k))
            metric_lists[f"p@{k}"].append(precision_at_k(ranked, relevant, k))
            metric_lists[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))

        metric_lists["map"].append(average_precision(ranked, relevant))
        metric_lists["mrr"].append(reciprocal_rank(ranked, relevant))

    per_query = dict(metric_lists)
    mean_metrics = {
        name: (sum(vals) / len(vals) if vals else 0.0)
        for name, vals in per_query.items()
    }
    return mean_metrics, per_query


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance tests
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_test(
    system_a: List[float],
    system_b: List[float],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired per-query scores (non-parametric).

    Preferred over t-test when score distributions are non-Gaussian (typical
    in IR evaluation).

    Parameters
    ----------
    system_a    : per-query metric scores for system A.
    system_b    : per-query metric scores for system B (same queries, same order).
    alternative : 'two-sided' | 'greater' | 'less'.

    Returns
    -------
    (statistic, p_value)
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise ImportError("scipy is required: pip install scipy") from exc

    if len(system_a) != len(system_b):
        raise ValueError("Score lists must be the same length (paired test)")

    stat, pval = wilcoxon(system_a, system_b, alternative=alternative, zero_method="zsplit")
    return float(stat), float(pval)


def paired_ttest(
    system_a: List[float],
    system_b: List[float],
) -> Tuple[float, float]:
    """
    Paired t-test (parametric alternative to Wilcoxon).

    Returns
    -------
    (statistic, p_value)
    """
    try:
        from scipy.stats import ttest_rel
    except ImportError as exc:
        raise ImportError("scipy is required: pip install scipy") from exc

    stat, pval = ttest_rel(system_a, system_b)
    return float(stat), float(pval)
