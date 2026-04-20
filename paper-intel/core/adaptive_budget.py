"""
Adaptive Graph Slot Budget Allocation (Part 3).

Replaces the fixed  kg = k // 4  slot reservation with a data-driven
allocation that reads query and retriever confidence signals.

Motivation
----------
The graph component improves recall when lexical and dense retrievers are
uncertain about the query (low score variance, high out-of-vocabulary ratio).
When BM25 shows high confidence - large gap between the top-1 score and the
rest - adding graph expansion is less beneficial and may introduce noise.

Budget formula
--------------
    budget = clamp(base + Δ_dense + Δ_bm25 + Δ_oov + Δ_short, min=1, max=k//2)

where:
    base   = round(base_fraction x k)           # default k/4

    Δ_dense_uncertainty:
        dense_var_norm = Var(scores) / (max^2 + ε)
        < 0.05  -> +2  (scores nearly equal -> retrieval uncertain)
        > 0.20  -> -1  (clear winner -> dense confident)

    Δ_bm25_confidence:
        gap = scores[0] − mean(scores[1:])
        gap > 5.0 -> -1  (BM25 is very sure about the top result)

    Δ_oov:
        oov_ratio = fraction of query tokens not in BM25 vocabulary
        > 0.50 -> +2   (many unknowns -> lexical retrieval unreliable)
        > 0.30 -> +1

    Δ_short_query:
        len(query.split()) <= 3 -> +1  (short queries are inherently ambiguous)

Public API
----------
    compute_graph_budget(query, dense_scores, bm25_scores, k, vocabulary, ...)
"""

from typing import List, Optional


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _variance(scores: List[float]) -> float:
    """Sample variance of a score list.  Returns 0 for lists shorter than 2."""
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    return sum((s - mean) ** 2 for s in scores) / (n - 1)


def _oov_ratio(query: str, vocabulary: set) -> float:
    """
    Fraction of query tokens absent from the retriever vocabulary.

    Parameters
    ----------
    query      : raw query string.
    vocabulary : set of known tokens (BM25 vocabulary).
    """
    tokens = query.lower().split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t not in vocabulary) / len(tokens)


# -----------------------------------------------------------------------------
# Public function
# -----------------------------------------------------------------------------

def compute_graph_budget(
    query: str,
    dense_scores: List[float],
    bm25_scores: List[float],
    k: int,
    vocabulary: Optional[set] = None,
    min_slots: int = 1,
    base_fraction: float = 0.25,
) -> int:
    """
    Compute the number of result slots to reserve for graph expansion.

    Parameters
    ----------
    query         : raw query string.
    dense_scores  : cosine similarity scores from dense retrieval (any length).
    bm25_scores   : BM25 scores from sparse retrieval (any length).
    k             : total number of results requested.
    vocabulary    : BM25 vocabulary set.  If None, OOV adjustment is skipped.
    min_slots     : guaranteed minimum graph slots (default 1).
    base_fraction : fraction of k used as the starting budget (default 0.25).

    Returns
    -------
    int  in [min_slots, k // 2]  - slots to reserve for graph expansion.

    Notes
    -----
    The remaining  k - budget  slots come from the hybrid retriever.
    """
    base = max(1, round(base_fraction * k))
    delta = 0

    # -- Dense uncertainty --------------------------------------------------
    if dense_scores:
        dense_var = _variance(dense_scores)
        dense_max = max(dense_scores) if dense_scores else 1.0
        dense_var_norm = dense_var / (dense_max ** 2 + 1e-9)
        if dense_var_norm < 0.05:
            delta += 2   # scores nearly uniform -> retrieval is uncertain
        elif dense_var_norm > 0.20:
            delta -= 1   # clear score gap -> dense is confident

    # -- BM25 confidence ---------------------------------------------------
    if len(bm25_scores) >= 2:
        rest_mean = sum(bm25_scores[1:]) / (len(bm25_scores) - 1)
        top1_gap = bm25_scores[0] - rest_mean
        if top1_gap > 5.0:
            delta -= 1   # BM25 strongly prefers one document

    # -- OOV ratio ---------------------------------------------------------
    if vocabulary:
        oov = _oov_ratio(query, vocabulary)
        if oov > 0.50:
            delta += 2   # many OOV tokens -> sparse retrieval breaks down
        elif oov > 0.30:
            delta += 1

    # -- Short query -------------------------------------------------------
    if len(query.strip().split()) <= 3:
        delta += 1   # short queries are ambiguous; structure helps

    budget = base + delta
    budget = max(min_slots, min(budget, k // 2))
    return budget
