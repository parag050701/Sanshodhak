"""
Result fusion strategies for hybrid retrieval.

Currently implements:
  reciprocal_rank_fusion  - Cormack, Clarke, Buettcher, SIGIR 2009.

RRF formula
-----------
For document d at rank r in ranked list i with weight w_i:

    RRF(d) += w_i / (k + r)

where k=60 is the smoothing constant that limits the influence of very high
ranks and makes the method robust to rank inconsistencies.
"""

from typing import Dict, List, Optional, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    Parameters
    ----------
    ranked_lists : list of ranked result lists; each element is a list of
                   (doc_id, score) pairs sorted descending by score.
                   The original scores are ignored - only rank positions matter.
    weights      : per-list importance weights (default: all 1.0).
    k            : RRF smoothing constant. Higher values reduce sensitivity to
                   top ranks (default 60, as in the original paper).

    Returns
    -------
    Fused ranked list of (doc_id, rrf_score) sorted descending.
    The rrf_score is the sum of weighted reciprocal ranks across all lists.

    Examples
    --------
    >>> dense = [("d1", 0.9), ("d2", 0.7), ("d3", 0.5)]
    >>> bm25  = [("d2", 12.0), ("d1", 9.0), ("d4", 8.0)]
    >>> fused = reciprocal_rank_fusion([dense, bm25])
    >>> [doc_id for doc_id, _ in fused[:3]]
    ['d1', 'd2', 'd4']
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    scores: Dict[str, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _normalised_entropy(scores: List[float]) -> float:
    """
    Compute normalised Shannon entropy of a score distribution.

    Returns a value in [0, 1]:
      0  -> all mass on one item (perfectly confident)
      1  -> uniform distribution (maximum uncertainty)
    """
    if not scores or len(scores) < 2:
        return 1.0

    # Shift to non-negative and normalise to a probability distribution
    min_s = min(scores)
    shifted = [s - min_s + 1e-9 for s in scores]
    total = sum(shifted)
    probs = [s / total for s in shifted]

    import math
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(probs))
    return entropy / max_entropy if max_entropy > 0 else 1.0


def confidence_weighted_rrf(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    lambda_cw: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Confidence-Weighted Reciprocal Rank Fusion (CW-RRF).

    Extends standard RRF by adjusting per-list weights based on each
    retriever's score distribution entropy.  When a retriever is "confident"
    (low normalised entropy - strong score separation between top results
    and the rest), its ranked list receives proportionally higher weight.

    Weight formula
    --------------
        w_i = 1 + λ x (1 − H_norm(scores_i))

    where H_norm ∈ [0, 1] is the normalised Shannon entropy of the
    retriever's score distribution and λ (default 0.5) controls the
    strength of the confidence adjustment.

    When λ = 0, CW-RRF degenerates to standard RRF with uniform weights.

    Parameters
    ----------
    ranked_lists : list of ranked result lists; each element is a list of
                   (doc_id, score) pairs sorted descending by score.
                   Unlike standard RRF, the original *scores* are used to
                   compute confidence weights.
    k            : RRF smoothing constant (default 60).
    lambda_cw    : confidence adjustment coefficient (default 0.5).
                   Higher values amplify confident retrievers more.

    Returns
    -------
    Fused ranked list of (doc_id, cwrrf_score) sorted descending.

    Examples
    --------
    >>> dense  = [("d1", 0.95), ("d2", 0.40), ("d3", 0.38)]  # confident
    >>> bm25   = [("d2", 12.0), ("d1", 11.5), ("d4", 11.3)]  # uncertain
    >>> fused = confidence_weighted_rrf([dense, bm25])
    >>> # dense gets higher weight because its scores are less uniform
    """
    # Compute confidence weights from score distributions
    weights: List[float] = []
    for ranked in ranked_lists:
        scores_only = [s for _, s in ranked]
        h_norm = _normalised_entropy(scores_only)
        w = 1.0 + lambda_cw * (1.0 - h_norm)
        weights.append(w)

    # Apply weighted RRF
    return reciprocal_rank_fusion(ranked_lists, weights=weights, k=k)
