"""
Query-biased Personalized PageRank over a paper-level NetworkX graph.

Standalone module: depends only on networkx, numpy, and stdlib.
Designed to be wired into the AHR retrieval pipeline as either:
  (a) a graph-candidate generator (top-k by PPR score), or
  (b) a centrality boost for re-ranking dense+graph fused candidates.

Both functions are deterministic and never raise on the documented edge cases.
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx
import numpy as np


def _normalize_seeds(
    graph: nx.Graph, seed_papers: Dict[str, float]
) -> Dict[str, float]:
    """Filter seeds to those present in graph, drop non-positive weights, renormalize.

    Returns an empty dict if no usable seeds remain.
    """
    if not seed_papers:
        return {}
    nodes = set(graph.nodes())
    filtered: Dict[str, float] = {}
    for paper, weight in seed_papers.items():
        if paper in nodes:
            try:
                w = float(weight)
            except (TypeError, ValueError):
                continue
            if w > 0.0:
                filtered[paper] = w
    total = sum(filtered.values())
    if total <= 0.0:
        return {}
    return {p: w / total for p, w in filtered.items()}


def query_biased_ppr(
    graph: nx.Graph,
    seed_papers: Dict[str, float],
    alpha: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Run personalized PageRank on `graph` with the restart distribution
    built from `seed_papers` (normalized to sum to 1). Uses the 'weight'
    edge attribute.

    Returns: dict mapping every node in `graph` to its PPR score (scores sum to ~1).

    Edge cases (must NOT raise):
    - Empty graph        -> returns {}
    - No seeds in graph  -> returns uniform distribution over graph nodes
    - Some seeds missing -> use only the seeds present, renormalize
    """
    # Empty graph -> empty result
    if graph.number_of_nodes() == 0:
        return {}

    # Sort seeds for determinism (dict insertion order in personalization
    # can affect tie-breaking inside networkx's iteration).
    sorted_seeds = dict(sorted(seed_papers.items())) if seed_papers else {}
    personalization = _normalize_seeds(graph, sorted_seeds)

    # No usable seeds -> uniform distribution over graph nodes
    if not personalization:
        n = graph.number_of_nodes()
        uniform = 1.0 / n
        return {node: uniform for node in graph.nodes()}

    # If the graph has zero edges, networkx pagerank returns the
    # personalization vector itself (all mass stays at seeds). That is
    # the correct behavior for a disconnected graph.
    try:
        scores = nx.pagerank(
            graph,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
            weight="weight",
            dangling=personalization,  # also handle dangling nodes via seeds
        )
    except nx.PowerIterationFailedConvergence:
        # Fall back to a more permissive run; keeps the function total.
        scores = nx.pagerank(
            graph,
            alpha=alpha,
            personalization=personalization,
            max_iter=max(max_iter * 4, 200),
            tol=tol * 10,
            weight="weight",
            dangling=personalization,
        )
    except Exception:
        # Last-resort fallback: return personalization padded with zeros.
        scores = {node: 0.0 for node in graph.nodes()}
        for p, w in personalization.items():
            scores[p] = w

    # Ensure every node is present (defensive — networkx should already do this).
    return {node: float(scores.get(node, 0.0)) for node in graph.nodes()}


def centrality_boost(
    graph: nx.Graph,
    seed_papers: Dict[str, float],
    candidate_papers: List[str],
    scale: float = 1.0,
) -> Dict[str, float]:
    """
    Compute a re-ranking boost for `candidate_papers` based on their PPR
    centrality from `seed_papers`. Internally calls query_biased_ppr.

    Returns: dict mapping paper -> boost, in roughly [-scale, +scale]:
    z-score the PPR scores across the candidate_papers, then clamp to
    [-scale, +scale]. Candidates with high centrality get positive boost,
    low get negative.

    Edge cases (must NOT raise):
    - Empty candidates -> returns {}
    - All candidates have same PPR -> returns {p: 0.0 for p in candidates}
    """
    if not candidate_papers:
        return {}

    # De-duplicate while preserving order (input may have repeats).
    seen = set()
    unique_candidates: List[str] = []
    for c in candidate_papers:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    # Empty graph -> no information; return zeros.
    if graph.number_of_nodes() == 0:
        return {c: 0.0 for c in unique_candidates}

    ppr_scores = query_biased_ppr(graph, seed_papers)

    # Candidates not in the graph get score 0.0 (treated as low centrality).
    cand_scores = np.array(
        [float(ppr_scores.get(c, 0.0)) for c in unique_candidates],
        dtype=np.float64,
    )

    mean = float(cand_scores.mean())
    std = float(cand_scores.std())

    # All-equal case (or single candidate): no signal -> zero boost.
    if std <= 1e-12 or len(unique_candidates) < 2:
        return {c: 0.0 for c in unique_candidates}

    z = (cand_scores - mean) / std
    s = float(scale)
    clamped = np.clip(z, -s, s)

    return {c: float(clamped[i]) for i, c in enumerate(unique_candidates)}
