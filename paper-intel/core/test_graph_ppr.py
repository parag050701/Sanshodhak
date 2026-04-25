"""
Plain-script tests for core.graph_ppr.

Run:
    D:/Sanshodhak/.venv/Scripts/python.exe paper-intel/core/test_graph_ppr.py

No test framework — each case prints PASS on success and the script
exits 0 only if every assert holds. Final line: ALL TESTS PASSED.
"""

from __future__ import annotations

import os
import sys

# Allow running this script directly (without `python -m`).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import networkx as nx  # noqa: E402

from core.graph_ppr import centrality_boost, query_biased_ppr  # noqa: E402


def _build_chain_graph() -> nx.Graph:
    """A deliberately path-shaped graph so 'hops from seed' is well defined.

    Layout:

        seed --(1.0)-- near --(1.0)-- mid --(1.0)-- far --(1.0)-- distant
                                                |
                                              (1.0)
                                                |
                                              branch

    Plus an isolated extra node 'extra' linked weakly to far, to test that
    weights matter (low weight => low PPR mass transfer).
    """
    g = nx.Graph()
    g.add_edge("seed", "near", weight=1.0)
    g.add_edge("near", "mid", weight=1.0)
    g.add_edge("mid", "far", weight=1.0)
    g.add_edge("far", "distant", weight=1.0)
    g.add_edge("mid", "branch", weight=1.0)
    g.add_edge("far", "extra", weight=0.05)
    return g


def test_scores_sum_to_one() -> None:
    g = _build_chain_graph()
    seeds = {"seed": 1.0}
    scores = query_biased_ppr(g, seeds)
    total = sum(scores.values())
    assert abs(total - 1.0) < 1e-3, f"PPR scores sum to {total}, expected ~1.0"
    # Every node represented.
    assert set(scores.keys()) == set(g.nodes()), "Score keys must match graph nodes"
    print("PASS: query_biased_ppr returns normalized distribution over all nodes")


def test_near_beats_far() -> None:
    g = _build_chain_graph()
    seeds = {"seed": 1.0}
    scores = query_biased_ppr(g, seeds)
    near = scores["near"]
    distant = scores["distant"]
    assert near > distant, (
        f"Expected 'near' > 'distant' in PPR, got near={near:.6f} distant={distant:.6f}"
    )
    # Monotone decay along the chain seed -> near -> mid -> far -> distant.
    assert scores["near"] > scores["mid"] > scores["far"] > scores["distant"], (
        "PPR mass should decay monotonically with hops along an unbranched chain "
        f"(near={scores['near']:.4f}, mid={scores['mid']:.4f}, "
        f"far={scores['far']:.4f}, distant={scores['distant']:.4f})"
    )
    print("PASS: 1-hop neighbor outranks 3-hop neighbor under PPR")


def test_centrality_boost_signs() -> None:
    g = _build_chain_graph()
    seeds = {"seed": 1.0}
    candidates = ["near", "mid", "far", "distant", "extra"]
    boosts = centrality_boost(g, seeds, candidates, scale=1.0)
    assert set(boosts.keys()) == set(candidates), "Boost dict must cover all candidates"
    # 'near' is closest to seed of the candidates -> should be positive.
    # 'distant' / 'extra' are the farthest / weakest-linked -> should be negative.
    assert boosts["near"] > 0.0, f"Expected positive boost for 'near', got {boosts['near']}"
    assert boosts["distant"] < 0.0, (
        f"Expected negative boost for 'distant', got {boosts['distant']}"
    )
    # Clamp range.
    for c, b in boosts.items():
        assert -1.0 - 1e-9 <= b <= 1.0 + 1e-9, f"Boost for {c} out of [-1,1]: {b}"
    print("PASS: centrality_boost positive near seed, negative far from seed")


def test_empty_graph() -> None:
    empty = nx.Graph()
    out = query_biased_ppr(empty, {"x": 1.0})
    assert out == {}, f"Empty graph should give {{}}, got {out}"
    # centrality_boost on empty graph still returns 0.0 per candidate.
    boosts = centrality_boost(empty, {"x": 1.0}, ["a", "b"])
    assert boosts == {"a": 0.0, "b": 0.0}, f"Unexpected boosts on empty graph: {boosts}"
    print("PASS: empty graph handled without exception")


def test_missing_seeds_uniform() -> None:
    g = _build_chain_graph()
    out = query_biased_ppr(g, {"not_in_graph": 1.0})
    n = g.number_of_nodes()
    expected = 1.0 / n
    for node, score in out.items():
        assert abs(score - expected) < 1e-9, (
            f"Expected uniform {expected:.6f} for {node}, got {score:.6f}"
        )
    assert abs(sum(out.values()) - 1.0) < 1e-9, "Uniform distribution must sum to 1"
    print("PASS: missing seeds fall back to uniform distribution")


def test_all_equal_ppr_zero_boost() -> None:
    # A graph with no edges => PPR equals personalization exactly.
    # With personalization spread uniformly over candidates, every candidate
    # should get the same PPR -> zero boost everywhere.
    g = nx.Graph()
    g.add_nodes_from(["a", "b", "c"])
    seeds = {"a": 1.0, "b": 1.0, "c": 1.0}
    boosts = centrality_boost(g, seeds, ["a", "b", "c"])
    assert boosts == {"a": 0.0, "b": 0.0, "c": 0.0}, (
        f"All-equal PPR must give zero boost, got {boosts}"
    )
    print("PASS: equal-PPR candidates get zero boost (no spurious signal)")


def test_determinism() -> None:
    g = _build_chain_graph()
    seeds = {"seed": 0.7, "branch": 0.3}
    s1 = query_biased_ppr(g, seeds)
    s2 = query_biased_ppr(g, seeds)
    assert s1 == s2, "query_biased_ppr must be deterministic for identical inputs"
    print("PASS: query_biased_ppr is deterministic")


def main() -> int:
    test_scores_sum_to_one()
    test_near_beats_far()
    test_centrality_boost_signs()
    test_empty_graph()
    test_missing_seeds_uniform()
    test_all_equal_ppr_zero_boost()
    test_determinism()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
