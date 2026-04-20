"""
Enhanced Knowledge Graph module for Sanshodhak (Part 2).

Edge types
----------
1. Vocabulary-Jaccard   : TF-IDF top-200 token overlap between document pairs.
2. Embedding similarity : Cosine similarity of document-level embeddings.
3. DOI co-citation      : Exact DOI string match in body text of referencing doc.
4. Fuzzy title match    : Token Jaccard on titles + embedding confirmation
                         (for papers that lack DOI in their reference sections).

Edge weight formula
-------------------
    weight = alpha * jaccard + beta * embedding_sim + citation_boost

where alpha, beta, and citation_boost are configurable.  At least one
component must exceed its threshold for an edge to be created.

Scalability
-----------
Naive O(n^2) pairwise comparison is too slow for large corpora (>5k docs).
Instead, we build an inverted index over top-200 TF tokens and only compute
full edge weights for document pairs that share >= min_shared_tokens tokens.
This reduces candidate pairs from O(n^2) to O(n.s) where s is the average
number of token-sharing neighbours.

Traversal modes
---------------
- expand_1hop()             : direct graph neighbours of seed set
- expand_2hop()             : 1-hop then 1-hop again (second-hop discounted 0.5x)
- personalized_pagerank()   : PPR with uniform teleportation to seed set
- query_biased_pagerank()   : PPR personalised by query-document cosine similarity
"""

import logging
import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Token utilities
# -----------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "the a an of in on for and or to with is are was were be been have has do does "
    "this that these those we our their it its will can may would could should".split()
)


def _clean_tokens(text: str) -> List[str]:
    return [
        t
        for t in re.sub(r"[^\w\s]", " ", text.lower()).split()
        if t not in _STOPWORDS and len(t) > 2
    ]


def _top_tf_tokens(text: str, top_n: int = 200) -> Set[str]:
    """Return the top-N most frequent content tokens from text."""
    counter = Counter(_clean_tokens(text))
    return {t for t, _ in counter.most_common(top_n)}


def _title_tokens(title: str) -> Set[str]:
    return set(_clean_tokens(title)) if title else set()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# -----------------------------------------------------------------------------
# GraphIndex
# -----------------------------------------------------------------------------

class GraphIndex:
    """
    Pluggable knowledge graph over a document corpus.

    Parameters
    ----------
    alpha                 : weight for Jaccard component in edge weight.
    beta                  : weight for embedding cosine component.
    jaccard_threshold     : minimum Jaccard to create a vocabulary edge.
    embed_threshold       : minimum cosine to create an embedding edge.
    fuzzy_title_threshold : minimum title Jaccard to attempt fuzzy-title match.
    citation_boost        : additional weight per shared DOI citation.
    min_shared_tokens     : minimum shared TF tokens to consider a candidate pair
                            (inverted-index gate; set to 1 for maximum recall).
    max_candidates        : max candidates from inverted index per document
                            (caps computation for high-frequency tokens).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        jaccard_threshold: float = 0.30,
        embed_threshold: float = 0.75,
        fuzzy_title_threshold: float = 0.30,
        citation_boost: float = 0.20,
        min_shared_tokens: int = 2,
        max_candidates: int = 50,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.jaccard_threshold = jaccard_threshold
        self.embed_threshold = embed_threshold
        self.fuzzy_title_threshold = fuzzy_title_threshold
        self.citation_boost = citation_boost
        self.min_shared_tokens = min_shared_tokens
        self.max_candidates = max_candidates

        self.graph: nx.Graph = nx.Graph()
        # doc_id -> {doi, title, title_tokens, tf_tokens, embedding}
        self._meta: Dict[str, Dict] = {}

    # ----------------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------------

    def build(
        self,
        doc_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> "GraphIndex":
        """
        Build the knowledge graph.

        Parameters
        ----------
        doc_ids    : unique document identifiers.
        texts      : raw document text for each document.
        embeddings : pre-computed document embeddings, shape (N, dim).
        metadata   : optional per-doc dicts with keys: doi, title, year.
        """
        assert len(doc_ids) == len(texts) == len(embeddings), (
            "doc_ids, texts and embeddings must have equal length"
        )
        n = len(doc_ids)
        metadata = metadata or [{}] * n

        # Add nodes and compute per-document features
        for i, doc_id in enumerate(doc_ids):
            self.graph.add_node(doc_id)
            self._meta[doc_id] = {
                "doi": (metadata[i].get("doi") or "").lower().strip(),
                "title": metadata[i].get("title", ""),
                "title_tokens": _title_tokens(metadata[i].get("title", texts[i][:80])),
                "tf_tokens": _top_tf_tokens(texts[i]),
                "embedding": embeddings[i],
                "text_snippet": texts[i][:500].lower(),  # for DOI substring search
            }

        logger.info("GraphIndex: building edges for %d documents...", n)

        # Build inverted index: token -> [doc_ids containing that token in tf_tokens]
        inverted: Dict[str, List[str]] = defaultdict(list)
        for doc_id in doc_ids:
            for token in self._meta[doc_id]["tf_tokens"]:
                inverted[token].append(doc_id)

        edge_count = 0
        processed: Set[Tuple[str, str]] = set()

        for id_i in doc_ids:
            meta_i = self._meta[id_i]

            # Find candidates via inverted index
            candidate_counts: Counter = Counter()
            for token in meta_i["tf_tokens"]:
                for cand_id in inverted.get(token, []):
                    if cand_id != id_i:
                        candidate_counts[cand_id] += 1

            # Gate: must share at least min_shared_tokens
            candidates = [
                cid
                for cid, cnt in candidate_counts.most_common(self.max_candidates)
                if cnt >= self.min_shared_tokens
            ]

            for id_j in candidates:
                # Canonical pair key to avoid duplicate computation
                pair = (min(id_i, id_j), max(id_i, id_j))
                if pair in processed:
                    continue
                processed.add(pair)

                meta_j = self._meta[id_j]
                weight, reasons = self._compute_edge(meta_i, meta_j)

                if weight > 0:
                    self.graph.add_edge(id_i, id_j, weight=weight, reasons=reasons)
                    edge_count += 1

        logger.info(
            "GraphIndex: built - %d nodes, %d edges (%.1f avg degree)",
            self.graph.number_of_nodes(),
            edge_count,
            (2 * edge_count / n) if n > 0 else 0,
        )
        return self

    def _compute_edge(
        self, meta_i: Dict, meta_j: Dict
    ) -> Tuple[float, List[str]]:
        """Compute the edge weight and edge-type labels between two documents."""
        weight = 0.0
        reasons: List[str] = []

        # 1. Vocabulary-Jaccard
        jac = _jaccard(meta_i["tf_tokens"], meta_j["tf_tokens"])
        if jac >= self.jaccard_threshold:
            weight += self.alpha * jac
            reasons.append("jaccard")

        # 2. Embedding similarity
        cos = _cosine(meta_i["embedding"], meta_j["embedding"])
        if cos >= self.embed_threshold:
            weight += self.beta * cos
            reasons.append("embed")

        # 3. DOI co-citation (DOI of j appears in body text of i or vice-versa)
        doi_edge = False
        if meta_j["doi"] and meta_j["doi"] in meta_i["text_snippet"]:
            weight += self.citation_boost
            reasons.append("doi_cite")
            doi_edge = True
        elif meta_i["doi"] and meta_i["doi"] in meta_j["text_snippet"]:
            weight += self.citation_boost
            reasons.append("doi_cite")
            doi_edge = True

        # 4. Fuzzy title match - only if no DOI edge found
        if not doi_edge and meta_i["title_tokens"] and meta_j["title_tokens"]:
            title_jac = _jaccard(meta_i["title_tokens"], meta_j["title_tokens"])
            if title_jac >= self.fuzzy_title_threshold and cos >= 0.50:
                weight += self.citation_boost * 0.5
                reasons.append("fuzzy_title")

        return weight, reasons

    # ----------------------------------------------------------------------
    # Traversal
    # ----------------------------------------------------------------------

    def expand_1hop(self, seed_ids: Set[str]) -> List[Tuple[str, float]]:
        """
        1-hop graph expansion from seed documents.

        Aggregates edge weights across all seeds to score each neighbour.
        Seeds themselves are excluded from results.

        Returns
        -------
        list of (doc_id, aggregated_weight) sorted descending.
        """
        scores: Dict[str, float] = {}
        for sid in seed_ids:
            if sid not in self.graph:
                continue
            for nbr, attrs in self.graph[sid].items():
                if nbr not in seed_ids:
                    scores[nbr] = scores.get(nbr, 0.0) + attrs.get("weight", 1.0)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def expand_2hop(self, seed_ids: Set[str]) -> List[Tuple[str, float]]:
        """
        2-hop expansion.  1-hop neighbours are found first; then their neighbours
        are added at a 0.5x discount.  1-hop scores take precedence over 2-hop.

        Returns
        -------
        list of (doc_id, score) for all neighbours within 2 hops, sorted descending.
        """
        hop1 = dict(self.expand_1hop(seed_ids))
        hop1_ids = set(hop1.keys())

        hop2_raw = self.expand_1hop(hop1_ids | seed_ids)
        hop2 = {
            doc_id: score * 0.5
            for doc_id, score in hop2_raw
            if doc_id not in seed_ids and doc_id not in hop1_ids
        }

        combined = {**hop2, **hop1}  # hop1 wins on overlap
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    def personalized_pagerank(
        self,
        seed_ids: Set[str],
        alpha: float = 0.85,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Personalized PageRank with uniform teleportation to the seed set.

        Parameters
        ----------
        seed_ids : documents to personalise towards (teleportation targets).
        alpha    : damping factor - probability of following an edge (default 0.85).
        top_k    : number of results to return.
        """
        if self.graph.number_of_nodes() == 0 or not seed_ids:
            return []

        n_seeds = len([s for s in seed_ids if s in self.graph])
        if n_seeds == 0:
            return []

        personalisation = {
            nid: (1.0 / n_seeds if nid in seed_ids else 0.0)
            for nid in self.graph.nodes()
        }

        try:
            pr = nx.pagerank(
                self.graph, alpha=alpha, personalization=personalisation, weight="weight"
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("GraphIndex: PPR did not converge; using uniform scores")
            pr = {n: 1.0 / self.graph.number_of_nodes() for n in self.graph.nodes()}

        ranked = [
            (nid, score) for nid, score in pr.items() if nid not in seed_ids
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def query_biased_pagerank(
        self,
        query_embedding: np.ndarray,
        seed_ids: Set[str],
        alpha: float = 0.85,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Query-biased PageRank (QBPR).

        Teleportation distribution is proportional to query-document cosine
        similarity rather than uniform over seeds.  This is equivalent to the
        Andersen et al. (2006) formulation applied to dense embeddings.

        Parameters
        ----------
        query_embedding : unit-norm query vector (1-D float32).
        seed_ids        : documents already retrieved (excluded from output).
        alpha           : damping factor.
        top_k           : number of results to return.
        """
        if self.graph.number_of_nodes() == 0:
            return []

        sim_scores: Dict[str, float] = {}
        for nid, meta in self._meta.items():
            emb = meta.get("embedding")
            sim_scores[nid] = max(0.0, _cosine(query_embedding, emb)) if emb is not None else 0.0

        total = sum(sim_scores.values()) or 1.0
        personalisation = {
            nid: sim_scores.get(nid, 0.0) / total for nid in self.graph.nodes()
        }

        try:
            pr = nx.pagerank(
                self.graph, alpha=alpha, personalization=personalisation, weight="weight"
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("GraphIndex: QBPR did not converge; falling back to sim_scores")
            pr = {nid: sim_scores.get(nid, 0.0) for nid in self.graph.nodes()}

        ranked = [
            (nid, score) for nid, score in pr.items() if nid not in seed_ids
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def expand(
        self,
        seed_ids: Set[str],
        mode: str = "1hop",
        top_k: int = 20,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float]]:
        """
        Unified expansion interface.

        Parameters
        ----------
        seed_ids        : documents to expand from.
        mode            : one of '1hop', '2hop', 'ppr', 'qbpr'.
        top_k           : max results.
        query_embedding : required when mode='qbpr'.
        """
        if mode == "1hop":
            return self.expand_1hop(seed_ids)[:top_k]
        elif mode == "2hop":
            return self.expand_2hop(seed_ids)[:top_k]
        elif mode == "ppr":
            return self.personalized_pagerank(seed_ids, top_k=top_k)
        elif mode == "qbpr":
            if query_embedding is None:
                raise ValueError("query_embedding is required for mode='qbpr'")
            return self.query_biased_pagerank(query_embedding, seed_ids, top_k=top_k)
        else:
            raise ValueError(f"Unknown expansion mode: {mode!r}")

    # ----------------------------------------------------------------------
    # Statistics & persistence
    # ----------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return summary statistics about the graph."""
        n = self.graph.number_of_nodes()
        if n == 0:
            return {"nodes": 0, "edges": 0}
        degrees = [d for _, d in self.graph.degree()]
        edge_types: Counter = Counter()
        for _, _, attrs in self.graph.edges(data=True):
            for r in attrs.get("reasons", []):
                edge_types[r] += 1
        return {
            "nodes": n,
            "edges": self.graph.number_of_edges(),
            "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
            "max_degree": int(max(degrees)) if degrees else 0,
            "density": nx.density(self.graph),
            "components": nx.number_connected_components(self.graph),
            "edge_types": dict(edge_types),
        }

    def save(self, path: str) -> None:
        """Persist graph and node metadata (embeddings excluded to save space)."""
        slim_meta = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("text_snippet",)}
            for k, v in self._meta.items()
        }
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "graph": self.graph,
                    "meta": slim_meta,
                    "config": {
                        "alpha": self.alpha,
                        "beta": self.beta,
                        "jaccard_threshold": self.jaccard_threshold,
                        "embed_threshold": self.embed_threshold,
                        "fuzzy_title_threshold": self.fuzzy_title_threshold,
                        "citation_boost": self.citation_boost,
                    },
                },
                fh,
            )
        logger.info("GraphIndex: saved to %s", path)

    def load(self, path: str) -> "GraphIndex":
        """Load graph and metadata from a pickle file."""
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        self.graph = d["graph"]
        self._meta = d["meta"]
        cfg = d.get("config", {})
        self.alpha = cfg.get("alpha", self.alpha)
        self.beta = cfg.get("beta", self.beta)
        self.jaccard_threshold = cfg.get("jaccard_threshold", self.jaccard_threshold)
        self.embed_threshold = cfg.get("embed_threshold", self.embed_threshold)
        self.fuzzy_title_threshold = cfg.get("fuzzy_title_threshold", self.fuzzy_title_threshold)
        self.citation_boost = cfg.get("citation_boost", self.citation_boost)
        logger.info(
            "GraphIndex: loaded - %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self
