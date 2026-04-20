"""
Neo4j-backed Knowledge Graph for Sanshodhak.

Mirrors the GraphIndex interface from core/graph.py exactly, using a
Neo4j database as the persistent graph store instead of NetworkX in-memory.

Setup
-----
Install Neo4j Desktop (https://neo4j.com/download/), create a local DBMS,
install the Graph Data Science and APOC plugins, and start the server.
Default connection: bolt://localhost:7687, user=neo4j, password=sanshodhak123.

Configure via environment variables or constructor kwargs:
    NEO4J_URI      = bolt://localhost:7687
    NEO4J_USER     = neo4j
    NEO4J_PASSWORD = sanshodhak123

Edge computation is identical to GraphIndex (vocabulary-Jaccard + embedding
similarity + citation boosts). Embeddings are kept in Python memory (self._meta)
to support QBPR without round-tripping 1024-dim vectors through Neo4j.

PPR uses Neo4j GDS when available; falls back to local NetworkX if not.
QBPR always runs in Python (GDS has no vector-personalised PageRank).
"""

import json
import logging
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Import shared helpers from the NetworkX GraphIndex module
from core.graph import (
    _top_tf_tokens,
    _title_tokens,
    _jaccard,
    _cosine,
)

logger = logging.getLogger(__name__)

_DEFAULT_URI = "bolt://127.0.0.1:7687"
_DEFAULT_USER = "neo4j"
_DEFAULT_PASSWORD = "sanshodhak123"


class Neo4jGraphIndex:
    """
    Neo4j-backed knowledge graph for Sanshodhak.

    Drop-in replacement for GraphIndex.  Same public interface:
        build(), expand_1hop(), expand_2hop(), personalized_pagerank(),
        query_biased_pagerank(), expand(), get_stats(), save(), load().

    Parameters
    ----------
    uri      : Neo4j Bolt URI (default: NEO4J_URI env var or bolt://localhost:7687)
    user     : Neo4j username  (default: NEO4J_USER env var or 'neo4j')
    password : Neo4j password  (default: NEO4J_PASSWORD env var or 'sanshodhak123')
    alpha              : weight for vocabulary-Jaccard edges (default 0.5)
    beta               : weight for embedding-similarity edges (default 0.5)
    jaccard_threshold  : min Jaccard for vocab edge (default 0.30, paper-optimal)
    embed_threshold    : min cosine for embedding edge (default 0.75)
    citation_boost     : weight added for citation edges (default 0.20)
    min_shared_tokens  : inverted-index gate - min token overlap to compute edge
    max_candidates     : max candidate pairs from inverted index per document
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        jaccard_threshold: float = 0.30,
        embed_threshold: float = 0.75,
        citation_boost: float = 0.20,
        min_shared_tokens: int = 2,
        max_candidates: int = 50,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", _DEFAULT_URI)
        self.user = user or os.getenv("NEO4J_USER", _DEFAULT_USER)
        self.password = password or os.getenv("NEO4J_PASSWORD", _DEFAULT_PASSWORD)

        self.alpha = alpha
        self.beta = beta
        self.jaccard_threshold = jaccard_threshold
        self.embed_threshold = embed_threshold
        self.citation_boost = citation_boost
        self.min_shared_tokens = min_shared_tokens
        self.max_candidates = max_candidates

        # doc_id -> {doi, title, title_tokens, tf_tokens, embedding, text_snippet}
        # Kept in Python memory so QBPR doesn't need to fetch embeddings from Neo4j.
        self._meta: Dict[str, Dict] = {}

        # Local NetworkX copy used for QBPR personalisation; invalidated on build().
        self._nx_cache = None

        # Lazy-loaded Neo4j driver
        self._driver = None

    # --------------------------------------------------------------------------
    # Driver management
    # --------------------------------------------------------------------------

    def _get_driver(self):
        """Return a cached Neo4j driver, creating it on first call."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
                self._driver.verify_connectivity()
                logger.info("Neo4jGraphIndex: connected to %s", self.uri)
            except ImportError:
                raise ImportError(
                    "neo4j package not installed. Run: pip install neo4j>=5.0.0"
                )
            except Exception as exc:
                raise ConnectionError(
                    f"Could not connect to Neo4j at {self.uri}: {exc}\n"
                    "Make sure Neo4j Desktop is running and the DBMS is started."
                ) from exc
        return self._driver

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # --------------------------------------------------------------------------
    # Schema setup
    # --------------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create constraints and indexes if they don't exist."""
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                "CREATE CONSTRAINT sanshodhak_doc_id IF NOT EXISTS "
                "FOR (n:Document) REQUIRE n.doc_id IS UNIQUE"
            )

    # --------------------------------------------------------------------------
    # Edge computation (identical logic to GraphIndex)
    # --------------------------------------------------------------------------

    def _compute_edge(
        self, meta_i: Dict, meta_j: Dict
    ) -> Tuple[float, List[str]]:
        weight = 0.0
        reasons: List[str] = []

        # 1. Vocabulary-Jaccard
        jac = _jaccard(meta_i["tf_tokens"], meta_j["tf_tokens"])
        if jac >= self.jaccard_threshold:
            weight += self.alpha * jac
            reasons.append("jaccard")

        # 2. Embedding similarity
        emb_i = meta_i.get("embedding")
        emb_j = meta_j.get("embedding")
        if emb_i is not None and emb_j is not None:
            sim = _cosine(emb_i, emb_j)
            if sim >= self.embed_threshold:
                weight += self.beta * sim
                reasons.append("embed")

        # 3. DOI co-citation
        doi_i = meta_i.get("doi", "")
        doi_j = meta_j.get("doi", "")
        if doi_i and doi_j:
            if doi_j in meta_i.get("text_snippet", "") or doi_i in meta_j.get("text_snippet", ""):
                weight += self.citation_boost
                reasons.append("citation")

        # 4. Fuzzy title match (fallback when no DOI)
        if not reasons:
            title_jac = _jaccard(meta_i["title_tokens"], meta_j["title_tokens"])
            if title_jac >= 0.30:
                weight += self.citation_boost * 0.5
                reasons.append("title_fuzzy")

        return weight, reasons

    # --------------------------------------------------------------------------
    # Build
    # --------------------------------------------------------------------------

    def build(
        self,
        doc_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> "Neo4jGraphIndex":
        """
        Build the knowledge graph in Neo4j.

        Same signature as GraphIndex.build(). Edge computation logic is
        identical - vocabulary-Jaccard with inverted-index candidate gating.
        """
        assert len(doc_ids) == len(texts) == len(embeddings)
        n = len(doc_ids)
        metadata = metadata or [{}] * n

        # Populate local metadata cache (includes embeddings for QBPR)
        self._meta = {}
        for i, doc_id in enumerate(doc_ids):
            self._meta[doc_id] = {
                "doi": (metadata[i].get("doi") or "").lower().strip(),
                "title": metadata[i].get("title", ""),
                "title_tokens": _title_tokens(metadata[i].get("title", texts[i][:80])),
                "tf_tokens": _top_tf_tokens(texts[i]),
                "embedding": embeddings[i],
                "text_snippet": texts[i][:500].lower(),
            }

        self._nx_cache = None  # invalidate NetworkX cache

        self._ensure_schema()
        driver = self._get_driver()

        logger.info("Neo4jGraphIndex: writing %d nodes...", n)

        # Write nodes in batches of 100
        batch_size = 100
        for start in range(0, n, batch_size):
            batch_ids = doc_ids[start: start + batch_size]
            node_rows = [
                {
                    "doc_id": doc_id,
                    "doi": self._meta[doc_id]["doi"],
                    "title": self._meta[doc_id]["title"],
                    "text_snippet": self._meta[doc_id]["text_snippet"],
                }
                for doc_id in batch_ids
            ]
            with driver.session() as session:
                session.run(
                    "UNWIND $rows AS row "
                    "MERGE (n:Document {doc_id: row.doc_id}) "
                    "SET n.doi = row.doi, n.title = row.title, "
                    "    n.text_snippet = row.text_snippet",
                    rows=node_rows,
                )

        # Compute edges using the same inverted-index approach as GraphIndex
        logger.info("Neo4jGraphIndex: computing edges...")
        inverted: Dict[str, List[str]] = defaultdict(list)
        for doc_id in doc_ids:
            for token in self._meta[doc_id]["tf_tokens"]:
                inverted[token].append(doc_id)

        edge_rows = []
        processed: Set[Tuple[str, str]] = set()

        for id_i in doc_ids:
            meta_i = self._meta[id_i]
            candidate_counts: Counter = Counter()
            for token in meta_i["tf_tokens"]:
                for cand_id in inverted.get(token, []):
                    if cand_id != id_i:
                        candidate_counts[cand_id] += 1

            candidates = [
                cid
                for cid, cnt in candidate_counts.most_common(self.max_candidates)
                if cnt >= self.min_shared_tokens
            ]
            for id_j in candidates:
                pair = (min(id_i, id_j), max(id_i, id_j))
                if pair in processed:
                    continue
                processed.add(pair)

                weight, reasons = self._compute_edge(self._meta[id_i], self._meta[id_j])
                if weight > 0:
                    edge_rows.append({
                        "id_i": id_i,
                        "id_j": id_j,
                        "weight": weight,
                        "reasons": reasons,
                    })

        # Write edges in batches
        for start in range(0, len(edge_rows), batch_size):
            batch = edge_rows[start: start + batch_size]
            with driver.session() as session:
                session.run(
                    "UNWIND $rows AS row "
                    "MATCH (a:Document {doc_id: row.id_i}), "
                    "      (b:Document {doc_id: row.id_j}) "
                    "MERGE (a)-[r:RELATED]-(b) "
                    "SET r.weight = row.weight, r.reasons = row.reasons",
                    rows=batch,
                )

        logger.info(
            "Neo4jGraphIndex: built - %d nodes, %d edges", n, len(edge_rows)
        )
        return self

    # --------------------------------------------------------------------------
    # Traversal
    # --------------------------------------------------------------------------

    def expand_1hop(self, seed_ids: Set[str]) -> List[Tuple[str, float]]:
        """1-hop expansion via Cypher. Returns (doc_id, score) sorted descending."""
        if not seed_ids:
            return []
        driver = self._get_driver()
        with driver.session() as session:
            result = session.run(
                "MATCH (s:Document)-[r:RELATED]-(n:Document) "
                "WHERE s.doc_id IN $seeds AND NOT n.doc_id IN $seeds "
                "RETURN n.doc_id AS doc_id, SUM(r.weight) AS score "
                "ORDER BY score DESC",
                seeds=list(seed_ids),
            )
            return [(record["doc_id"], record["score"]) for record in result]

    def expand_2hop(self, seed_ids: Set[str]) -> List[Tuple[str, float]]:
        """2-hop expansion. 1-hop scores take precedence; 2-hop discounted 0.5x."""
        hop1 = dict(self.expand_1hop(seed_ids))
        if not hop1:
            return []

        hop1_ids = set(hop1.keys())
        hop2_raw = self.expand_1hop(hop1_ids | seed_ids)
        hop2 = {
            doc_id: score * 0.5
            for doc_id, score in hop2_raw
            if doc_id not in seed_ids and doc_id not in hop1_ids
        }

        combined = {**hop2, **hop1}
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    def personalized_pagerank(
        self,
        seed_ids: Set[str],
        alpha: float = 0.85,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        PPR with uniform teleportation to seed set.

        Tries Neo4j GDS first; falls back to NetworkX on a local graph copy.
        """
        if not seed_ids:
            return []

        # Try Neo4j GDS
        try:
            return self._ppr_gds(seed_ids, alpha, top_k)
        except Exception as exc:
            logger.warning(
                "Neo4j GDS PPR failed (%s); falling back to NetworkX PPR", exc
            )

        # NetworkX fallback
        nx_graph = self._get_nx_cache()
        if nx_graph is None or nx_graph.number_of_nodes() == 0:
            return []

        n_seeds = len([s for s in seed_ids if s in nx_graph])
        if n_seeds == 0:
            return []

        personalisation = {
            nid: (1.0 / n_seeds if nid in seed_ids else 0.0)
            for nid in nx_graph.nodes()
        }
        try:
            import networkx as nx
            pr = nx.pagerank(nx_graph, alpha=alpha, personalization=personalisation, weight="weight")
        except Exception:
            return self.expand_1hop(seed_ids)[:top_k]

        ranked = [(nid, score) for nid, score in pr.items() if nid not in seed_ids]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _ppr_gds(
        self, seed_ids: Set[str], alpha: float, top_k: int
    ) -> List[Tuple[str, float]]:
        """Run PPR via Neo4j Graph Data Science plugin."""
        driver = self._get_driver()
        graph_name = "sanshodhak_ppr_tmp"

        with driver.session() as session:
            # Project a named in-memory graph for GDS
            session.run(
                "CALL gds.graph.project($name, 'Document', "
                "{RELATED: {orientation: 'UNDIRECTED', properties: 'weight'}})",
                name=graph_name,
            )
            try:
                # Resolve Neo4j internal node IDs for the seed set
                seed_internal = session.run(
                    "MATCH (n:Document) WHERE n.doc_id IN $seeds "
                    "RETURN id(n) AS nid",
                    seeds=list(seed_ids),
                ).data()
                seed_node_ids = [r["nid"] for r in seed_internal]

                result = session.run(
                    "CALL gds.pageRank.stream($name, {"
                    "  maxIterations: 20, "
                    "  dampingFactor: $alpha, "
                    "  sourceNodes: $source_nodes"
                    "}) YIELD nodeId, score "
                    "MATCH (n) WHERE id(n) = nodeId "
                    "RETURN n.doc_id AS doc_id, score "
                    "ORDER BY score DESC LIMIT $top_k",
                    name=graph_name,
                    alpha=alpha,
                    source_nodes=seed_node_ids,
                    top_k=top_k + len(seed_ids),
                )
                rows = [(r["doc_id"], r["score"]) for r in result if r["doc_id"] not in seed_ids]
                return rows[:top_k]
            finally:
                session.run("CALL gds.graph.drop($name, false)", name=graph_name)

    def query_biased_pagerank(
        self,
        query_embedding: np.ndarray,
        seed_ids: Set[str],
        alpha: float = 0.85,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Query-biased PageRank (QBPR).

        Personalisation ∝ cosine(query, doc). GDS has no native QBPR, so this
        always runs in Python using a cached NetworkX copy of the graph.
        """
        nx_graph = self._get_nx_cache()
        if nx_graph is None or nx_graph.number_of_nodes() == 0:
            return []

        sim_scores: Dict[str, float] = {}
        for nid, meta in self._meta.items():
            emb = meta.get("embedding")
            sim_scores[nid] = max(0.0, _cosine(query_embedding, emb)) if emb is not None else 0.0

        total = sum(sim_scores.values()) or 1.0
        personalisation = {
            nid: sim_scores.get(nid, 0.0) / total for nid in nx_graph.nodes()
        }

        try:
            import networkx as nx
            pr = nx.pagerank(nx_graph, alpha=alpha, personalization=personalisation, weight="weight")
        except Exception as exc:
            logger.warning("QBPR did not converge: %s", exc)
            pr = {nid: sim_scores.get(nid, 0.0) for nid in nx_graph.nodes()}

        ranked = [(nid, score) for nid, score in pr.items() if nid not in seed_ids]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _get_nx_cache(self):
        """Build/return a cached NetworkX copy of the graph (used for QBPR/PPR fallback)."""
        if self._nx_cache is not None:
            return self._nx_cache

        try:
            import networkx as nx
            driver = self._get_driver()
            with driver.session() as session:
                result = session.run(
                    "MATCH (a:Document)-[r:RELATED]-(b:Document) "
                    "RETURN a.doc_id AS id_a, b.doc_id AS id_b, r.weight AS weight"
                )
                G = nx.Graph()
                for record in result:
                    G.add_edge(record["id_a"], record["id_b"], weight=record["weight"])
            self._nx_cache = G
            logger.debug("Neo4jGraphIndex: cached NetworkX graph (%d nodes, %d edges)",
                         G.number_of_nodes(), G.number_of_edges())
        except Exception as exc:
            logger.warning("Failed to build NetworkX cache: %s", exc)
            return None

        return self._nx_cache

    def expand(
        self,
        seed_ids: Set[str],
        mode: str = "1hop",
        top_k: int = 20,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float]]:
        """Unified expansion dispatcher (same interface as GraphIndex.expand)."""
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

    # --------------------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return summary statistics from Neo4j (mirrors GraphIndex.get_stats)."""
        try:
            driver = self._get_driver()
            with driver.session() as session:
                counts = session.run(
                    "MATCH (n:Document) "
                    "OPTIONAL MATCH (n)-[r:RELATED]-() "
                    "RETURN count(DISTINCT n) AS nodes, "
                    "       count(DISTINCT r) / 2 AS edges"
                ).single()

                degree_data = session.run(
                    "MATCH (n:Document)-[r:RELATED]-() "
                    "RETURN n.doc_id AS doc_id, count(r) AS deg"
                ).data()

                edge_types: Counter = Counter()
                et_result = session.run(
                    "MATCH ()-[r:RELATED]-() "
                    "RETURN r.reasons AS reasons LIMIT 10000"
                )
                for record in et_result:
                    for reason in (record["reasons"] or []):
                        edge_types[reason] += 1

            degrees = [r["deg"] for r in degree_data]
            n_nodes = counts["nodes"] if counts else 0
            n_edges = counts["edges"] if counts else 0

            return {
                "nodes": n_nodes,
                "edges": n_edges,
                "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
                "max_degree": int(max(degrees)) if degrees else 0,
                "density": (2 * n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0,
                "edge_types": dict(edge_types),
                "backend": "neo4j",
            }
        except Exception as exc:
            logger.warning("Neo4jGraphIndex.get_stats failed: %s", exc)
            return {"nodes": 0, "edges": 0, "backend": "neo4j", "error": str(exc)}

    # --------------------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save connection config and node metadata to disk.

        Graph structure is already persisted in Neo4j - this writes only the
        connection parameters and the Python-side metadata cache (for QBPR).
        """
        # Slim metadata: exclude embeddings (large) unless the caller wants full
        slim_meta = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("embedding",)}
            for k, v in self._meta.items()
        }
        payload = {
            "backend": "neo4j",
            "uri": self.uri,
            "user": self.user,
            "config": {
                "alpha": self.alpha,
                "beta": self.beta,
                "jaccard_threshold": self.jaccard_threshold,
                "embed_threshold": self.embed_threshold,
                "citation_boost": self.citation_boost,
            },
            "meta_slim": slim_meta,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)
        logger.info("Neo4jGraphIndex: saved config to %s", path)

    def load(self, path: str) -> "Neo4jGraphIndex":
        """
        Load connection config from disk and verify Neo4j connectivity.

        Call build() to (re)populate the graph in Neo4j if it's a fresh DB.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Neo4jGraphIndex config not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self.uri = payload.get("uri", self.uri)
        self.user = payload.get("user", self.user)
        cfg = payload.get("config", {})
        self.alpha = cfg.get("alpha", self.alpha)
        self.beta = cfg.get("beta", self.beta)
        self.jaccard_threshold = cfg.get("jaccard_threshold", self.jaccard_threshold)
        self.embed_threshold = cfg.get("embed_threshold", self.embed_threshold)
        self.citation_boost = cfg.get("citation_boost", self.citation_boost)
        self._meta = payload.get("meta_slim", {})

        # Verify connectivity
        self._get_driver()
        stats = self.get_stats()
        logger.info(
            "Neo4jGraphIndex: loaded from %s - %d nodes, %d edges in Neo4j",
            path, stats.get("nodes", 0), stats.get("edges", 0),
        )
        return self
