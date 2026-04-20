"""
Dense (FAISS) and Sparse (BM25) retrieval modules.

Classes
-------
EmbeddingBackend   : unified embedding (sentence-transformers primary, Ollama fallback)
BM25Retriever      : Okapi BM25, no external BM25 library dependency
DenseRetriever     : FAISS IndexFlatIP with L2-normalised vectors
HybridRetriever    : Dense + BM25 fused via RRF

All retrievers expose a unified interface::

    retriever.build(doc_ids, texts)
    results = retriever.retrieve(query, top_k=100)   # [(doc_id, score), ...]
    retriever.save(path) / retriever.load(path)
"""

import json
import logging
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Embedding Backend
# -----------------------------------------------------------------------------

class EmbeddingBackend:
    """
    Unified embedding backend.

    Tries sentence-transformers first; falls back to Ollama HTTP API when the
    package is unavailable or loading fails.

    Parameters
    ----------
    st_model     : sentence-transformers model name (HuggingFace Hub ID).
    ollama_model : Ollama model tag (used as fallback).
    ollama_url   : Ollama server base URL.
    dim          : expected embedding dimension (used only for zero-vectors on error).
    """

    def __init__(
        self,
        st_model: str = "BAAI/bge-m3",
        ollama_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        dim: int = 768,
    ) -> None:
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip("/")
        self.dim = dim
        self._st_model = None
        self._use_ollama = False

        try:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(st_model, device="cpu")
            # Detect actual embedding dimension from a test call
            probe = m.encode(["probe"], show_progress_bar=False)
            self.dim = probe.shape[1]
            self._st_model = m
            logger.info(
                "EmbeddingBackend: sentence-transformers [%s], dim=%d", st_model, self.dim
            )
        except Exception as e:
            logger.warning(
                "EmbeddingBackend: sentence-transformers unavailable (%s); using Ollama", e
            )
            self._use_ollama = True

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised float32 vectors.

        Returns
        -------
        np.ndarray of shape (len(texts), dim).
        """
        if not self._use_ollama and self._st_model is not None:
            vecs = self._st_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(vecs, dtype=np.float32)
        return self._ollama_encode(texts)

    def _ollama_encode(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """
        Encode via Ollama /api/embed (batch endpoint, available since Ollama 0.1.26+).
        Sends texts in batches with retry + exponential backoff for transient 500s.
        Warmup call ensures the model is loaded before bulk encoding begins.
        """
        import time
        import requests

        # Warmup: force model load before bulk encode to avoid cold-start 500s
        for attempt in range(5):
            try:
                r = requests.post(
                    f"{self.ollama_url}/api/embed",
                    json={"model": self.ollama_model, "input": ["warmup"]},
                    timeout=60,
                )
                r.raise_for_status()
                logger.info("EmbeddingBackend: Ollama warmup OK (attempt %d)", attempt + 1)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning("EmbeddingBackend: warmup failed (%s), retrying in %ds", exc, wait)
                time.sleep(wait)

        n = len(texts)
        all_vecs: List[np.ndarray] = []
        for start in range(0, n, batch_size):
            batch = [t[:4096] for t in texts[start: start + batch_size]]
            vecs_for_batch: Optional[List[np.ndarray]] = None
            for attempt in range(5):
                try:
                    resp = requests.post(
                        f"{self.ollama_url}/api/embed",
                        json={"model": self.ollama_model, "input": batch},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    raw = resp.json()["embeddings"]  # list of lists
                    vecs_for_batch = []
                    for vec_list in raw:
                        vec = np.array(vec_list, dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec /= norm
                        vecs_for_batch.append(vec)
                    break  # success
                except Exception as exc:
                    wait = 2 ** attempt
                    logger.warning(
                        "EmbeddingBackend: batch [%d:%d] attempt %d failed - %s; retry in %ds",
                        start, start + len(batch), attempt + 1, exc, wait,
                    )
                    time.sleep(wait)

            if vecs_for_batch is None:
                logger.error("EmbeddingBackend: batch [%d:%d] failed all retries - using zeros", start, start + len(batch))
                for _ in batch:
                    all_vecs.append(np.zeros(self.dim, dtype=np.float32))
            else:
                all_vecs.extend(vecs_for_batch)

            done = min(start + batch_size, n)
            if n > 200 and done % 1000 == 0:
                logger.info("  Ollama embed: %d / %d", done, n)

        return np.vstack(all_vecs)


# -----------------------------------------------------------------------------
# BM25 Retriever
# -----------------------------------------------------------------------------

class BM25Retriever:
    """
    Okapi BM25 sparse retriever (no external BM25 library dependency).

    Parameters
    ----------
    k1 : term-frequency saturation (default 1.5).
    b  : document-length normalisation (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: List[str] = []
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._avgdl: float = 1.0
        self._N: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def tokenise(text: str) -> List[str]:
        """Lowercase whitespace tokenisation."""
        return text.lower().split()

    def build(self, doc_ids: List[str], texts: List[str]) -> "BM25Retriever":
        """
        Build BM25 inverted index from documents.

        Parameters
        ----------
        doc_ids : unique document identifiers (parallel with texts).
        texts   : raw document text.
        """
        assert len(doc_ids) == len(texts), "doc_ids and texts must have equal length"
        self._doc_ids = list(doc_ids)
        self._N = len(texts)
        self._tf = []
        self._df = defaultdict(int)
        total_len = 0

        for text in texts:
            tokens = self.tokenise(text)
            total_len += len(tokens)
            tf: Dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self._tf.append(dict(tf))
            for t in set(tf):
                self._df[t] += 1

        self._avgdl = total_len / self._N if self._N > 0 else 1.0
        logger.info(
            "BM25Retriever: indexed %d docs, vocab=%d, avgdl=%.1f",
            self._N,
            len(self._df),
            self._avgdl,
        )
        return self

    @property
    def vocabulary(self) -> set:
        """Set of all known terms (used for OOV computation)."""
        return set(self._df.keys())

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        """BM25 score for a single document at index position doc_idx."""
        tf = self._tf[doc_idx]
        dl = sum(tf.values())
        score = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            df_t = self._df.get(t, 0)
            idf = math.log((self._N - df_t + 0.5) / (df_t + 0.5) + 1.0)
            tf_norm = (tf[t] * (self.k1 + 1)) / (
                tf[t] + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            )
            score += idf * tf_norm
        return score

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents by BM25 score.

        Returns
        -------
        list of (doc_id, bm25_score) sorted descending.
        """
        tokens = self.tokenise(query)
        scored = [
            (self._doc_ids[i], self.score(tokens, i)) for i in range(self._N)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def score_bulk(self, query: str) -> List[float]:
        """Return raw BM25 scores for all documents (useful for adaptive budget)."""
        tokens = self.tokenise(query)
        return [self.score(tokens, i) for i in range(self._N)]

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "doc_ids": self._doc_ids,
                    "tf": self._tf,
                    "df": dict(self._df),
                    "avgdl": self._avgdl,
                    "N": self._N,
                    "k1": self.k1,
                    "b": self.b,
                },
                fh,
            )

    def load(self, path: str) -> "BM25Retriever":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        self._doc_ids = d["doc_ids"]
        self._tf = d["tf"]
        self._df = defaultdict(int, d["df"])
        self._avgdl = d["avgdl"]
        self._N = d["N"]
        self.k1 = d["k1"]
        self.b = d["b"]
        return self


# -----------------------------------------------------------------------------
# Dense Retriever (FAISS)
# -----------------------------------------------------------------------------

class DenseRetriever:
    """
    Dense retriever using FAISS IndexFlatIP (cosine similarity via unit-norm vectors).

    One vector per document.  Long documents are truncated to 2048 characters
    before embedding.
    """

    def __init__(self, embedding_backend: EmbeddingBackend) -> None:
        self.backend = embedding_backend
        self._index = None
        self._doc_ids: List[str] = []

    def build(
        self,
        doc_ids: List[str],
        texts: List[str],
        batch_size: int = 32,
    ) -> "DenseRetriever":
        """
        Build FAISS index from documents.

        Parameters
        ----------
        doc_ids    : unique document identifiers.
        texts      : raw document text (truncated to 2048 chars internally).
        batch_size : embedding batch size.
        """
        import faiss

        assert len(doc_ids) == len(texts)
        self._doc_ids = list(doc_ids)
        truncated = [t[:2048] for t in texts]

        logger.info("DenseRetriever: encoding %d documents...", len(truncated))
        vecs = self.backend.encode(truncated, batch_size=batch_size)  # (N, dim)

        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)
        logger.info(
            "DenseRetriever: FAISS index built - dim=%d, n=%d",
            vecs.shape[1],
            self._index.ntotal,
        )
        return self

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Return top-k (doc_id, cosine_score) pairs."""
        q_vec = self.backend.encode([query])  # (1, dim)
        k = min(top_k, len(self._doc_ids))
        scores, indices = self._index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self._doc_ids[int(idx)], float(score)))
        return results

    def encode_query(self, query: str) -> np.ndarray:
        """Return unit-norm query embedding (used for query-biased PageRank)."""
        return self.backend.encode([query])[0]

    def save(self, index_path: str, meta_path: str) -> None:
        import faiss

        faiss.write_index(self._index, index_path)
        with open(meta_path, "w") as fh:
            json.dump(self._doc_ids, fh)

    def load(self, index_path: str, meta_path: str) -> "DenseRetriever":
        import faiss

        self._index = faiss.read_index(index_path)
        with open(meta_path) as fh:
            self._doc_ids = json.load(fh)
        return self


# -----------------------------------------------------------------------------
# Hybrid Retriever
# -----------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines DenseRetriever and BM25Retriever via Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    dense : pre-built DenseRetriever.
    bm25  : pre-built BM25Retriever.
    """

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever) -> None:
        self.dense = dense
        self.bm25 = bm25

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve via RRF fusion of dense and BM25 ranked lists.

        Parameters
        ----------
        query        : raw query string.
        top_k        : number of results to return.
        dense_weight : relative weight for the dense ranked list.
        bm25_weight  : relative weight for the BM25 ranked list.
        rrf_k        : RRF smoothing constant (Cormack et al. 2009, default 60).
        """
        from .fusion import reciprocal_rank_fusion

        fetch_k = max(top_k * 2, 100)
        dense_results = self.dense.retrieve(query, top_k=fetch_k)
        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)

        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_results, bm25_results],
            weights=[dense_weight, bm25_weight],
            k=rrf_k,
        )
        return fused[:top_k]
