"""
Cross-encoder reranker module (Part 4).

Reranks the top-N retrieved candidates using a cross-encoder model from
sentence-transformers.  Fully optional - the module loads gracefully when
sentence-transformers is unavailable, and the reranker returns candidates
unchanged in that case.

Recommended model (fast on CPU, ~66 MB):
    cross-encoder/ms-marco-MiniLM-L-6-v2

Higher-quality alternative (slower, ~88 MB):
    cross-encoder/ms-marco-MiniLM-L-12-v2

Usage
-----
    from core.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    if reranker.available:
        results = reranker.rerank(query, candidates, top_k=10)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker.

    Parameters
    ----------
    model_name  : HuggingFace model ID or local path.
    max_length  : maximum token length for the cross-encoder input pair.
    top_rerank  : rerank window size - only the top-N candidates are scored.
                  Results beyond this window are dropped.
    device      : 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        top_rerank: int = 50,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.top_rerank = top_rerank
        self.device = device
        self._model = None
        self._available = False
        self._load()

    # ----------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------

    def _load(self) -> None:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, max_length=self.max_length, device=self.device
            )
            self._available = True
            logger.info("CrossEncoderReranker: loaded %s on %s", self.model_name, self.device)
        except ImportError:
            logger.warning(
                "CrossEncoderReranker: sentence-transformers not installed - reranker disabled"
            )
        except Exception as exc:
            logger.warning("CrossEncoderReranker: model load failed - %s", exc)

    @property
    def available(self) -> bool:
        """True when the cross-encoder loaded successfully."""
        return self._available

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
        text_key: str = "text",
        id_key: str = "doc_id",
    ) -> List[Dict]:
        """
        Rerank a list of candidate documents using the cross-encoder.

        Parameters
        ----------
        query      : the raw query string.
        candidates : list of dicts, each containing text (text_key) and id (id_key).
        top_k      : maximum results to return (default: all candidates).
        text_key   : key holding the passage text.
        id_key     : key holding the document identifier.

        Returns
        -------
        Candidate list sorted by cross-encoder score (descending).
        Each candidate gains a 'rerank_score' key.
        If the reranker is unavailable, returns candidates unchanged.
        """
        if not self._available or not candidates:
            return candidates

        top_k = top_k or len(candidates)
        window = candidates[: self.top_rerank]

        # Build (query, passage) pairs - truncate passages for speed
        pairs = [(query, c.get(text_key, "")[:1024]) for c in window]
        scores = self._model.predict(pairs, show_progress_bar=False)

        reranked = [
            {**cand, "rerank_score": float(score)}
            for cand, score in zip(window, scores)
        ]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def rerank_ids(
        self,
        query: str,
        doc_ids: List[str],
        id_to_text: Dict[str, str],
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Convenience wrapper: rerank a list of doc_ids and return reranked doc_ids.

        Parameters
        ----------
        query       : query string.
        doc_ids     : ordered list of document IDs to rerank.
        id_to_text  : mapping from doc_id to text (for the cross-encoder).
        top_k       : maximum results.
        """
        if not self._available:
            return doc_ids

        candidates = [
            {"doc_id": did, "text": id_to_text.get(did, "")}
            for did in doc_ids[: self.top_rerank]
        ]
        reranked = self.rerank(query, candidates, top_k=top_k)
        return [c["doc_id"] for c in reranked]
