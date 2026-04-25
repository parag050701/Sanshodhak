"""
NVIDIA NIM cloud reranker (model: nvidia/rerank-qa-mistral-4b by default).

Drop-in replacement for core.reranker.CrossEncoderReranker — exposes the same
.available property, .rerank(query, candidates, top_k, text_key) method, and
.rerank_ids() convenience wrapper. graph_rag.py:_get_reranker() can swap one
class for the other without touching the search pipeline.

NIM rerank API (separate host from embeddings):
  POST https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking
  Body : {"model": "...", "query": {"text": "..."}, "passages": [{"text": "..."}]}
  Resp : {"rankings": [{"index": int, "logit": float}, ...]}  (sorted desc)
"""
import logging
import os
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
DEFAULT_MODEL = "nvidia/rerank-qa-mistral-4b"
PASSAGE_CHAR_CAP = 2000  # ~500 tokens, well under the model's 512-token context


class NIMReranker:
    """
    Mistral-4B cross-encoder reranker hosted on NVIDIA NIM.

    Parameters
    ----------
    model_name : NIM model ID. Default reads NIM_RERANK_MODEL env var.
    top_rerank : maximum candidates sent in one rerank call (NIM accepts ~100).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_rerank: int = 50,
        timeout: int = 30,
    ) -> None:
        self.model = model_name or os.getenv("NIM_RERANK_MODEL", DEFAULT_MODEL)
        self.url = os.getenv("NIM_RERANK_BASE_URL", DEFAULT_URL)
        self.api_key = os.getenv("NIM_RERANK_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
        self.top_rerank = top_rerank
        self.timeout = timeout
        self._available = bool(self.api_key)
        if self._available:
            logger.info("NIMReranker: ready (model=%s)", self.model)
        else:
            logger.warning("NIMReranker: no API key — disabled")

    @property
    def available(self) -> bool:
        return self._available

    # ----------------------------------------------------------------------

    def _post(self, body: Dict, retries: int = 2) -> Dict:
        last_err = None
        for attempt in range(retries + 1):
            try:
                r = requests.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                if attempt == retries:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"NIM rerank failed after {retries + 1} attempts: {last_err}")

    # ----------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
        text_key: str = "text",
        id_key: str = "doc_id",
    ) -> List[Dict]:
        if not self._available or not candidates:
            return candidates

        top_k = top_k or len(candidates)
        window = candidates[: self.top_rerank]
        passages = [{"text": (c.get(text_key, "") or "")[:PASSAGE_CHAR_CAP]} for c in window]
        body = {
            "model": self.model,
            "query": {"text": query},
            "passages": passages,
        }
        try:
            resp = self._post(body)
        except Exception as e:
            logger.warning("NIM rerank failed, returning original order: %s", e)
            return candidates[:top_k]

        # NIM returns {"rankings":[{"index": int, "logit": float}, ...]}
        rankings = resp.get("rankings", [])
        score_by_idx = {r["index"]: float(r["logit"]) for r in rankings}

        reranked = [
            {**cand, "rerank_score": score_by_idx.get(i, float("-inf"))}
            for i, cand in enumerate(window)
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
        if not self._available:
            return doc_ids
        cands = [{"doc_id": d, "text": id_to_text.get(d, "")} for d in doc_ids[: self.top_rerank]]
        out = self.rerank(query, cands, top_k=top_k)
        return [c["doc_id"] for c in out]
