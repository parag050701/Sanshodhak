"""
Centralized NVIDIA NIM embedding client.

Used by graph_rag.py, ollama_rag.py, enhanced_rag.py — all three classes used
by eval_compare.py for the 4-system comparison (VR-D, VR-H, GR, HGR). Keeping
the embedding code in one module ensures every system queries the same model
with the same input_type, so the eval is a fair comparison.

Model: nvidia/llama-3.2-nv-embedqa-1b-v2  (2048-dim, QA-tuned, L2-normalized)
"""
import os
import time
import logging
from pathlib import Path
from typing import List

import numpy as np

# Load .env at import time so any caller (eval_compare.py, app.py, etc.) picks
# up NVIDIA_NIM_API_KEY without having to call load_dotenv themselves.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

NIM_EMBED_MODEL = os.getenv("NIM_EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
NIM_EMBED_DIM = 2048

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.getenv("NVIDIA_NIM_API_KEY")
        base_url = os.getenv("NIM_API_BASE_URL", "https://integrate.api.nvidia.com/v1")
        if not api_key:
            raise RuntimeError("NVIDIA_NIM_API_KEY not set in environment")
        logger.info("Initializing NIM client (model=%s, dim=%d)", NIM_EMBED_MODEL, NIM_EMBED_DIM)
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def embed_text(text: str, input_type: str = "query") -> np.ndarray:
    """
    Single-text embedding. Returns float32 array shape (NIM_EMBED_DIM,).
    NIM returns L2-normalized vectors directly.

    input_type:
      - "query"   : retrieval-time queries
      - "passage" : ingestion-time corpus chunks
    """
    return embed_batch([text], input_type)[0]


def embed_batch(texts: List[str], input_type: str = "query") -> List[np.ndarray]:
    """Batched embedding. Returns list of float32 arrays."""
    client = _get_client()
    last_err = None
    for attempt in range(3):
        try:
            resp = client.embeddings.create(
                model=NIM_EMBED_MODEL,
                input=texts,
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            logger.warning("NIM embedding failed (attempt %d/3): %s — retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"NIM embedding failed after 3 attempts: {last_err}")
