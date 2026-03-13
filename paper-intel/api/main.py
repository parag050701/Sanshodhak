"""
Sanshodhak Production API — Part 6.

FastAPI application with:
  - Startup preloading of FAISS + BM25 + Graph indices
  - Async request handling
  - Single-request queue (one query at a time; prevents OOM on low-RAM servers)
  - Config file driven model selection
  - Structured logging
  - Health check endpoint

Endpoints
---------
    GET  /health            — liveness probe
    GET  /ready             — readiness probe (True once indices loaded)
    GET  /stats             — graph stats + index info
    POST /search            — retrieve top-k documents
    POST /query             — retrieve + generate answer via Ollama

Usage
-----
    # development
    uvicorn api.main:app --host 0.0.0.0 --port 8087 --reload

    # production (from project root)
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8087 --workers 1

Environment variables (override config.yaml)
--------------------------------------------
    CONFIG_PATH   : path to config.yaml (default: config.yaml in project root)
    INDEX_DIR     : directory containing pre-built indices
    LOG_LEVEL     : DEBUG | INFO | WARNING (default: INFO)
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Ensure project root is in path when running as `uvicorn api.main:app`
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.retrieval import EmbeddingBackend, BM25Retriever, DenseRetriever, HybridRetriever
from core.graph import GraphIndex
from core.fusion import reciprocal_rank_fusion
from core.adaptive_budget import compute_graph_budget
from core.reranker import CrossEncoderReranker


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%H:%M:%S",
    )


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_setup_logging(LOG_LEVEL)
logger = logging.getLogger("sanshodhak.api")


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    config_path = os.getenv(
        "CONFIG_PATH",
        str(Path(__file__).parent.parent / "config.yaml"),
    )
    if Path(config_path).exists():
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        logger.info("Loaded config from %s", config_path)
        return cfg
    logger.warning("config.yaml not found at %s — using defaults", config_path)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Application state
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    """Holds all loaded models and indices.  Populated at startup."""
    embed_backend: Optional[EmbeddingBackend] = None
    dense: Optional[DenseRetriever] = None
    bm25: Optional[BM25Retriever] = None
    graph: Optional[GraphIndex] = None
    reranker: Optional[CrossEncoderReranker] = None
    ready: bool = False
    start_time: float = time.time()
    config: Dict[str, Any] = {}
    # Index-level doc_id → text mapping for reranker
    id_to_text: Dict[str, str] = {}


_state = AppState()

# Single-request semaphore: prevents concurrent query processing on CPU
_query_semaphore = asyncio.Semaphore(1)


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

def _load_indices(cfg: Dict[str, Any]) -> None:
    """
    Load pre-built indices from disk (or build from raw texts if available).
    Called synchronously at startup.
    """
    index_dir = Path(
        os.getenv("INDEX_DIR", cfg.get("api", {}).get("index_dir", "enhanced_rag_index"))
    )

    embed_cfg = cfg.get("embedding", {})
    st_model = embed_cfg.get("model", "BAAI/bge-m3")
    ollama_model = embed_cfg.get("ollama_model", "bge-m3")
    ollama_url = embed_cfg.get("ollama_url", "http://localhost:11434")

    logger.info("Loading embedding backend [%s]...", st_model)
    _state.embed_backend = EmbeddingBackend(
        st_model=st_model,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )

    # ── Load FAISS ────────────────────────────────────────────────────────
    faiss_index = index_dir / "faiss.index"
    faiss_meta = index_dir / "faiss_meta.json"
    if faiss_index.exists() and faiss_meta.exists():
        logger.info("Loading FAISS index from %s...", index_dir)
        _state.dense = DenseRetriever(_state.embed_backend).load(
            str(faiss_index), str(faiss_meta)
        )
    else:
        logger.warning(
            "FAISS index not found at %s — dense retrieval unavailable until rebuilt",
            index_dir,
        )

    # ── Load BM25 ─────────────────────────────────────────────────────────
    bm25_path = index_dir / "bm25.pkl"
    if bm25_path.exists():
        logger.info("Loading BM25 index from %s...", bm25_path)
        _state.bm25 = BM25Retriever().load(str(bm25_path))
    else:
        logger.warning("BM25 index not found at %s", bm25_path)

    # ── Load Graph ────────────────────────────────────────────────────────
    graph_path = index_dir / "graph.pkl"
    graph_cfg = cfg.get("graph", {})
    if graph_cfg.get("enabled", True) and graph_path.exists():
        logger.info("Loading GraphIndex from %s...", graph_path)
        _state.graph = GraphIndex().load(str(graph_path))
    else:
        logger.warning("Graph index not found at %s (or graph disabled)", graph_path)

    # ── Reranker ──────────────────────────────────────────────────────────
    reranker_cfg = cfg.get("reranker", {})
    if reranker_cfg.get("enabled", False):
        model = reranker_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Loading reranker [%s]...", model)
        _state.reranker = CrossEncoderReranker(
            model_name=model,
            top_rerank=reranker_cfg.get("top_rerank", 50),
        )
    else:
        logger.info("Reranker disabled (set reranker.enabled=true in config.yaml to enable)")

    _state.ready = _state.dense is not None or _state.bm25 is not None
    logger.info(
        "Startup complete — dense=%s, bm25=%s, graph=%s, reranker=%s",
        _state.dense is not None,
        _state.bm25 is not None,
        _state.graph is not None,
        _state.reranker is not None and _state.reranker.available,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load indices at startup; clean up on shutdown."""
    logger.info("Sanshodhak API starting up...")
    cfg = _load_config()
    _state.config = cfg

    # Load indices in a thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_indices, cfg)

    yield  # application is running

    logger.info("Sanshodhak API shutting down")
    _state.ready = False


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sanshodhak Research Paper RAG API",
    description=(
        "Adaptive Hybrid Retrieval with Graph expansion for academic paper Q&A. "
        "Open-source, CPU-compatible, fully self-hosted."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    mode: str = Field(
        default="ahr",
        description="Retrieval mode: 'dense' | 'hybrid' | 'ahr' (hybrid+graph)",
    )
    expansion_mode: str = Field(
        default="1hop",
        description="Graph expansion mode: '1hop' | '2hop' | 'ppr' | 'qbpr'",
    )
    rerank: bool = Field(default=False, description="Apply cross-encoder reranking")


class SearchResult(BaseModel):
    doc_id: str
    score: float
    title: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None  # e.g. "dense", "bm25", "graph"


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    latency_ms: float
    mode: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="ahr")
    expansion_mode: str = Field(default="1hop")
    ollama_model: Optional[str] = Field(
        default=None,
        description="Ollama model to use (overrides config)",
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    uptime_s: float
    indices: Dict[str, bool]


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval helper
# ─────────────────────────────────────────────────────────────────────────────

def _retrieve(
    query: str,
    top_k: int,
    mode: str,
    expansion_mode: str,
) -> List[SearchResult]:
    """Synchronous retrieval — called inside run_in_executor."""
    from core.fusion import reciprocal_rank_fusion
    from core.adaptive_budget import compute_graph_budget

    results_with_source: List[SearchResult] = []

    if mode == "dense":
        if _state.dense is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Dense index not loaded")
        raw = _state.dense.retrieve(query, top_k=top_k)
        for did, score in raw:
            results_with_source.append(
                SearchResult(doc_id=did, score=round(score, 4), source="dense")
            )

    elif mode == "hybrid":
        if _state.dense is None or _state.bm25 is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Indices not fully loaded")
        fetch = max(top_k * 2, 50)
        d_res = _state.dense.retrieve(query, top_k=fetch)
        b_res = _state.bm25.retrieve(query, top_k=fetch)
        fused = reciprocal_rank_fusion([d_res, b_res])
        for did, score in fused[:top_k]:
            results_with_source.append(
                SearchResult(doc_id=did, score=round(score, 4), source="hybrid")
            )

    elif mode == "ahr":
        if _state.dense is None or _state.bm25 is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Indices not fully loaded")
        fetch = max(top_k * 2, 50)
        d_res = _state.dense.retrieve(query, top_k=fetch)
        b_res = _state.bm25.retrieve(query, top_k=fetch)

        dense_scores = [s for _, s in d_res[:20]]
        bm25_scores = [s for _, s in b_res[:20]]
        vocab = _state.bm25.vocabulary if _state.bm25 else set()

        graph_budget = compute_graph_budget(
            query, dense_scores, bm25_scores, top_k, vocabulary=vocab
        )

        fused = reciprocal_rank_fusion([d_res, b_res])
        base_ids = [did for did, _ in fused]

        seen: set = set()
        final: List[str] = []
        sources: Dict[str, str] = {}

        if _state.graph is None or graph_budget == 0:
            for did in base_ids[:top_k]:
                final.append(did)
                sources[did] = "hybrid"
        else:
            seed_ids = set(base_ids[: top_k - graph_budget])
            if expansion_mode == "qbpr":
                q_emb = _state.dense.encode_query(query)
                graph_results = _state.graph.expand(
                    seed_ids, mode="qbpr", top_k=graph_budget, query_embedding=q_emb
                )
            else:
                graph_results = _state.graph.expand(
                    seed_ids, mode=expansion_mode, top_k=graph_budget
                )
            graph_ids = set(did for did, _ in graph_results)

            for did in base_ids:
                if did not in seen and len(final) < top_k - graph_budget:
                    seen.add(did)
                    final.append(did)
                    sources[did] = "hybrid"
            for did, _ in graph_results:
                if did not in seen and len(final) < top_k:
                    seen.add(did)
                    final.append(did)
                    sources[did] = "graph"
            for did in base_ids:
                if did not in seen and len(final) < top_k:
                    seen.add(did)
                    final.append(did)
                    sources[did] = "hybrid"

        for rank, did in enumerate(final):
            results_with_source.append(
                SearchResult(
                    doc_id=did,
                    score=round(1.0 / (rank + 1), 4),
                    source=sources.get(did, "hybrid"),
                )
            )
    else:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unknown mode: {mode!r}. Use 'dense', 'hybrid', or 'ahr'.",
        )

    return results_with_source


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness probe — always returns 200 if the process is alive."""
    return HealthResponse(
        status="ok",
        uptime_s=round(time.time() - _state.start_time, 1),
        indices={
            "dense": _state.dense is not None,
            "bm25": _state.bm25 is not None,
            "graph": _state.graph is not None,
            "reranker": _state.reranker is not None and _state.reranker.available,
        },
    )


@app.get("/ready", tags=["System"])
async def ready() -> JSONResponse:
    """
    Readiness probe — returns 200 when at least one index is loaded.
    Returns 503 during startup.
    """
    if _state.ready:
        return JSONResponse({"ready": True})
    return JSONResponse(
        {"ready": False, "message": "Indices are still loading"},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.get("/stats", tags=["System"])
async def stats() -> Dict:
    """Return index statistics and graph topology metrics."""
    graph_stats = _state.graph.get_stats() if _state.graph is not None else {}
    return {
        "ready": _state.ready,
        "uptime_s": round(time.time() - _state.start_time, 1),
        "bm25_docs": _state.bm25._N if _state.bm25 is not None else 0,
        "dense_docs": (
            _state.dense._index.ntotal
            if _state.dense is not None and _state.dense._index is not None
            else 0
        ),
        "graph": graph_stats,
        "reranker_available": _state.reranker is not None and _state.reranker.available,
    }


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(request: SearchRequest) -> SearchResponse:
    """
    Retrieve relevant documents for a query.

    Modes
    -----
    dense   : FAISS cosine similarity only
    hybrid  : FAISS + BM25 fused via RRF
    ahr     : Hybrid + Graph expansion with adaptive slot budget (default)
    """
    if not _state.ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Indices are still loading")

    async with _query_semaphore:
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            _retrieve,
            request.query,
            request.top_k,
            request.mode,
            request.expansion_mode,
        )

        # Optional reranking
        if request.rerank and _state.reranker and _state.reranker.available:
            candidates = [
                {"doc_id": r.doc_id, "text": _state.id_to_text.get(r.doc_id, "")}
                for r in results
            ]
            reranked = _state.reranker.rerank(request.query, candidates, top_k=request.top_k)
            results = [
                SearchResult(
                    doc_id=c["doc_id"],
                    score=round(c.get("rerank_score", 0), 4),
                    source="reranker",
                )
                for c in reranked
            ]

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "SEARCH  query=%r  mode=%s  k=%d  latency=%.1fms",
            request.query[:60],
            request.mode,
            request.top_k,
            latency_ms,
        )

    return SearchResponse(
        query=request.query,
        results=results,
        latency_ms=round(latency_ms, 2),
        mode=request.mode,
    )


@app.post("/query", response_model=QueryResponse, tags=["Generation"])
async def query_with_generation(request: QueryRequest) -> QueryResponse:
    """
    Retrieve documents and generate an answer using a local Ollama LLM.

    The context window is built from the top-k retrieved document snippets.
    Generation is streamed through Ollama but returned as a single response.
    """
    if not _state.ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Indices are still loading")

    async with _query_semaphore:
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()

        # Retrieve
        results = await loop.run_in_executor(
            None, _retrieve, request.query, request.top_k, request.mode, request.expansion_mode
        )

        # Build context
        ctx_parts = []
        for i, r in enumerate(results, start=1):
            text = _state.id_to_text.get(r.doc_id, "")
            ctx_parts.append(f"[{i}] {text[:600]}")
        context = "\n\n".join(ctx_parts)

        # Generate
        ollama_cfg = _state.config.get("llm", {})
        ollama_url = ollama_cfg.get("ollama_url", "http://localhost:11434")
        model = request.ollama_model or ollama_cfg.get("model", "deepseek-r1:7b")

        answer = await loop.run_in_executor(
            None, _generate_ollama, request.query, context, model, ollama_url
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "QUERY  query=%r  model=%s  latency=%.1fms", request.query[:60], model, latency_ms
        )

    return QueryResponse(
        query=request.query,
        answer=answer,
        sources=results,
        latency_ms=round(latency_ms, 2),
    )


def _generate_ollama(query: str, context: str, model: str, ollama_url: str) -> str:
    """Call Ollama /api/generate and return the full response string."""
    import requests

    prompt = (
        f"You are a research assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely and cite the context numbers [1], [2] etc. where relevant:"
    )
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        logger.error("Ollama generation failed: %s", exc)
        return f"[Generation error: {exc}]"
