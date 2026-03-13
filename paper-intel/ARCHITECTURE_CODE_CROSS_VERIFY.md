# Architecture vs Code Cross-Verification

Prepared: 2026-02-23

This report validates the provided system architecture against the current implementation.

## Executive Verdict

- Overall: **Partially aligned**
- Strongly implemented: Ingestion, PDF parsing/fallback, FAISS RAG, resource recommendation, API/UI
- Partially implemented: Hybrid retrieval and reranking (present in an alternate RAG path, not in main runtime path)
- Not implemented in runtime: Neo4j graph storage/query, GraphRAG context builder, dedicated monitoring dashboard

---

## Layer-by-Layer Verification

### 1) Ingestion Layer

**Status: Implemented**

Evidence:
- Multi-source orchestration: [ingestion/discovery/search_engine.py](ingestion/discovery/search_engine.py#L26)
- Open-source search path: [ingestion/discovery/search_engine.py](ingestion/discovery/search_engine.py#L62)
- Closed-access search path: [ingestion/discovery/search_engine.py](ingestion/discovery/search_engine.py#L158)
- Engine-level orchestration: [ingestion/ingestion_engine.py](ingestion/ingestion_engine.py#L17)
- Download fallback chain: [ingestion/discovery/pdf_downloader.py](ingestion/discovery/pdf_downloader.py#L1)

Matches diagram sources (OpenAlex, CORE, S2, CrossRef, Unpaywall, arXiv) and fallback downloader behavior.

---

### 2) Information Extraction Layer

**Status: Implemented (Phase-2 pipeline), not wired into main end-user runtime pipeline**

Evidence:
- Extraction orchestrator: [extraction/extraction_engine.py](extraction/extraction_engine.py#L24)
- Graffiti primary extraction: [extraction/graffiti_client.py](extraction/graffiti_client.py#L24)
- LLM fallback extraction: [extraction/llm_fallback.py](extraction/llm_fallback.py#L25)
- Schema normalization: [extraction/schema.py](extraction/schema.py#L183)

Note: Main interactive agent path in [sanshodhak_agent.py](sanshodhak_agent.py#L47) currently runs Search → Download → Parse → Embed → Retrieve → Recommend and does not invoke the Phase-2 extraction engine.

---

### 3) Knowledge Graph Layer (Neo4j Graph Engine)

**Status: Not implemented in runtime**

Evidence:
- Triples are generated as files, marked “Neo4j-ready”: [extraction/triple_builder.py](extraction/triple_builder.py#L2), [extraction/extraction_engine.py](extraction/extraction_engine.py#L5)
- No active Neo4j driver/session query code found in runtime pipeline.

Conclusion: KG artifacts are produced, but no persisted Neo4j graph service + Cypher retrieval path is wired.

---

### 4) Embedding + Dense Index Layer

**Status: Implemented (FAISS), metadata-store claim not matched**

Evidence:
- Main RAG builds FAISS index: [ollama_rag.py](ollama_rag.py#L64), [ollama_rag.py](ollama_rag.py#L101)
- Main agent embed stage: [sanshodhak_agent.py](sanshodhak_agent.py#L387)
- Enhanced path includes FAISS + BM25 state: [enhanced_rag.py](enhanced_rag.py#L144), [enhanced_rag.py](enhanced_rag.py#L202)

Mismatch vs diagram:
- Diagram says metadata store is Redis/SQLite. No Redis/SQLite implementation is present in the current runtime code path.

---

### 5) Hybrid Retrieval Engine

**Status: Partially implemented**

Evidence:
- Hybrid retrieval + BM25 + RRF + optional reranking is implemented in alternate module: [enhanced_rag.py](enhanced_rag.py#L253), [enhanced_rag.py](enhanced_rag.py#L281), [enhanced_rag.py](enhanced_rag.py#L294)
- Main runtime agent uses OllamaRAG dense retrieval path: [sanshodhak_agent.py](sanshodhak_agent.py#L445), [ollama_rag.py](ollama_rag.py#L132)

Conclusion: Hybrid engine exists, but main production agent path currently runs dense retrieval without BM25/rerank.

---

### 6) GraphRAG Context Builder

**Status: Not implemented in runtime**

Expected in diagram: select top entities, select subgraph, attach top-k chunks, build final context bundle.

Observed:
- No runtime component performing subgraph assembly/Cypher context composition.
- No GraphRAG context-bundle pipeline found in serving path.

---

### 7) LLM Reasoning Engine

**Status: Partially implemented**

Evidence:
- Answer generation exists in RAG modules: [ollama_rag.py](ollama_rag.py#L151), [enhanced_rag.py](enhanced_rag.py#L370)
- Query expansion exists in enhanced retrieval: [enhanced_rag.py](enhanced_rag.py#L230)
- Recommendation keyword generation via LLM exists: [resource_recommender_v2.py](resource_recommender_v2.py#L44)

Mismatch:
- Citation alignment/provenance tracing in the strict GraphRAG sense is limited; current output is chunk-source based citations, not graph-grounded trace chains.

---

### 8) Resource Recommendation Engine

**Status: Implemented**

Evidence:
- Enhanced recommender class: [resource_recommender_v2.py](resource_recommender_v2.py#L17)
- Main recommend stage in agent: [sanshodhak_agent.py](sanshodhak_agent.py#L495)
- API endpoint for recommendations: [sanshodhak_app.py](sanshodhak_app.py#L111)

---

### 9) API + UI Layer

**Status: Implemented**

Evidence:
- Health endpoint: [sanshodhak_app.py](sanshodhak_app.py#L57)
- Query endpoint: [sanshodhak_app.py](sanshodhak_app.py#L74)
- Full pipeline endpoint: [sanshodhak_app.py](sanshodhak_app.py#L142)
- Progress streaming over WebSocket: [sanshodhak_app.py](sanshodhak_app.py#L46)

---

### 10) Monitoring + Provenance Dashboard

**Status: Partially implemented (lightweight), no dedicated dashboard service**

Evidence:
- Stage progress state and status APIs exist in agent/web app: [sanshodhak_agent.py](sanshodhak_agent.py#L112), [sanshodhak_app.py](sanshodhak_app.py#L174)
- Evaluation scripts provide offline metrics, not live observability dashboard: [run_comprehensive_eval.py](run_comprehensive_eval.py#L205)

Mismatch:
- Diagram suggests dedicated ingestion/retrieval/lineage dashboard; implementation currently provides progress events and batch evaluation reports.

---

## Critical Mismatches to Fix Before Final Paper Figure

1. If claiming Neo4j/GraphRAG online retrieval, you need runtime graph persistence + Cypher query path.
2. If claiming hybrid retrieval in production, main agent should use enhanced hybrid retrieval path, not dense-only path.
3. If claiming Redis/SQLite metadata store, implement and wire it (or remove from figure).
4. If claiming dedicated monitoring dashboard, either implement one or relabel as “progress + evaluation reports”.

---

## Recommended Paper Wording (Safe)

Use this wording to stay accurate with current code:

- “The current production path implements a robust multi-source ingestion and dense-vector RAG pipeline with recommendation services.”
- “A hybrid BM25+rereanking retrieval module and graph-ready extraction pipeline are implemented as advanced components and are being integrated into the serving path.”
- “Knowledge graph artifacts are produced as Neo4j-ready triples; end-to-end graph retrieval integration is future work.”
